"""Unit tests: the embedding provider strategy + model download.

The ONNX provider's math (CLS pooling, L2 normalization, batching,
the query-prefix asymmetry) is tested through injected fake
session/tokenizer factories — the real model files are exercised
by the skip-gated integration test at the bottom and by the live
hal9k build, not by the unit suite.
"""

import math
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from school_index.embed_index.providers import (
    QUERY_PREFIX,
    ModelNotDownloaded,
    OnnxBgeProvider,
    build_providers,
    download_model,
    model_present,
)


class FakeTokenizer:
    """Records what it was asked to encode; emits fixed-length ids."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode_batch(self, texts: Sequence[str]) -> list[SimpleNamespace]:
        self.seen.extend(texts)
        return [
            SimpleNamespace(ids=[101, 2000 + i, 102], attention_mask=[1, 1, 1])
            for i, _text in enumerate(texts)
        ]


class FakeSession:
    """Deterministic hidden states derived from the input ids."""

    def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        ids = feeds["input_ids"]
        batch, seq = ids.shape
        hidden = np.full((batch, seq, OnnxBgeProvider.dim), 9.0, dtype=np.float32)
        for row in range(batch):
            hidden[row, 0, :] = 0.0
            hidden[row, 0, 0] = 3.0
            hidden[row, 0, 1] = 4.0
        return [hidden]


def wired_provider(tmp_path: Path) -> tuple[OnnxBgeProvider, FakeTokenizer]:
    model_dir = tmp_path / "bge-small-en-v1.5"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"fake")
    (model_dir / "tokenizer.json").write_text("{}")
    tokenizer = FakeTokenizer()
    provider = OnnxBgeProvider(
        models_dir=tmp_path,
        session_factory=lambda _path: FakeSession(),
        tokenizer_factory=lambda _path: tokenizer,
    )
    return provider, tokenizer


def test_registry_has_the_onnx_arm() -> None:
    providers = build_providers(models_dir=Path("/nowhere"))
    assert set(providers) == {"onnx-bge-small"}
    assert providers["onnx-bge-small"].dim == 384


def test_missing_model_fails_with_precise_error(tmp_path: Path) -> None:
    provider = OnnxBgeProvider(models_dir=tmp_path)
    with pytest.raises(ModelNotDownloaded, match="fetch-model"):
        provider.embed_documents(["anything"])


def test_embed_documents_returns_normalized_cls_vectors(tmp_path: Path) -> None:
    provider, _tokenizer = wired_provider(tmp_path)
    vectors = provider.embed_documents(["first", "second"])
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
    for vector in vectors:
        assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0)
    # The fake session makes the CLS row (3, 4, rest 0-ish); pooling
    # must take position 0, so the direction is (0.6, 0.8, 0, ...).
    assert vectors[0][0] == pytest.approx(0.6)
    assert vectors[0][1] == pytest.approx(0.8)


def test_query_carries_the_instruction_prefix_documents_do_not(
    tmp_path: Path,
) -> None:
    provider, tokenizer = wired_provider(tmp_path)
    provider.embed_documents(["a passage"])
    provider.embed_query("a question")
    assert tokenizer.seen[0] == "a passage"
    assert tokenizer.seen[1] == QUERY_PREFIX + "a question"


def test_embed_documents_of_nothing_is_empty(tmp_path: Path) -> None:
    provider, _tokenizer = wired_provider(tmp_path)
    assert provider.embed_documents([]) == []


def test_download_model_fetches_both_files_atomically(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=f"bytes for {request.url.path}".encode())

    client = httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=True
    )
    target = download_model(tmp_path, client=client)
    assert model_present(tmp_path)
    assert (target / "model.onnx").read_bytes().endswith(b"onnx/model.onnx")
    assert (target / "tokenizer.json").read_bytes().endswith(b"tokenizer.json")
    # Idempotent: a second call keeps the same bytes (no re-fetch).
    download_model(tmp_path, client=client)
    assert (target / "model.onnx").read_bytes().endswith(b"onnx/model.onnx")


# -- real model (skip-gated) --------------------------------------------
#
# Set EMBED_TEST_MODEL_DIR to a models root holding bge-small-en-v1.5
# (see `embed-index fetch-model`) to run this. On machines without
# the model it skips with the reason recorded; the live hal9k build
# is the standing real-model evidence.

_MODEL_DIR = os.environ.get("EMBED_TEST_MODEL_DIR")


@pytest.mark.skipif(
    not _MODEL_DIR or not model_present(Path(_MODEL_DIR or "")),
    reason="bge-small model not downloaded (set EMBED_TEST_MODEL_DIR)",
)
def test_real_bge_model_ranks_the_relevant_passage_nearer() -> None:
    assert _MODEL_DIR is not None
    provider = OnnxBgeProvider(models_dir=Path(_MODEL_DIR))
    query = provider.embed_query("Pythagorean theorem practice")
    relevant, unrelated = provider.embed_documents(
        [
            "The Pythagorean theorem: a squared plus b squared equals c squared.",
            "Photosynthesis converts sunlight and carbon dioxide into oxygen.",
        ]
    )
    assert len(query) == 384

    def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return sum(x * y for x, y in zip(a, b))

    assert cosine(query, relevant) > cosine(query, unrelated)
