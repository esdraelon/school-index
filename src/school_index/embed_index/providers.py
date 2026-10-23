"""Embedding providers: one strategy per model, registry-dispatched.

Pattern: Strategy + Registry. An ``EmbeddingProvider`` turns text
into fixed-dimension, L2-normalized vectors; the runner and the
search path depend only on the protocol, and the composition root
selects through ``build_providers`` — there is no provider if/else
anywhere (coding standard). Bedrock slots in as a second registry
entry when M5's benchmark needs it.

The protocol separates ``embed_documents`` from ``embed_query``
because bge models are asymmetric: queries carry an instruction
prefix ("Represent this sentence for searching relevant
passages: "), passages do not. Encoding that in the API shape —
rather than a boolean flag at call sites — is what keeps M4/M5
from silently embedding queries as passages.

The v1 provider runs BAAI/bge-small-en-v1.5 (384-dim) directly on
onnxruntime + tokenizers: no torch, no fastembed — the lean route,
with aarch64 wheels for hal9k. Inference is CLS-pooled and
L2-normalized per the model's card. Model files live under the
models dir (``<models>/bge-small-en-v1.5/{model.onnx,tokenizer.json}``),
fetched once by ``download_model``; runs are offline after that.
Session/tokenizer construction is injectable (``session_factory`` /
``tokenizer_factory``) so the embedding math is unit-testable
without the real model files.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
from numpy.typing import NDArray

#: The bge query instruction prefix. Passages never carry it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

#: Hugging Face repo + the two files the direct-ONNX route needs.
HF_REPO = "BAAI/bge-small-en-v1.5"
MODEL_SUBDIR = "bge-small-en-v1.5"
_MODEL_FILES: tuple[tuple[str, str], ...] = (
    ("onnx/model.onnx", "model.onnx"),
    ("tokenizer.json", "tokenizer.json"),
)

_MAX_TOKENS = 512
_BATCH = 32

Vector = tuple[float, ...]


class ModelNotDownloaded(RuntimeError):
    """The model's files are not on disk (run `embed-index fetch-model`)."""


class EmbeddingProvider(Protocol):
    """One embedding model + how to encode documents and queries."""

    model_id: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Encode passages (no instruction prefix)."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Encode one query (instruction prefix, for bge-style models)."""
        ...


def _normalize(rows: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (rows / norms).astype(np.float32)


class OnnxBgeProvider:
    """BAAI/bge-small-en-v1.5 on onnxruntime. Pattern: Strategy.

    Lazy: the session and tokenizer load on first embed, so wiring
    the registry is cheap and a missing model fails at use with a
    precise error, not at process start.
    """

    model_id = "onnx-bge-small"
    dim = 384

    def __init__(
        self,
        *,
        models_dir: Path,
        session_factory: Callable[[Path], object] | None = None,
        tokenizer_factory: Callable[[Path], object] | None = None,
    ) -> None:
        self._model_dir = models_dir / MODEL_SUBDIR
        self._session_factory = session_factory
        self._tokenizer_factory = tokenizer_factory
        self._session: object | None = None
        self._tokenizer: object | None = None

    # -- protocol ------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(list(texts), query=False)

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], query=True)[0]

    # -- internals -----------------------------------------------------

    def _embed(self, texts: list[str], *, query: bool) -> list[Vector]:
        if not texts:
            return []
        session = self._load_session()
        tokenizer = self._load_tokenizer()
        if query:
            texts = [QUERY_PREFIX + text for text in texts]
        vectors: list[Vector] = []
        for start in range(0, len(texts), _BATCH):
            batch = texts[start : start + _BATCH]
            encodings = tokenizer.encode_batch(batch)  # type: ignore[attr-defined]
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            token_type = np.zeros_like(ids)
            outputs = session.run(  # type: ignore[attr-defined]
                None,
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "token_type_ids": token_type,
                },
            )
            pooled = _normalize(np.asarray(outputs[0][:, 0, :], dtype=np.float32))
            vectors.extend(tuple(float(x) for x in row) for row in pooled)
        return vectors

    def _model_file(self, name: str) -> Path:
        path = self._model_dir / name
        if not path.is_file():
            raise ModelNotDownloaded(
                f"{name} not found under {self._model_dir} — "
                "run `python -m school_index.embed_index fetch-model` first"
            )
        return path

    def _load_session(self) -> object:
        if self._session is None:
            path = self._model_file("model.onnx")
            if self._session_factory is not None:
                self._session = self._session_factory(path)
            else:
                import onnxruntime

                self._session = onnxruntime.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
        return self._session

    def _load_tokenizer(self) -> object:
        if self._tokenizer is None:
            path = self._model_file("tokenizer.json")
            if self._tokenizer_factory is not None:
                self._tokenizer = self._tokenizer_factory(path)
            else:
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(path))
                tokenizer.enable_padding()
                tokenizer.enable_truncation(max_length=_MAX_TOKENS)
                self._tokenizer = tokenizer
        return self._tokenizer


def model_dir(models_dir: Path) -> Path:
    """Where the bge-small files live under a models root."""
    return models_dir / MODEL_SUBDIR


def model_present(models_dir: Path) -> bool:
    """True when both model files are on disk under ``models_dir``."""
    return all((model_dir(models_dir) / dest).is_file() for _src, dest in _MODEL_FILES)


def download_model(models_dir: Path, *, client: httpx.Client | None = None) -> Path:
    """Fetch the model files from Hugging Face into ``models_dir``.

    Idempotent (existing files are left alone) and atomic per file
    (temp + rename), so an interrupted fetch never leaves a torn
    model behind. Returns the model directory.
    """
    target = model_dir(models_dir)
    target.mkdir(parents=True, exist_ok=True)
    http = client or httpx.Client(follow_redirects=True, timeout=600.0)
    for source, dest in _MODEL_FILES:
        path = target / dest
        if path.is_file():
            continue
        url = f"https://huggingface.co/{HF_REPO}/resolve/main/{source}"
        temporary = target / f"{dest}.tmp"
        with http.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for block in response.iter_bytes():
                    handle.write(block)
        temporary.replace(path)
    return target


def build_providers(*, models_dir: Path) -> Mapping[str, EmbeddingProvider]:
    """The production registry: model key → strategy.

    v1 has exactly one arm — the design's ONNX default. The Bedrock
    arm joins this mapping at M5 (benchmark), selected by config,
    never by a branch in the runner.
    """
    return {"onnx-bge-small": OnnxBgeProvider(models_dir=models_dir)}
