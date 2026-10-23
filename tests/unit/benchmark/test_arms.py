"""Unit tests: arm construction (production / throwaway / blocked).

The fixture corpus is built with the production pieces (EmbedStore,
the real chunker, the deterministic fake provider), so the arms are
tested against the same shapes they meet live — only the provider's
vectors are fake, and retrieval *quality* is the live benchmark's
job, not this suite's.
"""

import sqlite3
from pathlib import Path

import pytest

from school_index.benchmark import corpus_read
from school_index.benchmark.arms import (
    ARM_SPECS,
    ArmBlocked,
    build_arm,
    build_arm_providers,
    chunking_label,
)
from school_index.benchmark.config import BenchmarkConfig
from school_index.benchmark.domain import CorpusItemRow
from school_index.corpus_sync.lookup import Found
from school_index.embed_index.chunker import chunk_text
from school_index.embed_index.domain import EmbedState
from school_index.embed_index.store import EmbedStore
from school_index.index_mcp.store import IndexReadStore
from tests.unit.benchmark.fakes import RaisingProvider
from tests.unit.embed_index.fakes import FakeProvider

_SENTENCE = "The Pythagorean theorem relates the legs of a right triangle. "
_PAGE = _SENTENCE * 7  # ~434 chars: packs 2-per-chunk at 1000, singly at 600
LONG_TEXT = f"[[page 1]]\n{_PAGE}\n[[page 2]]\n{_PAGE}\n[[page 3]]\n{_PAGE}"
SHORT_TEXT = "[[page 1]]\nSlope intercept form word problems. y = mx + b."


def _item(item_id: str, *, title: str, status: str = "ok") -> CorpusItemRow:
    return CorpusItemRow(
        id=item_id,
        source_kind="AGENDA",
        source_ref=f"ref-{item_id}",
        student="sam",
        course="Math 8",
        title=title,
        deep_link=f"https://example.com/{item_id}",
        origin_url=f"https://example.com/{item_id}",
        format="slides",
        content_hash=f"hash-{item_id}",
        fetched_at="2026-10-23T00:00:00Z",
        status=status,
        error=None,
    )


def _insert_items(db_path: Path, items: list[CorpusItemRow]) -> None:
    connection = sqlite3.connect(db_path)
    connection.executemany(
        "INSERT INTO corpus_items (id, source_kind, source_ref, student,"
        " course, title, deep_link, origin_url, format, content_hash,"
        " fetched_at, status, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                i.id, i.source_kind, i.source_ref, i.student, i.course, i.title,
                i.deep_link, i.origin_url, i.format, i.content_hash,
                i.fetched_at, i.status, i.error,
            )
            for i in items
        ],
    )
    connection.commit()
    connection.close()


def _insert_text(db_path: Path, item_id: str, text: str) -> None:
    connection = sqlite3.connect(db_path)
    connection.execute(
        "INSERT INTO item_text (item_id, text, extractor, extracted_at)"
        " VALUES (?, ?, 'test', '2026-10-23T00:00:00Z')",
        (item_id, text),
    )
    connection.commit()
    connection.close()


def _build_prod_corpus(root: Path, provider: FakeProvider) -> list[CorpusItemRow]:
    """A two-item production corpus (+ one failed item), really built."""
    root.mkdir(parents=True, exist_ok=True)
    store = EmbedStore.open(root, dim=provider.dim)
    items = [
        _item("item-a", title="Slope Intercept"),
        _item("item-b", title="Pythagorean"),
        _item("item-c", title="Retracted", status="failed"),
    ]
    _insert_items(root / "corpus.db", items)
    for item, text in (("item-a", SHORT_TEXT), ("item-b", LONG_TEXT)):
        _insert_text(root / "corpus.db", item, text)
    for item_id, text in (("item-a", SHORT_TEXT), ("item-b", LONG_TEXT)):
        chunks = chunk_text(text, item_id=item_id)
        store.replace_item_chunks(
            chunks=chunks,
            vectors=provider.embed_documents([c.text for c in chunks]),
            state=EmbedState(
                item_id=item_id,
                source_hash=f"hash-{item_id}",
                model_id=provider.model_id,
                chunk_count=len(chunks),
                embedded_at="2026-10-23T00:00:00Z",
            ),
        )
    return items


def _config(tmp_path: Path, corpus_root: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        corpus_root=corpus_root,
        benchmark_root=tmp_path / "bench",
        models_dir=tmp_path / "models",
        queries_path=tmp_path / "queries.json",
    )


def test_arm_registry_shape() -> None:
    assert set(ARM_SPECS) == {"onnx-production", "bedrock-titan", "onnx-chunk600"}
    assert chunking_label(ARM_SPECS["onnx-production"]).startswith("production (")
    assert chunking_label(ARM_SPECS["onnx-chunk600"]).startswith("variant (")
    assert "target_chars=600" in chunking_label(ARM_SPECS["onnx-chunk600"])


def test_benchmark_registry_adds_bedrock_to_productions() -> None:
    providers = build_arm_providers(models_dir=Path("/nowhere"))
    assert set(providers) == {"onnx-bge-small", "bedrock-titan"}
    assert providers["bedrock-titan"].model_id == "bedrock-titan-embed-v2"


def test_production_arm_serves_the_live_corpus(tmp_path: Path) -> None:
    provider = FakeProvider()
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, provider)
    arm = build_arm(
        ARM_SPECS["onnx-production"],
        config=_config(tmp_path, prod),
        providers={"onnx-bge-small": provider},
        now=lambda: "2026-10-23T00:00:00Z",
    )
    assert arm.provider_model_id == "fake-hash"
    hits = arm.service.search_passages("pythagorean theorem", limit=3)
    assert hits  # the fixture index answers (quality is not asserted)
    assert all(hit.item_id in {"item-a", "item-b"} for hit in hits)


def test_bedrock_arm_blocked_verbatim_when_provider_fails(tmp_path: Path) -> None:
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, FakeProvider())
    with pytest.raises(ArmBlocked) as excinfo:
        build_arm(
            ARM_SPECS["bedrock-titan"],
            config=_config(tmp_path, prod),
            providers={"bedrock-titan": RaisingProvider()},
            now=lambda: "2026-10-23T00:00:00Z",
        )
    assert "Unable to locate credentials" in excinfo.value.reason


def test_unregistered_provider_blocks_with_the_registry_named(
    tmp_path: Path,
) -> None:
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, FakeProvider())
    with pytest.raises(ArmBlocked, match="not registered"):
        build_arm(
            ARM_SPECS["bedrock-titan"],
            config=_config(tmp_path, prod),
            providers={},
            now=lambda: "2026-10-23T00:00:00Z",
        )


def test_bedrock_arm_builds_a_throwaway_with_production_chunks(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, provider)
    arm = build_arm(
        ARM_SPECS["bedrock-titan"],
        config=_config(tmp_path, prod),
        providers={"bedrock-titan": provider},
        now=lambda: "2026-10-23T00:00:00Z",
    )
    arm_root = tmp_path / "bench" / "benchmark-arms" / "bedrock-titan"
    # Same chunking as production (the arms differ only in provider).
    assert len(corpus_read.read_chunks(arm_root, item_id="item-b")) == len(
        corpus_read.read_chunks(prod, item_id="item-b")
    )
    # The failed item was copied but never embedded.
    assert corpus_read.read_chunks(arm_root, item_id="item-c") == []
    census = IndexReadStore.open(arm_root, dim=provider.dim).index_model_id()
    assert isinstance(census, Found) and census.value == "fake-hash"
    assert arm.service.search_passages("slope", limit=2)


def test_provider_failing_mid_build_blocks_the_arm(tmp_path: Path) -> None:
    """A provider that probes fine but fails on documents (throttling,
    a revoked key) yields a blocked arm, not a crashed benchmark."""

    class ProbeOnlyProvider(FakeProvider):
        def embed_documents(self, texts):  # type: ignore[override]
            raise RuntimeError("ThrottlingException: rate exceeded")

    prod = tmp_path / "prod"
    _build_prod_corpus(prod, FakeProvider())
    with pytest.raises(ArmBlocked, match="ThrottlingException"):
        build_arm(
            ARM_SPECS["bedrock-titan"],
            config=_config(tmp_path, prod),
            providers={"bedrock-titan": ProbeOnlyProvider()},
            now=lambda: "2026-10-23T00:00:00Z",
        )


def test_chunk600_arm_rechunks_from_text(tmp_path: Path) -> None:
    provider = FakeProvider()
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, provider)
    build_arm(
        ARM_SPECS["onnx-chunk600"],
        config=_config(tmp_path, prod),
        providers={"onnx-bge-small": provider},
        now=lambda: "2026-10-23T00:00:00Z",
    )
    arm_root = tmp_path / "bench" / "benchmark-arms" / "onnx-chunk600"
    # ~434-char pages: 2 chunks at the 1000 target, 3 at 600.
    assert len(corpus_read.read_chunks(prod, item_id="item-b")) == 2
    assert len(corpus_read.read_chunks(arm_root, item_id="item-b")) == 3
