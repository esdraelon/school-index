"""Arms: what varies between benchmark runs, and how each is built.

Pattern: Strategy + Registry + Composition Root. An arm is a
(provider, chunking) pair evaluated over the same labeled set
through the same production search path (SearchService +
IndexReadStore) — arms differ in their inputs, never in the code
that ranks:

- ``onnx-production`` — the live path untouched: production chunks,
  the production registry's ONNX provider, the production corpus.
- ``bedrock-titan`` — production chunking, Bedrock embeddings, a
  throwaway index (the quality-benchmark candidate from the RAG
  evaluation).
- ``onnx-chunk600`` — a 600-char-target chunking variant, ONNX
  embeddings, a throwaway index. Built only when the production
  arm's pointer accuracy looks chunk-bound (the CLI decides from
  arm A's recorded numbers) or when forced.

Throwaway arms get a private corpus root under
``BENCHMARK_ROOT/benchmark-arms/<arm-key>``: item rows are copied
from production (read-only), chunks are (re)built per the arm, and
embeddings come from the arm's provider. The directory is deleted
and rebuilt on every arm build — it is an index, not evidence;
the evidence is the recorded outcomes in benchmark.db.

An arm that cannot run raises ``ArmBlocked`` with the reason
verbatim (a missing model, boto3's NoCredentialsError, ...): the
runner records blocked as a first-class result.
"""

import shutil
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from school_index.benchmark import corpus_read
from school_index.benchmark.config import BenchmarkConfig
from school_index.benchmark.domain import CorpusItemRow
from school_index.embed_index.chunker import DEFAULT_CONFIG, ChunkConfig, chunk_text
from school_index.embed_index.domain import Chunk, EmbedState
from school_index.embed_index.providers import (
    BedrockTitanProvider,
    EmbeddingProvider,
    build_providers,
)
from school_index.embed_index.store import EmbedStore
from school_index.index_mcp.queries import SearchService
from school_index.index_mcp.store import IndexReadStore


class ArmBlocked(RuntimeError):
    """The arm cannot run here; ``reason`` is recorded verbatim."""

    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, kw_only=True)
class ArmSpec:
    """One arm's recipe: which provider, which chunking."""

    key: str
    provider_key: str
    chunk_config: ChunkConfig | None  # None = production chunks as-is
    throwaway: bool


#: The arm registry. Selection is by key from this mapping — the
#: runner and CLI contain no per-arm branches.
ARM_SPECS: Mapping[str, ArmSpec] = {
    "onnx-production": ArmSpec(
        key="onnx-production",
        provider_key="onnx-bge-small",
        chunk_config=None,
        throwaway=False,
    ),
    "bedrock-titan": ArmSpec(
        key="bedrock-titan",
        provider_key="bedrock-titan",
        chunk_config=None,
        throwaway=True,
    ),
    "onnx-chunk600": ArmSpec(
        key="onnx-chunk600",
        provider_key="onnx-bge-small",
        chunk_config=ChunkConfig(target_chars=600, max_chars=840, overlap_chars=120),
        throwaway=True,
    ),
}


@dataclass(frozen=True, kw_only=True)
class BuiltArm:
    """A ready-to-run arm: the production search service + labels."""

    spec: ArmSpec
    service: SearchService
    provider_model_id: str
    chunking_label: str


def build_arm_providers(*, models_dir: Path) -> Mapping[str, EmbeddingProvider]:
    """The benchmark's provider registry: production's + Bedrock.

    The production registry (build_providers) is pinned ONNX-only
    by its own tests — serving must not be able to pick a paid
    cloud arm by config accident — so the composition happens here,
    at the benchmark's composition root.
    """
    return {
        **build_providers(models_dir=models_dir),
        "bedrock-titan": BedrockTitanProvider(),
    }


def chunking_label(spec: ArmSpec) -> str:
    """The arm's chunking, as recorded on the run (concrete values)."""
    config = spec.chunk_config or DEFAULT_CONFIG
    origin = "production" if spec.chunk_config is None else "variant"
    return (
        f"{origin} (target_chars={config.target_chars}, "
        f"max_chars={config.max_chars}, overlap_chars={config.overlap_chars})"
    )


def _probe(provider: EmbeddingProvider) -> None:
    """One throwaway embed: surfaces missing models/credentials."""
    try:
        provider.embed_query("benchmark arm probe")
    except Exception as exc:
        raise ArmBlocked(reason=f"{type(exc).__name__}: {exc}"[:400]) from exc


def _copy_items(db_path: Path, items: list[CorpusItemRow]) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.executemany(
            """
            INSERT INTO corpus_items (
                id, source_kind, source_ref, student, course, title,
                deep_link, origin_url, format, content_hash, fetched_at,
                status, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.id,
                    item.source_kind,
                    item.source_ref,
                    item.student,
                    item.course,
                    item.title,
                    item.deep_link,
                    item.origin_url,
                    item.format,
                    item.content_hash,
                    item.fetched_at,
                    item.status,
                    item.error,
                )
                for item in items
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _arm_chunks(
    spec: ArmSpec, *, corpus_root: Path, item: CorpusItemRow
) -> list[Chunk]:
    if spec.chunk_config is None:
        return [
            Chunk(
                item_id=stored.item_id,
                ordinal=stored.ordinal,
                heading=stored.heading,
                text=stored.text,
            )
            for stored in corpus_read.read_chunks(corpus_root, item_id=item.id)
        ]
    text = corpus_read.read_text(corpus_root, item_id=item.id)
    if text is None:
        return []
    return chunk_text(text, item_id=item.id, config=spec.chunk_config)


def _build_throwaway(
    spec: ArmSpec,
    *,
    provider: EmbeddingProvider,
    config: BenchmarkConfig,
    now: Callable[[], str],
) -> SearchService:
    arm_root = config.benchmark_root / "benchmark-arms" / spec.key
    if arm_root.exists():
        shutil.rmtree(arm_root)
    arm_root.mkdir(parents=True)
    store = EmbedStore.open(arm_root, dim=provider.dim)
    items = corpus_read.read_items(config.corpus_root)
    _copy_items(arm_root / "corpus.db", items)
    for item in items:
        if item.status != "ok":
            continue
        chunks = _arm_chunks(spec, corpus_root=config.corpus_root, item=item)
        try:
            vectors = provider.embed_documents([chunk.text for chunk in chunks])
        except Exception as exc:
            # A provider that probes fine can still fail mid-build
            # (throttling, a revoked key): that is a blocked arm,
            # recorded verbatim — not a crashed benchmark.
            raise ArmBlocked(reason=f"{type(exc).__name__}: {exc}"[:400]) from exc
        store.replace_item_chunks(
            chunks=chunks,
            vectors=vectors,
            state=EmbedState(
                item_id=item.id,
                source_hash=item.content_hash,
                model_id=provider.model_id,
                chunk_count=len(chunks),
                embedded_at=now(),
            ),
        )
    read_store = IndexReadStore.open(arm_root, dim=provider.dim)
    return SearchService(store=read_store, provider=provider)


def build_arm(
    spec: ArmSpec,
    *,
    config: BenchmarkConfig,
    providers: Mapping[str, EmbeddingProvider],
    now: Callable[[], str],
) -> BuiltArm:
    """Build one arm's search service, or raise ArmBlocked."""
    provider = providers.get(spec.provider_key)
    if provider is None:
        raise ArmBlocked(
            reason=(
                f"provider {spec.provider_key!r} is not registered "
                f"(registered: {sorted(providers)})"
            )
        )
    _probe(provider)
    if spec.throwaway:
        service = _build_throwaway(
            spec, provider=provider, config=config, now=now
        )
    else:
        read_store = IndexReadStore.open(config.corpus_root, dim=provider.dim)
        service = SearchService(store=read_store, provider=provider)
    return BuiltArm(
        spec=spec,
        service=service,
        provider_model_id=provider.model_id,
        chunking_label=chunking_label(spec),
    )
