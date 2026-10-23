"""Shared doubles + seeding for index_mcp tests.

``StubProvider`` has hand-set geometry: every query embeds to
(1, 0), so a chunk's cosine score is exactly its vector's x
component — ranking assertions are arithmetic, not vibes. It also
records what it was asked to embed, which is how the tests prove
queries go through ``embed_query`` and never ``embed_documents``.

``build_index`` seeds a real corpus root through the real
repositories (CorpusStore + EmbedStore, whichever vector backend
the machine probes) so the read store is tested against the
writer's actual artifacts.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from school_index.corpus_sync.domain import CorpusItem, ItemStatus
from school_index.corpus_sync.store import CorpusStore
from school_index.embed_index.domain import Chunk, EmbedState
from school_index.embed_index.providers import Vector
from school_index.embed_index.store import EmbedStore


class StubProvider:
    """Deterministic 2-d provider with recorded calls. Pattern: test double."""

    model_id = "stub-2d"
    dim = 2

    def __init__(self) -> None:
        self.queries_seen: list[str] = []
        self.documents_seen: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        self.documents_seen.extend(texts)
        return [(0.0, 1.0) for _ in texts]

    def embed_query(self, text: str) -> Vector:
        self.queries_seen.append(text)
        return (1.0, 0.0)


@dataclass(frozen=True, kw_only=True)
class SeedChunk:
    """One chunk to seed: text, heading, and its exact vector."""

    text: str
    heading: str | None
    vector: Vector


@dataclass(frozen=True, kw_only=True)
class SeedItem:
    """One item to seed, with its chunks."""

    url: str
    title: str
    course: str | None
    chunks: tuple[SeedChunk, ...]
    status: ItemStatus = ItemStatus.OK
    model_id: str = StubProvider.model_id


def build_index(root: Path, items: Sequence[SeedItem]) -> None:
    """Seed ``root`` with ``items`` through the real write stores."""
    corpus = CorpusStore.open(root)
    embed = EmbedStore.open(root, dim=StubProvider.dim)
    for seed in items:
        source_ref = f"sam:{seed.url}"
        item = CorpusItem(
            id=hashlib.sha256(source_ref.encode("utf-8")).hexdigest(),
            source_kind="AGENDA",
            source_ref=source_ref,
            student="sam",
            course=seed.course,
            title=seed.title,
            deep_link=seed.url,
            origin_url=seed.url,
            format="pdf",
            content_hash="hash-1",
            fetched_at="2026-10-23T13:00:00+00:00",
            status=seed.status,
            error=None,
        )
        corpus.upsert_item(item)
        if seed.chunks:
            embed.replace_item_chunks(
                chunks=[
                    Chunk(
                        item_id=item.id,
                        ordinal=ordinal,
                        heading=chunk.heading,
                        text=chunk.text,
                    )
                    for ordinal, chunk in enumerate(seed.chunks)
                ],
                vectors=[chunk.vector for chunk in seed.chunks],
                state=EmbedState(
                    item_id=item.id,
                    source_hash="hash-1",
                    model_id=seed.model_id,
                    chunk_count=len(seed.chunks),
                    embedded_at="2026-10-23T14:00:00+00:00",
                ),
            )


#: A small geometry with known orderings (query vector is (1, 0)):
#: deck-a best 1.0 / 2 hits, ws-c best 0.9 / 3 hits, sci-b best 0.0.
DECK_A = SeedItem(
    url="https://example.com/deck-a",
    title="Pythagorean Deck",
    course="Algebra I",
    chunks=(
        SeedChunk(
            text="Model Pythagorean theorem: a^2 + b^2 = c^2 worked out.",
            heading="page 3",
            vector=(1.0, 0.0),
        ),
        SeedChunk(
            text="Pythagorean practice problems.",
            heading="page 4",
            vector=(0.8, 0.6),
        ),
    ),
)
SCI_B = SeedItem(
    url="https://example.com/sci-b",
    title="Photosynthesis Notes",
    course="Science",
    chunks=(
        SeedChunk(
            text="Photosynthesis turns sunlight into sugars.",
            heading="page 1",
            vector=(0.0, 1.0),
        ),
    ),
)
WS_C = SeedItem(
    url="https://example.com/ws-c",
    title="Algebra Worksheet",
    course="Algebra I",
    chunks=(
        SeedChunk(text="Triangles drill one.", heading=None, vector=(0.9, 0.4359)),
        SeedChunk(text="Triangles drill two.", heading=None, vector=(0.85, 0.5268)),
        SeedChunk(text="Triangles drill three.", heading=None, vector=(0.7, 0.7141)),
    ),
)
