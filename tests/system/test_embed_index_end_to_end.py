"""System test: one full embed pass, end to end.

Everything is real — the runner, the chunker, both repositories,
the embed store (whichever vector backend this machine probes) —
over an on-disk corpus store seeded through the real M1/M2
repositories. The provider is a deterministic hash fake: embedding
*quality* is the real model's business (proven on hal9k), pipeline
*behavior* is what's under test here. Assertions read corpus.db
with raw SQL, independently of the repositories that wrote it.
"""

import hashlib
import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from school_index.corpus_sync.domain import CorpusItem, ItemStatus
from school_index.corpus_sync.store import CorpusStore
from school_index.doc_extract.domain import ExtractionOutcome, ExtractionState
from school_index.doc_extract.store import ExtractStore
from school_index.embed_index.providers import Vector
from school_index.embed_index.runner import EmbedRunner
from school_index.embed_index.store import EmbedStore, load_vec


class FakeProvider:
    model_id = "fake-hash"
    dim = 16

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [_vector(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return _vector(text)


def _vector(text: str) -> Vector:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [byte / 127.5 - 1.0 for byte in digest[:16]]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return tuple(x / norm for x in raw)


def seed_item(store: CorpusStore, url: str, **overrides: object) -> CorpusItem:
    source_ref = f"sam:{url}"
    fields: dict[str, object] = {
        "id": hashlib.sha256(source_ref.encode("utf-8")).hexdigest(),
        "source_kind": "AGENDA",
        "source_ref": source_ref,
        "student": "sam",
        "course": "Algebra I",
        "title": "Item",
        "deep_link": url,
        "origin_url": url,
        "format": "pdf",
        "content_hash": "hash-1",
        "fetched_at": "2026-10-23T13:00:00+00:00",
        "status": ItemStatus.OK,
        "error": None,
    }
    fields.update(overrides)
    item = CorpusItem(**fields)  # type: ignore[arg-type]
    store.upsert_item(item)
    return item


def seed_extraction(
    store: ExtractStore,
    item: CorpusItem,
    *,
    text: str | None,
    outcome: ExtractionOutcome = ExtractionOutcome.EXTRACTED,
) -> None:
    store.record_outcome(
        ExtractionState(
            item_id=item.id,
            outcome=outcome,
            extractor=(
                None if outcome is ExtractionOutcome.NOT_EXTRACTABLE else "pypdf"
            ),
            source_hash=item.content_hash,
            format=item.format,
            char_count=len(text) if text else 0,
            page_count=None,
            error=None,
            extracted_at="2026-10-23T14:00:00+00:00",
        ),
        text=text,
    )


def test_full_embed_noop_change_and_tombstone(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    corpus = CorpusStore.open(root)
    extracts = ExtractStore.open(root)
    embeds = EmbedStore.open(root, dim=FakeProvider.dim)

    deck = seed_item(corpus, "https://example.com/deck")
    seed_extraction(
        extracts,
        deck,
        text="[[page 1]]\nSystems of equations.\n[[page 2]]\nGraphing lines.",
    )
    blank = seed_item(corpus, "https://example.com/scan")
    seed_extraction(extracts, blank, text="", outcome=ExtractionOutcome.EMPTY)
    link = seed_item(corpus, "https://example.com/ixl", format="external-link",
                     content_hash=None)
    seed_extraction(
        extracts, link, text=None, outcome=ExtractionOutcome.NOT_EXTRACTABLE
    )

    runner = EmbedRunner(
        corpus_store=corpus,
        extract_store=extracts,
        embed_store=embeds,
        provider=FakeProvider(),
    )
    report = runner.run()
    assert (report.total, report.embedded, report.not_embeddable) == (3, 2, 1)
    assert report.chunks_total == 1  # the two small pages pack into one chunk

    connection = sqlite3.connect(root / "corpus.db")
    load_vec(connection)  # the embeddings table is vec0 on this machine
    chunk_rows = connection.execute(
        "SELECT item_id, ordinal, heading, text FROM chunks"
    ).fetchall()
    assert chunk_rows == [
        (deck.id, 0, "page 1", "Systems of equations.\nGraphing lines.")
    ]
    states = dict(
        connection.execute("SELECT item_id, chunk_count FROM embed_state").fetchall()
    )
    assert states == {deck.id: 1, blank.id: 0}
    vector_count = connection.execute(
        "SELECT COUNT(*) FROM embeddings"
    ).fetchone()[0]
    assert vector_count == 1
    health = connection.execute(
        "SELECT last_status FROM sync_state WHERE source = 'embed-index'"
    ).fetchone()
    assert health == ("ok",)

    # The index answers: querying with the chunk's own text ranks it first.
    hits = embeds.search(
        query_vector=_vector("Systems of equations.\nGraphing lines."),
        model_id="fake-hash",
        limit=3,
    )
    assert [hit.chunk.item_id for hit in hits] == [deck.id]
    assert hits[0].score == 1.0

    # Re-run is a no-op.
    rerun = runner.run()
    assert (rerun.embedded, rerun.skipped, rerun.chunks_written) == (0, 2, 0)

    # A changed extraction re-embeds that item alone.
    revised = seed_item(corpus, "https://example.com/deck", content_hash="hash-2")
    seed_extraction(extracts, revised, text="[[page 1]]\nQuadratic functions.")
    changed = runner.run()
    assert changed.embedded == 1 and changed.skipped == 1
    texts = connection.execute("SELECT text FROM chunks").fetchall()
    assert texts == [("Quadratic functions.",)]

    # Tombstoning the item purges its chunks, vectors, and state.
    seed_item(corpus, "https://example.com/deck", status=ItemStatus.TOMBSTONED)
    purged = runner.run()
    assert purged.purged == 1
    assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    remaining = dict(
        connection.execute("SELECT item_id, chunk_count FROM embed_state").fetchall()
    )
    assert remaining == {blank.id: 0}
