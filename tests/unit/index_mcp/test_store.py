"""Unit tests: the read-only IndexReadStore against real artifacts.

The fixture index is written by the real write stores (whichever
vector backend this machine probes), and the read store must match
that backend from the artifact alone — plus never write: row counts
and the schema inventory are asserted identical across searches.
"""

import sqlite3
from pathlib import Path

import pytest

from school_index.corpus_sync.domain import ItemStatus
from school_index.corpus_sync.lookup import Found, Missing
from school_index.corpus_sync.store import CorpusStore
from school_index.embed_index.store import EmbedStore
from school_index.index_mcp.store import IndexNotReady, IndexReadStore
from tests.unit.index_mcp.fakes import (
    DECK_A,
    SCI_B,
    WS_C,
    SeedChunk,
    SeedItem,
    StubProvider,
    build_index,
)

TOMBSTONED = SeedItem(
    url="https://example.com/gone",
    title="Retracted Deck",
    course="Algebra I",
    status=ItemStatus.TOMBSTONED,
    chunks=(
        SeedChunk(text="Retracted pythagorean text.", heading=None, vector=(1.0, 0.0)),
    ),
)


def _census(db_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(db_path)
    try:
        tables = (
            "corpus_items",
            "chunks",
            "embed_state",
            "schema_migrations",
        )
        return {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in tables
        }
    finally:
        connection.close()


def test_open_rejects_a_missing_corpus_db(tmp_path: Path) -> None:
    with pytest.raises(IndexNotReady, match="no corpus.db"):
        IndexReadStore.open(tmp_path, dim=StubProvider.dim)


def test_backend_matches_the_writer(tmp_path: Path) -> None:
    build_index(tmp_path, [DECK_A])
    writer = EmbedStore.open(tmp_path, dim=StubProvider.dim)
    reader = IndexReadStore.open(tmp_path, dim=StubProvider.dim)
    assert reader.backend_name == writer.backend_name


def test_search_joins_provenance_and_ranks(tmp_path: Path) -> None:
    build_index(tmp_path, [DECK_A, SCI_B, WS_C])
    reader = IndexReadStore.open(tmp_path, dim=StubProvider.dim)
    hits = reader.search(
        query_vector=(1.0, 0.0), model_id="stub-2d", course=None, fetch_limit=10
    )
    assert hits[0].text.startswith("Model Pythagorean theorem")
    assert hits[0].title == "Pythagorean Deck"
    assert hits[0].course == "Algebra I"
    assert hits[0].deep_link == "https://example.com/deck-a"
    assert hits[0].heading == "page 3"
    assert hits[0].score == pytest.approx(1.0)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_carries_each_items_first_chunk(tmp_path: Path) -> None:
    build_index(tmp_path, [DECK_A, SCI_B, WS_C])
    reader = IndexReadStore.open(tmp_path, dim=StubProvider.dim)
    hits = reader.search(
        query_vector=(1.0, 0.0), model_id="stub-2d", course=None, fetch_limit=10
    )
    by_title = {hit.title: hit for hit in hits}
    # DECK_A's second chunk (page 4) still reports the FIRST chunk's
    # anchor + text: the fallback disambiguator is per item, not
    # per hit chunk.
    deck_hits = [hit for hit in hits if hit.title == "Pythagorean Deck"]
    assert len(deck_hits) == 2
    assert all(hit.first_heading == "page 3" for hit in deck_hits)
    assert all(
        hit.first_text.startswith("Model Pythagorean theorem") for hit in deck_hits
    )
    assert by_title["Algebra Worksheet"].first_heading is None
    assert by_title["Algebra Worksheet"].first_text == "Triangles drill one."


def test_search_filters_by_course(tmp_path: Path) -> None:
    build_index(tmp_path, [DECK_A, SCI_B])
    reader = IndexReadStore.open(tmp_path, dim=StubProvider.dim)
    hits = reader.search(
        query_vector=(1.0, 0.0),
        model_id="stub-2d",
        course="Science",
        fetch_limit=10,
    )
    assert [hit.title for hit in hits] == ["Photosynthesis Notes"]


def test_search_never_serves_tombstoned_items(tmp_path: Path) -> None:
    build_index(tmp_path, [DECK_A, TOMBSTONED])
    reader = IndexReadStore.open(tmp_path, dim=StubProvider.dim)
    hits = reader.search(
        query_vector=(1.0, 0.0), model_id="stub-2d", course=None, fetch_limit=10
    )
    assert all(hit.title != "Retracted Deck" for hit in hits)


def test_empty_index_reads_as_missing_model_and_no_hits(tmp_path: Path) -> None:
    CorpusStore.open(tmp_path)  # schema exists; nothing embedded
    reader = IndexReadStore.open(tmp_path, dim=StubProvider.dim)
    assert reader.backend_name == "none"
    assert isinstance(reader.index_model_id(), Missing)
    assert (
        reader.search(
            query_vector=(1.0, 0.0), model_id="stub-2d", course=None, fetch_limit=5
        )
        == []
    )


def test_index_model_id_reads_the_census(tmp_path: Path) -> None:
    build_index(tmp_path, [DECK_A])
    reader = IndexReadStore.open(tmp_path, dim=StubProvider.dim)
    census = reader.index_model_id()
    assert isinstance(census, Found)
    assert census.value == "stub-2d"


def test_mixed_model_index_is_loud(tmp_path: Path) -> None:
    other = SeedItem(
        url="https://example.com/other",
        title="Other Model Deck",
        course="Science",
        model_id="someone-else",
        chunks=(SeedChunk(text="Other text.", heading=None, vector=(0.0, 1.0)),),
    )
    build_index(tmp_path, [DECK_A, other])
    reader = IndexReadStore.open(tmp_path, dim=StubProvider.dim)
    with pytest.raises(IndexNotReady, match="mixed-model"):
        reader.index_model_id()


def test_reads_do_not_write(tmp_path: Path) -> None:
    build_index(tmp_path, [DECK_A, SCI_B, WS_C])
    before = _census(tmp_path / "corpus.db")
    reader = IndexReadStore.open(tmp_path, dim=StubProvider.dim)
    reader.search(
        query_vector=(1.0, 0.0), model_id="stub-2d", course=None, fetch_limit=10
    )
    reader.index_model_id()
    assert _census(tmp_path / "corpus.db") == before
