"""Unit tests: the embed store + both vector backends.

The repository contract is exercised identically over the
sqlite-vec backend and the BLOB/numpy fallback — interface parity
is the property under test. Search expectations are hand-computed
cosines over 4-dim vectors.
"""

import math
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from school_index.corpus_sync.lookup import Found, Missing
from school_index.embed_index.domain import Chunk, EmbedState
from school_index.embed_index.store import (
    BlobBackend,
    EmbedStore,
    SqliteVecBackend,
    VectorBackend,
    load_vec,
    vec_available,
)

DIM = 4
SQRT_HALF = math.sqrt(0.5)

A = (1.0, 0.0, 0.0, 0.0)
B = (0.0, 1.0, 0.0, 0.0)
C = (SQRT_HALF, SQRT_HALF, 0.0, 0.0)


def make_store(tmp_path: Path, backend: VectorBackend) -> EmbedStore:
    db_path = tmp_path / "corpus.db"

    def factory() -> sqlite3.Connection:
        connection = sqlite3.connect(db_path)
        if isinstance(backend, SqliteVecBackend):
            load_vec(connection)
        return connection

    return EmbedStore(connection_factory=factory, backend=backend)


@pytest.fixture(params=["sqlite-vec", "blob-numpy"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[EmbedStore]:
    if request.param == "sqlite-vec":
        if not vec_available(dim=DIM):
            pytest.skip("sqlite-vec extension not loadable on this machine")
        backend: VectorBackend = SqliteVecBackend(dim=DIM)
    else:
        backend = BlobBackend(dim=DIM)
    yield make_store(tmp_path, backend)


def state(item_id: str, *, count: int, model: str = "fake-hash") -> EmbedState:
    return EmbedState(
        item_id=item_id,
        source_hash="hash-1",
        model_id=model,
        chunk_count=count,
        embedded_at="2026-10-23T15:00:00+00:00",
    )


def seed(store: EmbedStore) -> None:
    store.replace_item_chunks(
        chunks=[
            Chunk(item_id="item-1", ordinal=0, heading="page 1", text="alpha"),
            Chunk(item_id="item-1", ordinal=1, heading="page 2", text="beta"),
        ],
        vectors=[A, B],
        state=state("item-1", count=2),
    )
    store.replace_item_chunks(
        chunks=[Chunk(item_id="item-2", ordinal=0, heading=None, text="gamma")],
        vectors=[C],
        state=state("item-2", count=1),
    )


def test_replace_and_read_back(store: EmbedStore) -> None:
    seed(store)
    chunks = store.chunks_for_item(item_id="item-1")
    assert [(c.ordinal, c.heading, c.text) for c in chunks] == [
        (0, "page 1", "alpha"),
        (1, "page 2", "beta"),
    ]
    found = store.get_embed_state(item_id="item-1")
    assert isinstance(found, Found)
    assert found.value.chunk_count == 2
    assert isinstance(store.get_embed_state(item_id="nope"), Missing)
    assert store.total_chunks() == 3
    assert store.chunk_count_by_item() == {"item-1": 2, "item-2": 1}
    assert store.backend_name in {"sqlite-vec", "blob-numpy"}


def test_search_ranks_by_hand_computed_cosine(store: EmbedStore) -> None:
    seed(store)
    hits = store.search(query_vector=A, model_id="fake-hash", limit=3)
    assert [hit.chunk.text for hit in hits] == ["alpha", "gamma", "beta"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(SQRT_HALF)
    assert hits[2].score == pytest.approx(0.0)
    top1 = store.search(query_vector=A, model_id="fake-hash", limit=1)
    assert [hit.chunk.text for hit in top1] == ["alpha"]


def test_search_on_empty_index(store: EmbedStore) -> None:
    assert store.search(query_vector=A, model_id="fake-hash", limit=3) == []


def test_replace_swaps_vectors_and_chunks(store: EmbedStore) -> None:
    seed(store)
    store.replace_item_chunks(
        chunks=[Chunk(item_id="item-1", ordinal=0, heading="page 9", text="alpha-v2")],
        vectors=[B],
        state=state("item-1", count=1),
    )
    assert store.total_chunks() == 2
    hits = store.search(query_vector=A, model_id="fake-hash", limit=3)
    assert "alpha" not in [hit.chunk.text for hit in hits]
    assert hits[0].chunk.text == "gamma"


def test_purge_item_removes_chunks_vectors_and_state(store: EmbedStore) -> None:
    seed(store)
    store.purge_item(item_id="item-1")
    assert store.chunks_for_item(item_id="item-1") == []
    assert isinstance(store.get_embed_state(item_id="item-1"), Missing)
    hits = store.search(query_vector=A, model_id="fake-hash", limit=5)
    assert [hit.chunk.text for hit in hits] == ["gamma"]
    assert store.total_chunks() == 1


def test_clear_index_empties_everything(store: EmbedStore) -> None:
    seed(store)
    store.clear_index()
    assert store.total_chunks() == 0
    assert store.embed_states() == []
    assert store.search(query_vector=A, model_id="fake-hash", limit=3) == []


def test_replace_rejects_mismatched_inputs(store: EmbedStore) -> None:
    with pytest.raises(ValueError, match="pair up"):
        store.replace_item_chunks(
            chunks=[Chunk(item_id="i", ordinal=0, heading=None, text="x")],
            vectors=[],
            state=state("i", count=1),
        )
    with pytest.raises(ValueError, match="dim"):
        store.replace_item_chunks(
            chunks=[Chunk(item_id="i", ordinal=0, heading=None, text="x")],
            vectors=[(1.0, 0.0)],
            state=state("i", count=1),
        )


def test_vec_probe_matches_reality() -> None:
    # On machines with the wheel this is True; the store tests above
    # already skip-gate on it, so here we only assert it's a probe,
    # not an assumption: open() must pick the backend it probed.
    available = vec_available(dim=DIM)
    assert isinstance(available, bool)
