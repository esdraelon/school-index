"""Unit tests: CorpusStore repository (in-memory DB, tmp blob root)."""

import sqlite3
from pathlib import Path

import pytest

from school_index.corpus_sync.domain import CorpusItem, ItemStatus, SyncState
from school_index.corpus_sync.lookup import Found, Missing
from school_index.corpus_sync.store import CorpusStore

NOW = "2026-10-23T13:00:00+00:00"


@pytest.fixture
def store(tmp_path: Path) -> CorpusStore:
    connection = sqlite3.connect(":memory:")
    return CorpusStore(
        connection_factory=lambda: connection, files_root=tmp_path / "files"
    )


def make_item(source_ref: str, **overrides: object) -> CorpusItem:
    import hashlib

    fields: dict[str, object] = {
        "id": hashlib.sha256(source_ref.encode("utf-8")).hexdigest(),
        "source_kind": "AGENDA",
        "source_ref": source_ref,
        "student": "sam",
        "course": "Algebra I",
        "title": "Unit 2 notes",
        "deep_link": source_ref.split(":", 1)[1],
        "origin_url": source_ref.split(":", 1)[1],
        "format": "pdf",
        "content_hash": "ab12",
        "fetched_at": NOW,
        "status": ItemStatus.OK,
        "error": None,
    }
    fields.update(overrides)
    return CorpusItem(**fields)  # type: ignore[arg-type]


def test_upsert_then_get_roundtrips(store: CorpusStore) -> None:
    item = make_item("sam:https://example.com/a")
    store.upsert_item(item)
    result = store.get_item(source_ref=item.source_ref)
    assert isinstance(result, Found)
    assert result.value == item


def test_get_missing_item_returns_missing(store: CorpusStore) -> None:
    result = store.get_item(source_ref="sam:https://nope")
    assert isinstance(result, Missing)


def test_upsert_replaces_the_existing_row(store: CorpusStore) -> None:
    ref = "sam:https://example.com/a"
    store.upsert_item(make_item(ref))
    store.upsert_item(
        make_item(ref, status=ItemStatus.FAILED, error="HTTP 403", content_hash=None)
    )
    result = store.get_item(source_ref=ref)
    assert isinstance(result, Found)
    assert result.value.status is ItemStatus.FAILED
    assert result.value.error == "HTTP 403"
    assert len(store.items_for_student(student="sam")) == 1


def test_items_for_student_is_scoped(store: CorpusStore) -> None:
    store.upsert_item(make_item("sam:https://example.com/a"))
    store.upsert_item(make_item("owen:https://example.com/b", student="owen"))
    assert [i.source_ref for i in store.items_for_student(student="sam")] == [
        "sam:https://example.com/a"
    ]
    assert store.items_for_student(student="nobody") == []


def test_tombstone_missing_marks_only_unseen_active_items(
    store: CorpusStore,
) -> None:
    store.upsert_item(make_item("sam:https://example.com/seen"))
    store.upsert_item(make_item("sam:https://example.com/unseen"))
    store.upsert_item(
        make_item("sam:https://example.com/gone", status=ItemStatus.TOMBSTONED)
    )
    marked = store.tombstone_missing(
        student="sam", seen_refs={"sam:https://example.com/seen"}, at=NOW
    )
    assert marked == 1
    by_ref = {i.source_ref: i for i in store.items_for_student(student="sam")}
    assert by_ref["sam:https://example.com/seen"].status is ItemStatus.OK
    assert by_ref["sam:https://example.com/unseen"].status is ItemStatus.TOMBSTONED
    # Already tombstoned rows stay rows (pointers fail honestly).
    assert by_ref["sam:https://example.com/gone"].status is ItemStatus.TOMBSTONED


def test_tombstone_missing_with_empty_seen_set(store: CorpusStore) -> None:
    store.upsert_item(make_item("sam:https://example.com/a"))
    assert store.tombstone_missing(student="sam", seen_refs=set(), at=NOW) == 1


def test_sync_state_roundtrips(store: CorpusStore) -> None:
    state = SyncState(
        source="school-mcp",
        watermark="2026-10-23T05:15:00+00:00",
        last_run_at=NOW,
        last_status="partial",
        counts={"total": 200, "failed": 3},
    )
    store.record_sync(state)
    result = store.get_sync_state(source="school-mcp")
    assert isinstance(result, Found)
    assert result.value == state


def test_sync_state_missing_before_first_run(store: CorpusStore) -> None:
    assert isinstance(store.get_sync_state(source="school-mcp"), Missing)


def test_record_sync_replaces_previous_state(store: CorpusStore) -> None:
    base = {
        "source": "school-mcp",
        "watermark": None,
        "last_run_at": NOW,
        "last_status": "ok",
        "counts": {"total": 1},
    }
    store.record_sync(SyncState(**base))  # type: ignore[arg-type]
    store.record_sync(SyncState(**{**base, "last_status": "partial"}))  # type: ignore[arg-type]
    result = store.get_sync_state(source="school-mcp")
    assert isinstance(result, Found)
    assert result.value.last_status == "partial"


def test_write_blob_lands_bytes_by_item_id(store: CorpusStore, tmp_path: Path) -> None:
    store.write_blob(item_id="deadbeef", content=b"%PDF-fake")
    target = tmp_path / "files" / "deadbeef"
    assert target.read_bytes() == b"%PDF-fake"
    store.write_blob(item_id="deadbeef", content=b"%PDF-newer")
    assert target.read_bytes() == b"%PDF-newer"
    assert store.blob_path(item_id="deadbeef") == target


def test_open_creates_a_file_backed_store(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    first = CorpusStore.open(root)
    first.upsert_item(make_item("sam:https://example.com/a"))
    # A second open on the same root sees the first one's rows:
    # migrations are idempotent and the data persists.
    second = CorpusStore.open(root)
    assert isinstance(second.get_item(source_ref="sam:https://example.com/a"), Found)
    assert (root / "corpus.db").exists()
