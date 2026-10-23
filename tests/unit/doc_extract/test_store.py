"""Unit tests: ExtractStore repository (in-memory DB)."""

import sqlite3

import pytest

from school_index.corpus_sync.lookup import Found, Missing
from school_index.doc_extract.domain import ExtractionOutcome, ExtractionState
from school_index.doc_extract.store import ExtractStore

NOW = "2026-10-23T14:00:00+00:00"
LATER = "2026-10-23T15:00:00+00:00"


@pytest.fixture
def store() -> ExtractStore:
    connection = sqlite3.connect(":memory:")
    return ExtractStore(connection_factory=lambda: connection)


def make_state(item_id: str, **overrides: object) -> ExtractionState:
    fields: dict[str, object] = {
        "item_id": item_id,
        "outcome": ExtractionOutcome.EXTRACTED,
        "extractor": "pypdf",
        "source_hash": "ab12",
        "format": "pdf",
        "char_count": 42,
        "page_count": 2,
        "error": None,
        "extracted_at": NOW,
    }
    fields.update(overrides)
    return ExtractionState(**fields)  # type: ignore[arg-type]


def test_record_then_read_back_text_and_state(store: ExtractStore) -> None:
    store.record_outcome(make_state("item-1"), text="[[page 1]]\nHello")
    text = store.get_text(item_id="item-1")
    assert isinstance(text, Found)
    assert text.value.text == "[[page 1]]\nHello"
    assert text.value.extractor == "pypdf"
    state = store.get_state(item_id="item-1")
    assert isinstance(state, Found)
    assert state.value == make_state("item-1")


def test_get_missing_returns_missing(store: ExtractStore) -> None:
    assert isinstance(store.get_text(item_id="nope"), Missing)
    assert isinstance(store.get_state(item_id="nope"), Missing)


def test_empty_outcome_stores_an_empty_text_row(store: ExtractStore) -> None:
    store.record_outcome(
        make_state(
            "item-1",
            outcome=ExtractionOutcome.EMPTY,
            char_count=0,
        ),
        text="",
    )
    text = store.get_text(item_id="item-1")
    assert isinstance(text, Found)
    assert text.value.text == ""


def test_failure_record_deletes_stale_text(store: ExtractStore) -> None:
    # The blob changed and the new revision is corrupt: text from the
    # previous revision must not survive as if it described this one.
    store.record_outcome(make_state("item-1"), text="old text")
    store.record_outcome(
        make_state(
            "item-1",
            outcome=ExtractionOutcome.FAILED,
            source_hash="cd34",
            char_count=0,
            page_count=None,
            error="PdfReadError: broken",
            extracted_at=LATER,
        ),
        text=None,
    )
    assert isinstance(store.get_text(item_id="item-1"), Missing)
    state = store.get_state(item_id="item-1")
    assert isinstance(state, Found)
    assert state.value.outcome is ExtractionOutcome.FAILED
    assert state.value.error == "PdfReadError: broken"


def test_outcome_counts_tally_by_outcome(store: ExtractStore) -> None:
    store.record_outcome(make_state("a"), text="x")
    store.record_outcome(
        make_state("b", outcome=ExtractionOutcome.EMPTY, char_count=0), text=""
    )
    store.record_outcome(
        make_state("c", outcome=ExtractionOutcome.NOT_EXTRACTABLE, extractor=None),
        text=None,
    )
    assert store.outcome_counts() == {
        "extracted": 1,
        "empty": 1,
        "not-extractable": 1,
    }


def test_text_without_an_extractor_is_rejected(store: ExtractStore) -> None:
    with pytest.raises(ValueError, match="extractor"):
        store.record_outcome(
            make_state("item-1", extractor=None), text="orphan text"
        )
