"""Unit tests: the harvest run (real store, real registry, fakes at
the two ports — resource source and downloader).

The fixture payload is modeled on the real adapter data shape from
the 2026-10-23 live probe: a public deck, a Drive file, an IXL
link, and a login-walled document.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from school_index.corpus_sync.client import PayloadError
from school_index.corpus_sync.domain import ItemStatus, ResourceRef
from school_index.corpus_sync.fetch import RawResponse
from school_index.corpus_sync.harvester import Harvester
from school_index.corpus_sync.lookup import Found
from school_index.corpus_sync.store import CorpusStore
from school_index.corpus_sync.strategies import build_strategies

FIXED_NOW = datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc)
SCRAPED = "2026-10-23T05:15:00+00:00"

DECK_URL = "https://docs.google.com/presentation/d/deck1/pub"
DECK_EXPORT = "https://docs.google.com/presentation/d/deck1/export?format=pdf"
DRIVE_URL = "https://drive.google.com/file/d/file1/view"
DRIVE_DOWNLOAD = "https://drive.google.com/uc?export=download&id=file1"
IXL_URL = "https://www.ixl.com/math/algebra-1/pythagorean-theorem"
WALLED_URL = "https://docs.google.com/document/d/doc1/edit"
WALLED_EXPORT = "https://docs.google.com/document/d/doc1/export?format=pdf"


def resource(url: str, **overrides: object) -> ResourceRef:
    fields: dict[str, object] = {
        "student_id": "sam",
        "student_name": "Sam",
        "course_name": "Algebra I",
        "title": "Resource",
        "kind": "DOCUMENT",
        "url": url,
        "source_kind": "AGENDA",
        "scraped_at": SCRAPED,
    }
    fields.update(overrides)
    return ResourceRef(**fields)  # type: ignore[arg-type]


RESOURCES = [
    resource(DECK_URL, title="Pythagorean slides"),
    resource(DRIVE_URL, title="Worksheet", source_kind="HAC"),
    resource(IXL_URL, title="IXL practice", kind="TUTORIAL", course_name=None),
    resource(WALLED_URL, title="Walled doc"),
]


class FakeSource:
    def __init__(self, resources: list[ResourceRef]) -> None:
        self.resources = resources

    def list_resources(self, *, student: str) -> list[ResourceRef]:
        return list(self.resources)


class ExplodingSource:
    def list_resources(self, *, student: str) -> list[ResourceRef]:
        raise PayloadError("contract broke")


class FakeDownloader:
    def __init__(self, responses: dict[str, RawResponse]) -> None:
        self.responses = responses

    def get(self, *, url: str) -> RawResponse:
        return self.responses[url]


def canned_downloads() -> dict[str, RawResponse]:
    return {
        DECK_EXPORT: RawResponse(
            status_code=200,
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF-deck",
        ),
        DRIVE_DOWNLOAD: RawResponse(
            status_code=200,
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF-worksheet",
        ),
        WALLED_EXPORT: RawResponse(status_code=403, headers={}, content=b""),
    }


@pytest.fixture
def files_root(tmp_path: Path) -> Path:
    return tmp_path / "files"


@pytest.fixture
def store(files_root: Path) -> CorpusStore:
    connection = sqlite3.connect(":memory:")
    return CorpusStore(connection_factory=lambda: connection, files_root=files_root)


def make_harvester(
    store: CorpusStore,
    source: object,
    downloader: FakeDownloader | None = None,
) -> Harvester:
    return Harvester(
        source=source,  # type: ignore[arg-type]
        downloader=downloader or FakeDownloader(canned_downloads()),
        strategies=build_strategies(),
        store=store,
        now=lambda: FIXED_NOW,
    )


def test_first_run_folds_every_outcome(store: CorpusStore, files_root: Path) -> None:
    report = make_harvester(store, FakeSource(RESOURCES)).run(student="sam")
    assert (report.total, report.downloaded, report.link_only) == (4, 2, 1)
    assert (report.failed, report.tombstoned, report.unchanged) == (1, 0, 0)
    assert report.failure_tally() == {"HTTP 403": 1}

    items = {i.source_ref: i for i in store.items_for_student(student="sam")}
    deck = items[f"sam:{DECK_URL}"]
    assert deck.status is ItemStatus.OK and deck.format == "pdf"
    assert (files_root / deck.id).read_bytes() == b"%PDF-deck"
    link = items[f"sam:{IXL_URL}"]
    assert link.status is ItemStatus.OK and link.format == "external-link"
    assert link.content_hash is None and link.course is None
    walled = items[f"sam:{WALLED_URL}"]
    assert walled.status is ItemStatus.FAILED and walled.error == "HTTP 403"
    assert not (files_root / walled.id).exists()


def test_second_run_is_a_noop(store: CorpusStore, files_root: Path) -> None:
    harvester = make_harvester(store, FakeSource(RESOURCES))
    harvester.run(student="sam")
    deck_id = resource(DECK_URL).item_id
    blob = files_root / deck_id
    mtime_before = blob.stat().st_mtime_ns
    hashes_before = {
        i.source_ref: i.content_hash for i in store.items_for_student(student="sam")
    }

    report = harvester.run(student="sam")
    assert report.downloaded == 0 and report.link_only == 0
    assert report.unchanged == 3 and report.tombstoned == 0
    assert report.failed == 1  # the walled doc is re-attempted, still data
    assert blob.stat().st_mtime_ns == mtime_before
    hashes_after = {
        i.source_ref: i.content_hash for i in store.items_for_student(student="sam")
    }
    assert hashes_after == hashes_before


def test_vanished_resource_is_tombstoned(store: CorpusStore) -> None:
    source = FakeSource(RESOURCES)
    harvester = make_harvester(store, source)
    harvester.run(student="sam")
    source.resources = [r for r in RESOURCES if r.url != DRIVE_URL]
    report = harvester.run(student="sam")
    assert report.tombstoned == 1
    result = store.get_item(source_ref=f"sam:{DRIVE_URL}")
    assert isinstance(result, Found)
    assert result.value.status is ItemStatus.TOMBSTONED
    # The row (and its bytes) survive: pointers fail honestly.
    assert result.value.content_hash is not None


def test_changed_bytes_are_rewritten(store: CorpusStore, files_root: Path) -> None:
    downloader = FakeDownloader(canned_downloads())
    harvester = make_harvester(store, FakeSource(RESOURCES), downloader)
    harvester.run(student="sam")
    downloader.responses[DECK_EXPORT] = RawResponse(
        status_code=200,
        headers={"Content-Type": "application/pdf"},
        content=b"%PDF-deck-v2",
    )
    report = harvester.run(student="sam")
    assert report.downloaded == 1
    deck_id = resource(DECK_URL).item_id
    assert (files_root / deck_id).read_bytes() == b"%PDF-deck-v2"


def test_failure_after_success_preserves_stored_bytes(
    store: CorpusStore, files_root: Path
) -> None:
    downloader = FakeDownloader(canned_downloads())
    harvester = make_harvester(store, FakeSource(RESOURCES), downloader)
    harvester.run(student="sam")
    downloader.responses[DECK_EXPORT] = RawResponse(
        status_code=403, headers={}, content=b""
    )
    report = harvester.run(student="sam")
    assert report.failed == 2
    result = store.get_item(source_ref=f"sam:{DECK_URL}")
    assert isinstance(result, Found)
    assert result.value.status is ItemStatus.FAILED
    assert result.value.content_hash is not None  # last good bytes kept
    assert (files_root / result.value.id).read_bytes() == b"%PDF-deck"


def test_metadata_drift_refreshes_row_without_rewriting_blob(
    store: CorpusStore, files_root: Path
) -> None:
    source = FakeSource(RESOURCES)
    harvester = make_harvester(store, source)
    harvester.run(student="sam")
    deck_id = resource(DECK_URL).item_id
    mtime_before = (files_root / deck_id).stat().st_mtime_ns
    source.resources = [
        resource(DECK_URL, title="Pythagorean slides (revised)"),
        *RESOURCES[1:],
    ]
    report = harvester.run(student="sam")
    assert report.downloaded == 0
    assert (files_root / deck_id).stat().st_mtime_ns == mtime_before
    result = store.get_item(source_ref=f"sam:{DECK_URL}")
    assert isinstance(result, Found)
    assert result.value.title == "Pythagorean slides (revised)"


def test_sync_state_records_the_run(store: CorpusStore) -> None:
    make_harvester(store, FakeSource(RESOURCES)).run(student="sam")
    result = store.get_sync_state(source="school-mcp")
    assert isinstance(result, Found)
    state = result.value
    assert state.last_status == "partial"  # one failure in the fixture set
    assert state.watermark == SCRAPED
    assert state.counts["total"] == 4 and state.counts["failed"] == 1


def test_source_failure_aborts_without_tombstoning(store: CorpusStore) -> None:
    make_harvester(store, FakeSource(RESOURCES)).run(student="sam")
    with pytest.raises(PayloadError):
        make_harvester(store, ExplodingSource()).run(student="sam")
    # Nothing was tombstoned on the strength of a broken contract.
    statuses = {i.status for i in store.items_for_student(student="sam")}
    assert ItemStatus.TOMBSTONED not in statuses
