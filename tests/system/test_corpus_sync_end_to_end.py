"""System test: one full harvest, end to end.

Everything is real — the harvester, the strategy registry, the
HTTP downloader, the SQLite store on disk — except the network edge:
a stub HTTP server on localhost stands in for docs.google.com /
drive.google.com (a StubTransport rewrites the host; the requests
themselves travel real sockets to the stub). The resource source is
a fixture list shaped exactly like the adapter's probed payload.
"""

import sqlite3
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

from school_index.corpus_sync.domain import ItemStatus, ResourceRef
from school_index.corpus_sync.fetch import HttpDownloader
from school_index.corpus_sync.harvester import Harvester
from school_index.corpus_sync.lookup import Found
from school_index.corpus_sync.store import CorpusStore
from school_index.corpus_sync.strategies import build_strategies

SCRAPED = "2026-10-23T05:15:00+00:00"
DECK_URL = "https://docs.google.com/presentation/d/deck1/pub"
DRIVE_URL = "https://drive.google.com/file/d/file1/view"
IXL_URL = "https://www.ixl.com/math/algebra-1/pythagorean-theorem"
WALLED_URL = "https://docs.google.com/document/d/doc1/edit"

_ROUTES: dict[str, tuple[int, bytes]] = {
    "/presentation/d/deck1/export": (200, b"%PDF-deck"),
    "/uc": (200, b"%PDF-worksheet"),
    "/document/d/doc1/export": (403, b"forbidden"),
}


class _StubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        status, body = _ROUTES.get(urlsplit(self.path).path, (404, b"nope"))
        self.send_response(status)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass  # keep test output clean


class StubTransport(httpx.BaseTransport):
    """Forwards every request to the stub server (the network edge)."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._client = httpx.Client(trust_env=False)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        parts = urlsplit(str(request.url))
        target = self._base_url + parts.path + (f"?{parts.query}" if parts.query else "")
        response = self._client.get(target)
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
        )


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


class FixtureSource:
    def __init__(self, resources: list[ResourceRef]) -> None:
        self.resources = resources

    def list_resources(self, *, student: str) -> list[ResourceRef]:
        return list(self.resources)


@pytest.fixture
def stub_base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


def make_harvester(root: Path, base_url: str, source: FixtureSource) -> Harvester:
    client = httpx.Client(
        transport=StubTransport(base_url), follow_redirects=True, trust_env=False
    )
    return Harvester(
        source=source,
        downloader=HttpDownloader(client=client),
        strategies=build_strategies(),
        store=CorpusStore.open(root),
    )


def test_full_harvest_and_noop_rerun(tmp_path: Path, stub_base_url: str) -> None:
    root = tmp_path / "corpus"
    resources = [
        resource(DECK_URL, title="Pythagorean slides"),
        resource(DRIVE_URL, title="Worksheet", source_kind="HAC"),
        resource(IXL_URL, title="IXL practice", kind="TUTORIAL", course_name=None),
        resource(WALLED_URL, title="Walled doc"),
    ]
    source = FixtureSource(resources)
    harvester = make_harvester(root, stub_base_url, source)

    report = harvester.run(student="sam")
    assert (report.downloaded, report.link_only, report.failed) == (2, 1, 1)
    assert report.failure_tally() == {"HTTP 403": 1}

    # The store on disk says so, independently of the harvester.
    connection = sqlite3.connect(root / "corpus.db")
    rows = connection.execute(
        "SELECT status, format FROM corpus_items ORDER BY title"
    ).fetchall()
    assert rows == [
        ("ok", "external-link"),  # IXL practice
        ("ok", "pdf"),  # Pythagorean slides
        ("failed", ""),  # Walled doc
        ("ok", "pdf"),  # Worksheet
    ]
    state = connection.execute(
        "SELECT last_status FROM sync_state WHERE source = 'school-mcp'"
    ).fetchone()
    assert state == ("partial",)
    blobs = sorted(p.name for p in (root / "files").iterdir())
    assert len(blobs) == 2
    mtimes = {p.name: p.stat().st_mtime_ns for p in (root / "files").iterdir()}

    rerun = harvester.run(student="sam")
    assert rerun.downloaded == 0 and rerun.tombstoned == 0
    assert {p.name: p.stat().st_mtime_ns for p in (root / "files").iterdir()} == mtimes

    # A vanished resource tombstones; its row and bytes stay put.
    source.resources = [r for r in resources if r.url != DRIVE_URL]
    third = harvester.run(student="sam")
    assert third.tombstoned == 1
    store = CorpusStore.open(root)
    result = store.get_item(source_ref=f"sam:{DRIVE_URL}")
    assert isinstance(result, Found)
    assert result.value.status is ItemStatus.TOMBSTONED
    assert len(list((root / "files").iterdir())) == 2
