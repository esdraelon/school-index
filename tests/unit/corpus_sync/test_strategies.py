"""Unit tests: download strategies + the HTTP downloader port.

Downloads are canned (FakeDownloader / httpx MockTransport) — the
network edge is the only substitution; the registry, the URL-shape
matching, and the outcome classification are the real code.
"""

import httpx
import pytest

from school_index.corpus_sync.domain import ResourceRef
from school_index.corpus_sync.fetch import (
    FetchError,
    HttpDownloader,
    RawResponse,
)
from school_index.corpus_sync.strategies import (
    Downloaded,
    DownloadFailed,
    DriveFileDownloadStrategy,
    ExternalLinkStrategy,
    GoogleDocsExportStrategy,
    LinkOnly,
    build_strategies,
)


def make_resource(url: str) -> ResourceRef:
    return ResourceRef(
        student_id="sam",
        student_name="Sam",
        course_name=None,
        title="",
        kind="LINK",
        url=url,
        source_kind="HAC",
        scraped_at="2026-10-23T05:15:00+00:00",
    )


class FakeDownloader:
    """Downloader port double: canned responses keyed by URL."""

    def __init__(
        self,
        responses: dict[str, RawResponse] | None = None,
        error: FetchError | None = None,
    ) -> None:
        self._responses = responses or {}
        self._error = error
        self.requested: list[str] = []

    def get(self, *, url: str) -> RawResponse:
        self.requested.append(url)
        if self._error is not None:
            raise self._error
        return self._responses[url]


def ok_response(content: bytes = b"bytes", **headers: str) -> RawResponse:
    return RawResponse(status_code=200, headers=headers, content=content)


# -- GoogleDocsExportStrategy ------------------------------------------


@pytest.mark.parametrize(
    "app", ["presentation", "document", "spreadsheets"]
)
def test_google_docs_exports_each_app_as_pdf(app: str) -> None:
    strategy = GoogleDocsExportStrategy()
    url = f"https://docs.google.com/{app}/d/AbC123/edit?usp=sharing"
    export_url = f"https://docs.google.com/{app}/d/AbC123/export?format=pdf"
    downloader = FakeDownloader({export_url: ok_response(b"%PDF")})
    assert strategy.matches(url=url)
    outcome = strategy.fetch(resource=make_resource(url), downloader=downloader)
    assert outcome == Downloaded(content=b"%PDF", format="pdf")
    assert downloader.requested == [export_url]


def test_google_docs_rejects_non_exportable_shapes() -> None:
    strategy = GoogleDocsExportStrategy()
    assert not strategy.matches(url="https://docs.google.com/forms/d/AbC123/viewform")
    assert not strategy.matches(url="https://docs.google.com/document/d//edit")
    assert not strategy.matches(url="https://example.com/document/d/AbC123/edit")


def test_google_docs_http_failure_is_an_outcome() -> None:
    strategy = GoogleDocsExportStrategy()
    url = "https://docs.google.com/presentation/d/AbC123/pub"
    export_url = "https://docs.google.com/presentation/d/AbC123/export?format=pdf"
    forbidden = RawResponse(status_code=403, headers={}, content=b"")
    downloader = FakeDownloader({export_url: forbidden})
    outcome = strategy.fetch(resource=make_resource(url), downloader=downloader)
    assert outcome == DownloadFailed(error="HTTP 403")


def test_transport_failure_is_an_outcome() -> None:
    strategy = GoogleDocsExportStrategy()
    url = "https://docs.google.com/document/d/AbC123/edit"
    downloader = FakeDownloader(error=FetchError("ConnectError: refused"))
    outcome = strategy.fetch(resource=make_resource(url), downloader=downloader)
    assert isinstance(outcome, DownloadFailed)
    assert "ConnectError" in outcome.error


# -- DriveFileDownloadStrategy ------------------------------------------


def test_drive_file_downloads_via_uc_endpoint() -> None:
    strategy = DriveFileDownloadStrategy()
    url = "https://drive.google.com/file/d/XyZ789/view?usp=sharing"
    download_url = "https://drive.google.com/uc?export=download&id=XyZ789"
    downloader = FakeDownloader(
        {download_url: ok_response(b"%PDF", **{"Content-Type": "application/pdf"})}
    )
    assert strategy.matches(url=url)
    outcome = strategy.fetch(resource=make_resource(url), downloader=downloader)
    assert outcome == Downloaded(content=b"%PDF", format="pdf")
    assert downloader.requested == [download_url]


def test_drive_format_falls_back_to_filename_suffix() -> None:
    strategy = DriveFileDownloadStrategy()
    url = "https://drive.google.com/file/d/XyZ789/view"
    download_url = "https://drive.google.com/uc?export=download&id=XyZ789"
    response = ok_response(
        b"data",
        **{
            "Content-Type": "application/octet-stream",
            "Content-Disposition": 'attachment; filename="Worksheet.DOCX"',
        },
    )
    downloader = FakeDownloader({download_url: response})
    outcome = strategy.fetch(resource=make_resource(url), downloader=downloader)
    assert outcome == Downloaded(content=b"data", format="docx")


def test_drive_format_defaults_to_bin() -> None:
    strategy = DriveFileDownloadStrategy()
    url = "https://drive.google.com/file/d/XyZ789/view"
    download_url = "https://drive.google.com/uc?export=download&id=XyZ789"
    downloader = FakeDownloader(
        {download_url: ok_response(b"data", **{"Content-Type": "application/x-weird"})}
    )
    outcome = strategy.fetch(resource=make_resource(url), downloader=downloader)
    assert outcome == Downloaded(content=b"data", format="bin")


def test_drive_rejects_non_file_shapes() -> None:
    strategy = DriveFileDownloadStrategy()
    assert not strategy.matches(url="https://drive.google.com/drive/folders/XyZ789")
    assert not strategy.matches(url="https://docs.google.com/file/d/XyZ789/view")


def test_fetch_on_unmatched_url_is_a_defensive_failure() -> None:
    # fetch() re-checks the shape itself: called off-chain (which the
    # harvester never does), it fails as data rather than crashing.
    downloader = FakeDownloader()
    for strategy in (GoogleDocsExportStrategy(), DriveFileDownloadStrategy()):
        outcome = strategy.fetch(
            resource=make_resource("https://example.com/x"), downloader=downloader
        )
        assert isinstance(outcome, DownloadFailed)
    assert downloader.requested == []


# -- ExternalLinkStrategy + the chain -----------------------------------


def test_external_link_never_downloads() -> None:
    strategy = ExternalLinkStrategy()
    downloader = FakeDownloader()
    for url in (
        "https://www.ixl.com/math/algebra-1/pythagorean-theorem",
        "https://sites.google.com/leanderisd.org/8th-grade-team-site/home",
    ):
        assert strategy.matches(url=url)
        outcome = strategy.fetch(resource=make_resource(url), downloader=downloader)
        assert outcome == LinkOnly(format="external-link")
    assert downloader.requested == []


def test_chain_dispatches_real_world_shapes_in_order() -> None:
    chain = build_strategies()

    def owner(url: str) -> str:
        for strategy in chain:
            if strategy.matches(url=url):
                return strategy.name
        raise AssertionError("chain must be total")

    assert owner("https://docs.google.com/presentation/d/X/pub") == "google-docs-export"
    assert owner("https://drive.google.com/file/d/X/view") == "drive-file-download"
    assert owner("https://www.ixl.com/math") == "external-link"
    assert owner("https://www.google.com/search?q=x") == "external-link"
    assert chain[-1].name == "external-link"


# -- HttpDownloader ------------------------------------------------------


def test_http_downloader_returns_raw_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"X-Test": "1"}, content=b"body")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = HttpDownloader(client=client).get(url="https://example.com/f")
    assert response.status_code == 200
    assert response.content == b"body"
    assert response.header("x-test") == "1"
    assert response.content_type == ""


def test_http_downloader_passes_error_statuses_through() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _req: httpx.Response(403))
    )
    response = HttpDownloader(client=client).get(url="https://example.com/f")
    assert response.status_code == 403


def test_http_downloader_wraps_transport_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(FetchError, match="ConnectError"):
        HttpDownloader(client=client).get(url="https://example.com/f")


def test_raw_response_content_type_normalizes() -> None:
    response = RawResponse(
        status_code=200,
        headers={"Content-Type": "Application/PDF; charset=binary"},
        content=b"",
    )
    assert response.content_type == "application/pdf"
    assert response.filename is None
