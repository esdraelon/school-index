"""Download strategies: one per URL shape, chained in a registry.

Pattern: Strategy + Chain of Responsibility. Each strategy owns one
URL shape (``matches``) and one way of turning a resource URL into
bytes (``fetch``). The harvester walks the injected chain and the
first matching strategy wins — there is no host if/else anywhere;
adding a source shape means adding a strategy to the chain the
composition root builds (``build_strategies``). The chain always
ends in ``ExternalLinkStrategy`` (matches everything), so dispatch
is total: every resource gets an outcome.

Outcomes are a sum type — ``Downloaded`` / ``LinkOnly`` /
``DownloadFailed`` — consumed by exhaustive ``match`` in the
harvester. A failure is *data* (it becomes the item's stored error
record), never an exception that could abort the run.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias
from urllib.parse import urlsplit

from school_index.corpus_sync.domain import ResourceRef
from school_index.corpus_sync.fetch import Downloader, FetchError, RawResponse

#: Stored format for pointer-only items (nothing was downloaded).
EXTERNAL_LINK_FORMAT = "external-link"

#: Failure recorded when a download lands on Google's sign-in page:
#: the file is not public, and the page's bytes are per-request (they
#: would never hash-match on re-runs, and they are not the resource).
SIGNIN_ERROR = "google sign-in page (file not publicly accessible)"


def _is_signin_bytes(content: bytes) -> bool:
    """True when bytes are Google's sign-in page, not a document.

    Unauthenticated requests for non-public Google files redirect to
    accounts.google.com; httpx follows and lands on a 200 HTML page.
    The marker is the sign-in flow's own base URL, present in the
    first bytes of every such page (seen live, 2026-10-23: 28 Drive
    items were silently storing this page as their "content").
    """
    return b"accounts.google.com" in content[:4096]


@dataclass(frozen=True, kw_only=True)
class Downloaded:
    """Bytes in hand. Pattern: sum-type variant (fetch success)."""

    content: bytes
    format: str


@dataclass(frozen=True, kw_only=True)
class LinkOnly:
    """Pointer-only item — by design, not a failure.

    Interactive or external material (IXL practice, web pages) has no
    file to harvest; the corpus keeps the deep link and moves on.
    """

    format: str = EXTERNAL_LINK_FORMAT


@dataclass(frozen=True, kw_only=True)
class DownloadFailed:
    """A fetch that produced an error record (HTTP status, transport)."""

    error: str


#: What a strategy returns for one resource.
FetchOutcome: TypeAlias = Downloaded | LinkOnly | DownloadFailed


class DownloadStrategy(Protocol):
    """One URL shape + how to fetch it."""

    name: str

    def matches(self, *, url: str) -> bool:
        """True when this strategy owns ``url``'s shape."""
        ...

    def fetch(
        self, *, resource: ResourceRef, downloader: Downloader
    ) -> FetchOutcome:
        """Resolve + fetch one resource; failures are outcomes."""
        ...


def _get(
    *, downloader: Downloader, url: str, format_for: Callable[[RawResponse], str]
) -> FetchOutcome:
    """Shared fetch tail: GET, classify the status, wrap the bytes."""
    try:
        response = downloader.get(url=url)
    except FetchError as exc:
        return DownloadFailed(error=f"download error: {exc}")
    if response.status_code != 200:
        return DownloadFailed(error=f"HTTP {response.status_code}")
    return Downloaded(content=response.content, format=format_for(response))


def _path_segments(url: str) -> list[str]:
    return urlsplit(url).path.split("/")


class GoogleDocsExportStrategy:
    """docs.google.com files, via the export endpoint.

    Pattern: Strategy. Presentations, documents, and spreadsheets all
    export as PDF — fidelity over editability: the corpus stores
    what students saw, and M2 extracts text from the PDF. The
    app-segment → format choice is a Mapping lookup, not a branch.
    """

    name = "google-docs-export"

    #: Google app path segment → export format.
    _FORMATS = {"presentation": "pdf", "document": "pdf", "spreadsheets": "pdf"}

    def _parse(self, url: str) -> tuple[str, str] | None:
        """(app segment, document id) for exportable docs URLs, else None."""
        if urlsplit(url).hostname != "docs.google.com":
            return None
        segments = _path_segments(url)
        # Shape: /<app>/d/<id>/... — app must be an exportable kind.
        if len(segments) >= 4 and segments[2] == "d":
            app, doc_id = segments[1], segments[3]
            if app in self._FORMATS and doc_id:
                return app, doc_id
        return None

    def matches(self, *, url: str) -> bool:
        return self._parse(url) is not None

    def fetch(
        self, *, resource: ResourceRef, downloader: Downloader
    ) -> FetchOutcome:
        parsed = self._parse(resource.url)
        if parsed is None:  # unreachable via the chain; defensive
            return DownloadFailed(error=f"not an exportable docs URL: {resource.url}")
        app, doc_id = parsed
        export_format = self._FORMATS[app]
        export_url = (
            f"https://docs.google.com/{app}/d/{doc_id}/export?format={export_format}"
        )
        outcome = _get(
            downloader=downloader,
            url=export_url,
            format_for=lambda _response: export_format,
        )
        if isinstance(outcome, Downloaded) and not outcome.content.startswith(
            b"%PDF"
        ):
            # A 200 that is not a PDF is not the deck: for a non-public
            # file it is the sign-in page (see _is_signin_bytes).
            if _is_signin_bytes(outcome.content):
                return DownloadFailed(error=SIGNIN_ERROR)
            return DownloadFailed(error="export returned non-PDF content")
        return outcome


class DriveFileDownloadStrategy:
    """drive.google.com file links, via the direct-download endpoint.

    Pattern: Strategy. The stored format comes from the response —
    Content-Type first (a Mapping), then the Content-Disposition
    filename's suffix, then "bin": Drive serves the file as-is, so
    the response is the only honest source of what it is.
    """

    name = "drive-file-download"

    _CONTENT_TYPE_FORMATS = {
        "application/pdf": "pdf",
        "application/msword": "doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "application/zip": "zip",
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "text/plain": "txt",
        "text/html": "html",
    }

    def _file_id(self, url: str) -> str | None:
        if urlsplit(url).hostname != "drive.google.com":
            return None
        segments = _path_segments(url)
        # Shape: /file/d/<id>/...
        if len(segments) >= 4 and segments[1] == "file" and segments[2] == "d":
            return segments[3] or None
        return None

    def matches(self, *, url: str) -> bool:
        return self._file_id(url) is not None

    def fetch(
        self, *, resource: ResourceRef, downloader: Downloader
    ) -> FetchOutcome:
        file_id = self._file_id(resource.url)
        if file_id is None:  # unreachable via the chain; defensive
            return DownloadFailed(error=f"not a drive file URL: {resource.url}")
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        outcome = _get(
            downloader=downloader, url=download_url, format_for=self._format_for
        )
        if isinstance(outcome, Downloaded) and _is_signin_bytes(outcome.content):
            # Non-public file: Drive redirected to the sign-in page.
            # Genuine HTML files stored in Drive do not carry the
            # sign-in marker and still land as format "html".
            return DownloadFailed(error=SIGNIN_ERROR)
        return outcome

    def _format_for(self, response: RawResponse) -> str:
        by_type = self._CONTENT_TYPE_FORMATS.get(response.content_type)
        if by_type is not None:
            return by_type
        filename = response.filename
        if filename and "." in filename:
            return filename.rsplit(".", 1)[1].lower()
        return "bin"


class ExternalLinkStrategy:
    """Catch-all: pointer-only items. Pattern: Strategy (null object).

    IXL practice, school sites, and everything unrecognized: the
    corpus stores the deep link with format "external-link" and
    downloads nothing. This is a correct outcome, not a miss —
    dispatch falling through to here is by design.
    """

    name = "external-link"

    def matches(self, *, url: str) -> bool:
        return True

    def fetch(
        self, *, resource: ResourceRef, downloader: Downloader
    ) -> FetchOutcome:
        return LinkOnly()


def build_strategies() -> tuple[DownloadStrategy, ...]:
    """The production chain, in dispatch order (composition root)."""
    return (
        GoogleDocsExportStrategy(),
        DriveFileDownloadStrategy(),
        ExternalLinkStrategy(),
    )
