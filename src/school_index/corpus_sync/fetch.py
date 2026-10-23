"""HTTP download transport.

Pattern: Port (the ``Downloader`` protocol) + Adapter
(``HttpDownloader`` over httpx). Strategies depend on the port, never
on httpx, so unit tests inject canned responses and only system
tests touch real sockets (a localhost stub at the network edge).

The port does not raise for HTTP error statuses — a 403 is *data*
for the strategy to record as a per-item failure — but
transport-level failures (DNS, connect, timeout, reset) raise
``FetchError``, which strategies likewise convert to failure records.
"""

import re
from dataclasses import dataclass
from typing import Protocol

import httpx

#: Generous read timeout (published decks export slowly), tight connect.
DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
USER_AGENT = "school-index-corpus-sync/0.1"


class FetchError(Exception):
    """Transport-level download failure (no HTTP response received)."""


@dataclass(frozen=True, kw_only=True)
class RawResponse:
    """One HTTP response, reduced to what strategies decide on."""

    status_code: int
    headers: dict[str, str]
    content: bytes

    def header(self, name: str) -> str | None:
        """Case-insensitive header lookup (wire case is not stable)."""
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None

    @property
    def content_type(self) -> str:
        """Content-Type without parameters, lowercased ("" if absent)."""
        raw = self.header("content-type") or ""
        return raw.split(";", 1)[0].strip().lower()

    @property
    def filename(self) -> str | None:
        """The Content-Disposition filename, if the server offered one."""
        disposition = self.header("content-disposition") or ""
        match = re.search(r'filename="?([^";]+)"?', disposition)
        return match.group(1) if match else None


class Downloader(Protocol):
    """The network edge, as a port. Implementations: HttpDownloader."""

    def get(self, *, url: str) -> RawResponse:
        """GET ``url`` (redirects followed). Raises FetchError only."""
        ...


class HttpDownloader:
    """Downloader over httpx. The client is injectable for tests."""

    def __init__(self, *, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            follow_redirects=True,
            timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )

    def get(self, *, url: str) -> RawResponse:
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise FetchError(f"{type(exc).__name__}: {exc}") from exc
        return RawResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            content=response.content,
        )
