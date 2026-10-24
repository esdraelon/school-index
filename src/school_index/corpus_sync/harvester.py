"""The harvest run.

Pattern: Pipeline (orchestration service). One ``run(student)``:

1. pull the resource list through the ResourceSource port;
2. dispatch each resource to the first matching download strategy;
3. fold the outcome into the store, idempotently;
4. tombstone items whose resource vanished;
5. record the run in ``sync_state`` and return a ``SyncReport``.

Per-item failures are contained: each becomes the item's stored
error record and the run continues. Only contract-level failure
(the source itself raising) aborts the run — then the run's picture
of the world is untrustworthy, and tombstoning on a partial list
would be destructive.

Idempotency: an item whose stored hash equals the freshly
downloaded hash keeps its blob and its row timestamps; only drifted
metadata (title/course) is refreshed. A second run over an unchanged
world therefore writes no bytes and changes no hashes.
"""

import hashlib
from collections.abc import Callable, Iterable
from datetime import datetime, timezone

from school_index.corpus_sync.client import ResourceSource
from school_index.corpus_sync.domain import (
    CorpusItem,
    ItemStatus,
    ResourceRef,
    SyncReport,
    SyncState,
)
from school_index.corpus_sync.fetch import Downloader
from school_index.corpus_sync.lookup import Found
from school_index.corpus_sync.store import CorpusStore
from school_index.corpus_sync.strategies import (
    DownloadFailed,
    Downloaded,
    DownloadStrategy,
    LinkOnly,
)

#: sync_state key for the adapter source (one row per source, ever).
SOURCE_NAME = "school-mcp"


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


class Harvester:
    """Runs syncs. All collaborators are injected (composition root)."""

    def __init__(
        self,
        *,
        source: ResourceSource,
        downloader: Downloader,
        strategies: Iterable[DownloadStrategy],
        store: CorpusStore,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._source = source
        self._downloader = downloader
        self._strategies = tuple(strategies)
        self._store = store
        self._now = now or (lambda: datetime.now(timezone.utc))

    def run(self, *, student: str) -> SyncReport:
        """Harvest one student's resources; returns the run report.

        Tombstoning requires an exhaustive listing (see
        ListedResources): a run that cannot prove the world it
        saw is the whole world tombstones nothing — an item
        absent from a partial listing is "unproven", never
        "vanished". That is the 2026-10-24 lesson: the old
        single-call harvest tombstoned live items that had
        drifted past the adapter's title-200 boundary.
        """
        at = _iso(self._now())
        listed = self._source.list_resources(student=student)
        resources = listed.resources
        seen: set[str] = set()
        downloaded = link_only = unchanged = failed = 0
        failures: list[tuple[str, str]] = []
        for resource in resources:
            if resource.source_ref in seen:
                # The payload repeats resources — one team-site link
                # is attached to 101 of the 200 live entries. The
                # first occurrence this run already synced the row;
                # a repeat is unchanged by construction and must not
                # be re-fetched.
                unchanged += 1
                continue
            seen.add(resource.source_ref)
            strategy = self._strategy_for(resource.url)
            outcome = strategy.fetch(resource=resource, downloader=self._downloader)
            match outcome:
                case Downloaded(content=content, format=fmt):
                    if self._absorb_download(resource, content, fmt, at=at):
                        downloaded += 1
                    else:
                        unchanged += 1
                case LinkOnly(format=fmt):
                    if self._absorb_link_only(resource, fmt, at=at):
                        link_only += 1
                    else:
                        unchanged += 1
                case DownloadFailed(error=error):
                    self._absorb_failure(resource, error, at=at)
                    failed += 1
                    failures.append((resource.source_ref, error))
        tombstoned = (
            self._store.tombstone_missing(student=student, seen_refs=seen, at=at)
            if listed.exhaustive
            else 0
        )
        counts = {
            "total": len(resources),
            "downloaded": downloaded,
            "link_only": link_only,
            "unchanged": unchanged,
            "failed": failed,
            "tombstoned": tombstoned,
            "partition_union": len(resources),
            "probe_distinct": listed.probe_distinct,
            "partition_calls": listed.partition_calls,
        }
        self._store.record_sync(
            SyncState(
                source=SOURCE_NAME,
                watermark=max((r.scraped_at for r in resources), default=None),
                last_run_at=at,
                last_status=(
                    "ok" if failed == 0 and listed.exhaustive else "partial"
                ),
                counts=counts,
            )
        )
        return SyncReport(
            student=student,
            total=len(resources),
            downloaded=downloaded,
            link_only=link_only,
            unchanged=unchanged,
            failed=failed,
            tombstoned=tombstoned,
            failures=tuple(failures),
            exhaustive=listed.exhaustive,
            probe_distinct=listed.probe_distinct,
            partition_calls=listed.partition_calls,
        )

    # -- outcome folding ------------------------------------------------

    def _strategy_for(self, url: str) -> DownloadStrategy:
        for strategy in self._strategies:
            if strategy.matches(url=url):
                return strategy
        raise RuntimeError(  # the production chain ends in a catch-all
            f"strategy chain is not total: nothing matches {url!r}"
        )

    def _item(
        self,
        resource: ResourceRef,
        *,
        status: ItemStatus,
        format: str,
        content_hash: str | None,
        error: str | None,
        at: str,
    ) -> CorpusItem:
        return CorpusItem(
            id=resource.item_id,
            source_kind=resource.source_kind,
            source_ref=resource.source_ref,
            student=resource.student_id,
            course=resource.course_name,
            title=resource.title,
            deep_link=resource.url,
            origin_url=resource.url,
            format=format,
            content_hash=content_hash,
            fetched_at=at,
            status=status,
            error=error,
        )

    def _absorb_download(
        self, resource: ResourceRef, content: bytes, format: str, *, at: str
    ) -> bool:
        """Store downloaded bytes unless identical. True if bytes written."""
        digest = hashlib.sha256(content).hexdigest()
        existing = self._store.get_item(source_ref=resource.source_ref)
        if isinstance(existing, Found):
            item = existing.value
            if (
                item.status is ItemStatus.OK
                and item.content_hash == digest
                and item.format == format
            ):
                if (item.title, item.course, item.source_kind) != (
                    resource.title,
                    resource.course_name,
                    resource.source_kind,
                ):
                    # Metadata drift only: refresh the row, keep the blob
                    # (and its mtime) — not a download.
                    self._store.upsert_item(
                        self._item(
                            resource,
                            status=ItemStatus.OK,
                            format=format,
                            content_hash=digest,
                            error=None,
                            at=at,
                        )
                    )
                return False
        self._store.write_blob(item_id=resource.item_id, content=content)
        self._store.upsert_item(
            self._item(
                resource,
                status=ItemStatus.OK,
                format=format,
                content_hash=digest,
                error=None,
                at=at,
            )
        )
        return True

    def _absorb_link_only(self, resource: ResourceRef, format: str, *, at: str) -> bool:
        """Record a pointer-only item. True if the row changed state."""
        existing = self._store.get_item(source_ref=resource.source_ref)
        if isinstance(existing, Found):
            item = existing.value
            if (
                item.status is ItemStatus.OK
                and item.format == format
                and item.content_hash is None
            ):
                return False
        self._store.upsert_item(
            self._item(
                resource,
                status=ItemStatus.OK,
                format=format,
                content_hash=None,
                error=None,
                at=at,
            )
        )
        return True

    def _absorb_failure(self, resource: ResourceRef, error: str, *, at: str) -> None:
        """Record a failed fetch, preserving any previously stored bytes."""
        existing = self._store.get_item(source_ref=resource.source_ref)
        keep_hash = existing.value.content_hash if isinstance(existing, Found) else None
        keep_format = existing.value.format if isinstance(existing, Found) else ""
        self._store.upsert_item(
            self._item(
                resource,
                status=ItemStatus.FAILED,
                format=keep_format,
                content_hash=keep_hash,
                error=error,
                at=at,
            )
        )
