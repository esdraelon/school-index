"""Corpus domain records: what a resource is, what the store holds.

Pattern: frozen dataclasses (keyword-only construction) + a StrEnum
for item status. These are the package's vocabulary — the client
translates the adapter's wire shape into ``ResourceRef``, the store
persists ``CorpusItem``, and a run is summarized as a ``SyncReport``.
No I/O lives here.

Identity: a resource is identified by ``source_ref`` =
``"<student>:<url>"`` — the stable key the design calls for (student +
url). The store primary key ``id`` is the SHA-256 of the source_ref:
deterministic across runs and machines, so re-harvests upsert the
same rows instead of duplicating them.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class ItemStatus(StrEnum):
    """Lifecycle of a corpus item (stored by value in SQLite)."""

    OK = "ok"
    FAILED = "failed"
    TOMBSTONED = "tombstoned"


@dataclass(frozen=True, kw_only=True)
class ResourceRef:
    """One resource as the adapter's ``get_resources`` reports it.

    Pattern: anti-corruption record — the wire shape (nested student /
    course objects) is flattened once, here at the boundary, and the
    rest of the package never sees it. ``title`` is whitespace-stripped
    and may legitimately be empty (the live corpus has blank titles);
    ``course_name`` is None when the adapter reports no course.
    """

    student_id: str
    student_name: str
    course_name: str | None
    title: str
    kind: str
    url: str
    source_kind: str
    scraped_at: str

    @property
    def source_ref(self) -> str:
        """Stable identity key: student + url."""
        return f"{self.student_id}:{self.url}"

    @property
    def item_id(self) -> str:
        """Store primary key: SHA-256 of the source_ref."""
        return hashlib.sha256(self.source_ref.encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class CorpusItem:
    """One row of ``corpus_items`` (see store.py for the schema).

    ``deep_link`` is the pointer consumers cite — for v1 sources it is
    the resource's own URL. ``origin_url`` records where the item came
    from (also the resource URL in v1; the export/download URL actually
    fetched is a fetch-time detail, not provenance). ``content_hash``
    is the SHA-256 of the stored bytes, None for link-only and
    never-downloaded items. ``fetched_at`` is the last fetch attempt
    (success or failure), ISO-8601 UTC.
    """

    id: str
    source_kind: str
    source_ref: str
    student: str
    course: str | None
    title: str
    deep_link: str
    origin_url: str
    format: str
    content_hash: str | None
    fetched_at: str
    status: ItemStatus
    error: str | None


@dataclass(frozen=True, kw_only=True)
class SyncState:
    """One row of ``sync_state``: the per-source health record.

    Mirrors the adapter's per-source health shape — a source's last
    run, its outcome, and aggregate counts (JSON in the store, a
    Mapping here). ``watermark`` is the newest ``scraped_at`` the run
    consumed, i.e. how current the source data was.
    """

    source: str
    watermark: str | None
    last_run_at: str
    last_status: str
    counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class SyncReport:
    """Aggregate outcome of one harvest run (the CLI prints this).

    Counts are by *transition*, not by final state: ``downloaded``
    counts items whose stored bytes were written this run,
    ``unchanged`` counts items the run proved identical (or already
    link-only) and left untouched, so a healthy re-run reports
    downloaded=0 / tombstoned=0. ``failures`` keeps every
    (source_ref, error) pair so callers can tally reasons.
    """

    student: str
    total: int
    downloaded: int
    link_only: int
    unchanged: int
    failed: int
    tombstoned: int
    failures: tuple[tuple[str, str], ...] = ()

    def failure_tally(self) -> dict[str, int]:
        """Failure reasons tallied, e.g. ``{"HTTP 403": 12}``."""
        tally: dict[str, int] = {}
        for _ref, error in self.failures:
            tally[error] = tally.get(error, 0) + 1
        return tally
