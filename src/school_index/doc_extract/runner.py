"""The extraction run.

Pattern: Pipeline (orchestration service). One ``run()``:

1. list every corpus item (through corpus_sync's repository);
2. skip items sync never landed (failed / tombstoned — counted,
   not considered: there is no blob to extract);
3. skip items whose stored outcome already covers the current
   ``content_hash`` + format (delta-only re-runs);
4. dispatch each remaining item through the format registry,
   fold the outcome into the store as a record;
5. record the run in ``sync_state`` and return an ``ExtractReport``.

Per-item containment: a corrupt file, a missing blob, an extractor
failure — each becomes the item's stored FAILED record and the run
continues. The containment is total, not just over the extractor:
an *unexpected* exception anywhere in one item's processing is
caught by the run loop and folded into a FAILED record the same
way (a live run died exactly there before this guard existed —
a store write raised on one item's text and aborted the pass).
Only a failure of the failure-record write itself propagates: at
that point the store, not the item, is broken. Failures are the
one outcome that never delta-skips: the next run retries them,
because a re-harvest or an extractor fix may have removed the
cause.
"""

from collections.abc import Callable, Mapping
from datetime import datetime, timezone

from school_index.corpus_sync.domain import CorpusItem, ItemStatus, SyncState
from school_index.corpus_sync.lookup import Found
from school_index.corpus_sync.store import CorpusStore
from school_index.doc_extract.domain import (
    ExtractionOutcome,
    ExtractionState,
    ExtractReport,
)
from school_index.doc_extract.extractors import (
    ExtractedText,
    ExtractionFailed,
    Extractor,
    error_text,
    render_pages,
)
from school_index.doc_extract.store import ExtractStore

#: sync_state key for this stage (one health row per pipeline stage).
SOURCE_NAME = "doc-extract"


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


class ExtractionRunner:
    """Runs extraction passes. All collaborators are injected."""

    def __init__(
        self,
        *,
        corpus_store: CorpusStore,
        extract_store: ExtractStore,
        extractors: Mapping[str, Extractor],
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._corpus_store = corpus_store
        self._extract_store = extract_store
        self._extractors = extractors
        self._now = now or (lambda: datetime.now(timezone.utc))

    def run(self) -> ExtractReport:
        """Extract every corpus item that needs it; returns the report."""
        at = _iso(self._now())
        items = self._corpus_store.all_items()
        extracted = empty = failed = not_extractable = skipped = 0
        not_harvested = 0
        failures: list[tuple[str, str]] = []
        for item in items:
            if item.status is not ItemStatus.OK:
                not_harvested += 1
                continue
            if self._already_current(item):
                skipped += 1
                continue
            try:
                outcome, error = self._extract_one(item, at=at)
            except Exception as exc:  # containment: one item, never the run
                outcome, error = self._record_unexpected(item, exc, at=at)
            match outcome:
                case ExtractionOutcome.EXTRACTED:
                    extracted += 1
                case ExtractionOutcome.EMPTY:
                    empty += 1
                case ExtractionOutcome.NOT_EXTRACTABLE:
                    not_extractable += 1
                case ExtractionOutcome.FAILED:
                    failed += 1
                    if error is not None:
                        failures.append((item.id, error))
        counts = {
            "total": len(items),
            "extracted": extracted,
            "empty": empty,
            "failed": failed,
            "not_extractable": not_extractable,
            "skipped": skipped,
            "not_harvested": not_harvested,
        }
        self._corpus_store.record_sync(
            SyncState(
                source=SOURCE_NAME,
                watermark=None,
                last_run_at=at,
                last_status="ok" if failed == 0 else "partial",
                counts=counts,
            )
        )
        return ExtractReport(
            total=len(items),
            extracted=extracted,
            empty=empty,
            failed=failed,
            not_extractable=not_extractable,
            skipped=skipped,
            not_harvested=not_harvested,
            failures=tuple(failures),
        )

    # -- per-item work ---------------------------------------------------

    def _record_unexpected(
        self, item: CorpusItem, exc: Exception, *, at: str
    ) -> tuple[ExtractionOutcome, str]:
        """Fold an unexpected per-item exception into a FAILED record.

        The strategy name is recorded when the item's format has one
        (the failure happened in or around it). If this record write
        raises, it propagates — the store itself is broken, and a
        run that cannot record its failures must not pretend to
        have run.
        """
        error = error_text(exc)
        extractor = self._extractors.get(item.format)
        self._record(
            item,
            outcome=ExtractionOutcome.FAILED,
            extractor=extractor.name if extractor is not None else None,
            error=error,
            at=at,
        )
        return ExtractionOutcome.FAILED, error

    def _already_current(self, item: CorpusItem) -> bool:
        """True when a stored outcome already covers this blob revision."""
        state = self._extract_store.get_state(item_id=item.id)
        if not isinstance(state, Found):
            return False
        record = state.value
        return (
            record.outcome is not ExtractionOutcome.FAILED
            and record.source_hash == item.content_hash
            and record.format == item.format
        )

    def _extract_one(
        self, item: CorpusItem, *, at: str
    ) -> tuple[ExtractionOutcome, str | None]:
        """Dispatch, extract, record — one item, fully contained.

        Returns the outcome and the failure text (FAILED only).
        """
        extractor = self._extractors.get(item.format)
        if extractor is None:
            self._record(
                item,
                outcome=ExtractionOutcome.NOT_EXTRACTABLE,
                extractor=None,
                at=at,
            )
            return ExtractionOutcome.NOT_EXTRACTABLE, None
        blob = self._corpus_store.blob_path(item_id=item.id)
        if not blob.is_file():
            # Sync said OK and recorded a hash, but the bytes are not
            # on disk — an honest failure record beats a crash, and
            # the retry rule means a later run heals it if the blob
            # reappears.
            error = "blob missing on disk"
            self._record(
                item,
                outcome=ExtractionOutcome.FAILED,
                extractor=extractor.name,
                error=error,
                at=at,
            )
            return ExtractionOutcome.FAILED, error
        match extractor.extract(path=blob):
            case ExtractedText(pages=pages) as result:
                if result.is_empty:
                    self._record(
                        item,
                        outcome=ExtractionOutcome.EMPTY,
                        extractor=extractor.name,
                        page_count=result.page_count,
                        text="",
                        at=at,
                    )
                    return ExtractionOutcome.EMPTY, None
                rendered = render_pages(pages)
                self._record(
                    item,
                    outcome=ExtractionOutcome.EXTRACTED,
                    extractor=extractor.name,
                    page_count=result.page_count,
                    char_count=len(rendered),
                    text=rendered,
                    at=at,
                )
                return ExtractionOutcome.EXTRACTED, None
            case ExtractionFailed(error=error):
                self._record(
                    item,
                    outcome=ExtractionOutcome.FAILED,
                    extractor=extractor.name,
                    error=error,
                    at=at,
                )
                return ExtractionOutcome.FAILED, error

    def _record(
        self,
        item: CorpusItem,
        *,
        outcome: ExtractionOutcome,
        extractor: str | None,
        at: str,
        page_count: int | None = None,
        char_count: int = 0,
        error: str | None = None,
        text: str | None = None,
    ) -> None:
        self._extract_store.record_outcome(
            ExtractionState(
                item_id=item.id,
                outcome=outcome,
                extractor=extractor,
                source_hash=item.content_hash,
                format=item.format,
                char_count=char_count,
                page_count=page_count,
                error=error,
                extracted_at=at,
            ),
            text=text,
        )
