"""The embed run: chunk + embed whatever extraction changed.

Pattern: Pipeline (orchestration service). One ``run()``:

1. refuse (or, with ``rebuild``, clear) a mixed-model index —
   vectors from different models are not comparable, so adding to
   one silently would corrupt every future search;
2. purge index entries whose item vanished, was tombstoned, or is
   no longer extractable (its text is gone from ``item_text`` too —
   the index must not serve what extraction retracted);
3. for each remaining sync-OK item with an EXTRACTED or EMPTY
   outcome, compare its ``embed_state`` against the extraction
   record: same ``source_hash`` + same model → skip; otherwise
   re-chunk from the stored text, embed, and replace the item's
   chunks/vectors/state atomically;
4. record the run in ``sync_state`` and return an ``EmbedReport``.

Per-item containment mirrors M2: an exception chunking or embedding
one item becomes a counted failure in the report and the run
continues; the item keeps no new state, so the next run retries
it. A failure of the run-level guards (the model census) is not
contained — it means the index itself is in a state the run must
not paper over.
"""

from collections.abc import Callable
from datetime import datetime, timezone

from school_index.corpus_sync.domain import ItemStatus, SyncState
from school_index.corpus_sync.lookup import Found
from school_index.corpus_sync.store import CorpusStore
from school_index.doc_extract.domain import ExtractionOutcome
from school_index.doc_extract.store import ExtractStore
from school_index.embed_index.chunker import chunk_text
from school_index.embed_index.domain import EmbedReport, EmbedState
from school_index.embed_index.providers import EmbeddingProvider
from school_index.embed_index.store import EmbedStore

#: sync_state key for this stage (one health row per pipeline stage).
SOURCE_NAME = "embed-index"


class ModelMismatchError(RuntimeError):
    """The index holds vectors from another model (use --rebuild)."""


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _error_text(exc: Exception) -> str:
    message = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {message}"[:200]


class EmbedRunner:
    """Runs embed passes. All collaborators are injected."""

    def __init__(
        self,
        *,
        corpus_store: CorpusStore,
        extract_store: ExtractStore,
        embed_store: EmbedStore,
        provider: EmbeddingProvider,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._corpus_store = corpus_store
        self._extract_store = extract_store
        self._embed_store = embed_store
        self._provider = provider
        self._now = now or (lambda: datetime.now(timezone.utc))

    def run(self, *, rebuild: bool = False) -> EmbedReport:
        """Embed every item whose extraction changed; returns the report."""
        at = _iso(self._now())
        self._guard_single_model(rebuild=rebuild)
        items = self._corpus_store.all_items()
        live_ids = {item.id for item in items}
        embedded = skipped = purged = failed = not_embeddable = 0
        chunks_written = 0
        failures: list[tuple[str, str]] = []

        # Orphaned states: items the corpus no longer knows at all.
        for state in self._embed_store.embed_states():
            if state.item_id not in live_ids:
                self._embed_store.purge_item(item_id=state.item_id)
                purged += 1

        for item in items:
            if item.status is not ItemStatus.OK:
                if self._has_index(item.id):
                    self._embed_store.purge_item(item_id=item.id)
                    purged += 1
                continue
            extraction = self._extract_store.get_state(item_id=item.id)
            if not isinstance(extraction, Found) or extraction.value.outcome not in (
                ExtractionOutcome.EXTRACTED,
                ExtractionOutcome.EMPTY,
            ):
                not_embeddable += 1
                if self._has_index(item.id):
                    self._embed_store.purge_item(item_id=item.id)
                    purged += 1
                continue
            record = extraction.value
            current = self._embed_store.get_embed_state(item_id=item.id)
            if (
                not rebuild
                and isinstance(current, Found)
                and current.value.source_hash == record.source_hash
                and current.value.model_id == self._provider.model_id
            ):
                skipped += 1
                continue
            try:
                written = self._embed_one(item.id, source_hash=record.source_hash, at=at)
            except Exception as exc:  # containment: one item, never the run
                failed += 1
                failures.append((item.id, _error_text(exc)))
                continue
            embedded += 1
            chunks_written += written

        counts = {
            "total": len(items),
            "embedded": embedded,
            "skipped": skipped,
            "purged": purged,
            "failed": failed,
            "not_embeddable": not_embeddable,
            "chunks_total": self._embed_store.total_chunks(),
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
        return EmbedReport(
            total=len(items),
            embedded=embedded,
            skipped=skipped,
            purged=purged,
            failed=failed,
            not_embeddable=not_embeddable,
            chunks_written=chunks_written,
            chunks_total=counts["chunks_total"],
            failures=tuple(failures),
        )

    # -- internals ---------------------------------------------------------

    def _guard_single_model(self, *, rebuild: bool) -> None:
        """Enforce one model per index; --rebuild is the only reset."""
        models = {state.model_id for state in self._embed_store.embed_states()}
        if rebuild:
            if models:
                self._embed_store.clear_index()
            return
        foreign = models - {self._provider.model_id}
        if foreign:
            raise ModelMismatchError(
                f"index holds vectors from {sorted(foreign)}, this run is "
                f"{self._provider.model_id!r} — mixing models is invalid; "
                "re-run with --rebuild to re-embed everything"
            )

    def _has_index(self, item_id: str) -> bool:
        return isinstance(self._embed_store.get_embed_state(item_id=item_id), Found)

    def _embed_one(self, item_id: str, *, source_hash: str | None, at: str) -> int:
        """Chunk + embed + store one item; returns its chunk count."""
        text = self._extract_store.get_text(item_id=item_id)
        if not isinstance(text, Found):
            raise RuntimeError(f"extraction says embedded-able but no text for {item_id}")
        chunks = chunk_text(text.value.text, item_id=item_id)
        vectors = self._provider.embed_documents([chunk.text for chunk in chunks])
        self._embed_store.replace_item_chunks(
            chunks=chunks,
            vectors=vectors,
            state=EmbedState(
                item_id=item_id,
                source_hash=source_hash,
                model_id=self._provider.model_id,
                chunk_count=len(chunks),
                embedded_at=at,
            ),
        )
        return len(chunks)
