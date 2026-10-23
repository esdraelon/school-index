"""Embed domain records: chunks, per-item embed state, run reports.

Pattern: frozen dataclasses (keyword-only construction). These are
the package's vocabulary — ``Chunk`` is a chunker's output (no store
identity yet), ``StoredChunk`` is a ``chunks`` row, ``EmbedState``
is the per-item record the delta logic keys on, and a run is
summarized as an ``EmbedReport``. No I/O lives here.

The state record is the data discipline made concrete, same as
M2's extraction_state: "has this item been embedded, from which
extraction, under which model, into how many chunks?" is answered
from the record, never from recollection. ``source_hash`` copies
the extraction state's ``source_hash`` (the corpus content hash the
current text was extracted from): text and index can each be
current-or-stale independently, and the pair of records says which.
"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Chunk:
    """One chunk of an item's text, as the chunker emits it.

    ``ordinal`` is 0-based within the item and stable: chunking is
    deterministic (same text + same config → same chunks), because
    incremental re-embedding keys off chunk identity per item.
    ``heading`` is the chunk's structural anchor — for v1, the page
    it starts on, rendered ``"page N"`` from the ``[[page N]]``
    marker contract; None when the text carried no page markers.
    """

    item_id: str
    ordinal: int
    heading: str | None
    text: str


@dataclass(frozen=True, kw_only=True)
class StoredChunk:
    """One row of ``chunks``: a Chunk plus its store identity."""

    id: int
    item_id: str
    ordinal: int
    heading: str | None
    text: str


@dataclass(frozen=True, kw_only=True)
class EmbedState:
    """One row of ``embed_state``: an item's embedding record.

    ``model_id`` is the provider that produced the current vectors.
    A store whose states name more than one model is an invalid
    index (vectors from different models are not comparable) — the
    runner refuses to add to it; changing models is the explicit
    ``--rebuild`` op. ``chunk_count`` is denormalized from
    ``chunks`` so health surfaces can census the index from state
    records alone.
    """

    item_id: str
    source_hash: str | None
    model_id: str
    chunk_count: int
    embedded_at: str


@dataclass(frozen=True, kw_only=True)
class SearchHit:
    """One search result: a stored chunk + its cosine score.

    ``score`` is cosine similarity in [-1, 1], higher is nearer —
    both vector backends speak this scale (the sqlite-vec backend
    converts its cosine distance). Consumers join provenance
    (title, course, deep link) from the corpus store; the hit
    itself carries only what the index knows.
    """

    chunk: StoredChunk
    score: float


@dataclass(frozen=True, kw_only=True)
class EmbedReport:
    """Aggregate outcome of one embed run (the CLI prints this).

    Counts are by *this run's* disposition: ``embedded`` items were
    (re-)chunked and (re-)embedded; ``skipped`` items were already
    current under the same model; ``purged`` items lost their
    chunks (tombstoned, vanished, or no longer extractable);
    ``not_embeddable`` items are sync-OK but their extraction
    outcome (failed / not-extractable / never extracted) gives the
    index nothing valid to embed. ``total`` is every row in
    ``corpus_items``; ``chunks_total`` is the index's size after
    the run. ``failures`` keeps every (item_id, error) pair so
    callers can tally reasons.
    """

    total: int
    embedded: int
    skipped: int
    purged: int
    failed: int
    not_embeddable: int
    chunks_written: int
    chunks_total: int
    failures: tuple[tuple[str, str], ...] = ()

    def failure_tally(self) -> dict[str, int]:
        """Failure reasons tallied, e.g. ``{"ModelNotDownloaded": 3}``."""
        tally: dict[str, int] = {}
        for _item_id, error in self.failures:
            tally[error] = tally.get(error, 0) + 1
        return tally
