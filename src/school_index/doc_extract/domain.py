"""Extraction domain records: what extraction knows about an item.

Pattern: frozen dataclasses (keyword-only construction) + a StrEnum
for outcome. These are the package's vocabulary — ``ItemText`` is the
design's ``item_text`` row, ``ExtractionState`` is the per-item
outcome record, and a run is summarized as an ``ExtractReport``.
No I/O lives here.

The state record is the data discipline made concrete: for every
corpus item the store must answer "text? empty? failed?
not-extractable?" from records, never from recollection. The four
outcomes are deliberately distinct — an image-only deck (valid PDF,
zero extractable text) is ``EMPTY``, not a failure; a pointer-only
resource (no blob was ever harvested, by design) is
``NOT_EXTRACTABLE``, not a skip.
"""

from dataclasses import dataclass
from enum import StrEnum


class ExtractionOutcome(StrEnum):
    """What extraction concluded for one item (stored by value)."""

    EXTRACTED = "extracted"
    EMPTY = "empty"
    FAILED = "failed"
    NOT_EXTRACTABLE = "not-extractable"


@dataclass(frozen=True, kw_only=True)
class ItemText:
    """One row of ``item_text``: an item's extracted text.

    ``text`` is the extractor's rendered output — for PDFs, page
    texts joined by ``[[page N]]`` boundary markers (the contract
    M3's structure-aware chunker consumes). An ``EMPTY`` item has a
    row too, with ``text == ""``: presence-with-no-text is exactly
    what distinguishes "extracted, was empty" from "never ran".
    ``extractor`` is the strategy id that produced the text.
    """

    item_id: str
    text: str
    extractor: str
    extracted_at: str


@dataclass(frozen=True, kw_only=True)
class ExtractionState:
    """One row of ``extraction_state``: the outcome record for an item.

    ``source_hash`` is the ``corpus_items.content_hash`` the outcome
    was produced from — the delta key. A later run re-extracts only
    when the item's current hash differs (living decks change) or
    the outcome was ``FAILED`` (failures always retry: the cause may
    have been fixed by a re-harvest or an extractor fix). ``format``
    is the item's format at extraction time, part of the delta key
    because a format change re-routes the item to another strategy.
    ``char_count`` is the stored text's length (0 unless EXTRACTED);
    ``page_count`` is set by extractors that know it (PDFs), else
    None. ``extractor`` is None only for NOT_EXTRACTABLE.
    """

    item_id: str
    outcome: ExtractionOutcome
    extractor: str | None
    source_hash: str | None
    format: str
    char_count: int
    page_count: int | None
    error: str | None
    extracted_at: str


@dataclass(frozen=True, kw_only=True)
class ExtractReport:
    """Aggregate outcome of one extraction run (the CLI prints this).

    Counts are by *this run's* disposition of each item, so a
    healthy re-run over an unchanged corpus reports extracted=0 /
    skipped=<everything it considered>. ``total`` is every row in
    ``corpus_items``; ``not_harvested`` counts the rows extraction
    does not consider (sync status failed or tombstoned — countable
    in the corpus, but no blob exists to extract). ``failures``
    keeps every (item_id, error) pair so callers can tally reasons.
    """

    total: int
    extracted: int
    empty: int
    failed: int
    not_extractable: int
    skipped: int
    not_harvested: int
    failures: tuple[tuple[str, str], ...] = ()

    def failure_tally(self) -> dict[str, int]:
        """Failure reasons tallied, e.g. ``{"PdfReadError: ...": 2}``."""
        tally: dict[str, int] = {}
        for _item_id, error in self.failures:
            tally[error] = tally.get(error, 0) + 1
        return tally
