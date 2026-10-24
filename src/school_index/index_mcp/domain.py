"""Index query records: what the search MCP serves.

Pattern: frozen dataclasses (keyword-only construction). These are
the read side's vocabulary — a ``Passage`` is one stored chunk
joined with the provenance of the corpus item it came from, and a
``TopicResource`` is the per-item aggregate that
``resources_for_topic`` ranks. Both carry the item's ``deep_link``:
the pointer consumers cite (design: provenance in every result).
"""

import re
from dataclasses import dataclass

_PAGE_HEADING = re.compile(r"^page (\d+)$")


@dataclass(frozen=True, kw_only=True)
class Passage:
    """One chunk + its item's provenance, with a similarity score.

    ``score`` is cosine similarity on the embed side's scale
    ([-1, 1], higher is nearer). ``heading`` is the chunk's
    structural anchor from the chunker (``"page N"`` for paged
    sources, else None); ``page`` is that anchor as an int when it
    has the page shape. ``first_heading`` / ``first_text`` are
    the item's FIRST chunk's anchor and text (lowest ordinal) —
    the presentation fallback's disambiguator (see
    ``index_mcp.presentation``); they are internal, never
    serialized as separate wire fields.
    """

    chunk_id: int
    item_id: str
    text: str
    heading: str | None
    score: float
    title: str
    course: str | None
    deep_link: str
    source_kind: str
    first_heading: str | None = None
    first_text: str = ""

    @property
    def page(self) -> int | None:
        """The chunk's starting page, when its heading names one."""
        if self.heading is None:
            return None
        match = _PAGE_HEADING.match(self.heading)
        return int(match.group(1)) if match else None


@dataclass(frozen=True, kw_only=True)
class TopicResource:
    """One item as a topic pointer: its chunks' aggregate standing.

    ``best_score`` is the item's single nearest chunk; ``chunk_hits``
    is how many of its chunks appeared in the candidate set;
    ``rank_score`` is the documented aggregate the ranking used
    (see index_mcp.queries) — carried on the record so the wire can
    show its work instead of hiding the ordering rule.
    ``first_heading`` / ``first_text`` are the item's first
    chunk's anchor and text (copied from its best hit, which
    carries them for every chunk of the item) — the presentation
    fallback's disambiguator, internal only.
    """

    item_id: str
    title: str
    course: str | None
    deep_link: str
    source_kind: str
    best_score: float
    chunk_hits: int
    rank_score: float
    first_heading: str | None = None
    first_text: str = ""
