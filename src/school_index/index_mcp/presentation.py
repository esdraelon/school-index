"""Presentation titles: what the tutor shows when a title is ambiguous.

Pattern: pure presentation functions (no I/O, no store) — the
serializer calls these over the window it is about to emit, so
the wire shape stays exactly the tutor contract's (the fallback
replaces the ``title`` value, it never adds a wire field).

Ground truth that shaped this (live corpus, 2026-10-24): exactly
one chunked item has an empty title, and the duplicated chunked
titles are ``KEY`` x5, ``Stations`` x2, ``Drive file`` x2 — and
every one of those items' first chunks is headed ``page 1``.
Course + first-chunk heading therefore does NOT disambiguate on
its own. The fallback below is the task's allowed equivalent:
course + first-chunk heading + a snippet of the first chunk's
text, all fields already present on the record. The five KEY
openings diverge inside the snippet ("Writing Equations in
standard form", "Word problems involving rate of change", ...),
as do both other pairs.

Scope, stated so a reviewer can disagree on purpose: duplication
is judged within the presented window — the list the tutor
actually shows. A singleton whose title exists elsewhere in the
corpus keeps its raw title, because its pointer (deep_link /
item_id) is unambiguous; the fixed ambiguity is collisions in a
presented list plus empty titles, which are ambiguous alone.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")

#: Snippet length: long enough for the live duplicate pairs to
#: diverge (the Stations pair diverges at the colony name, the
#: KEY pair inside the first line), short enough to speak.
_SNIPPET_CHARS = 80


@dataclass(frozen=True, kw_only=True)
class TitleContext:
    """The fields a display title may be built from (all on-record)."""

    item_id: str
    title: str
    course: str | None
    first_heading: str | None
    first_text: str


def snippet(text: str, *, limit: int = _SNIPPET_CHARS) -> str:
    """First ``limit`` chars of ``text``, whitespace-collapsed.

    Cuts at the last word boundary inside the limit when the text
    is longer, with an ellipsis marking the cut; short text is
    returned whole. Deterministic: same text, same snippet.
    """
    collapsed = _WHITESPACE.sub(" ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed.rfind(" ", 0, limit)
    if cut < limit // 2:
        cut = limit
    return collapsed[:cut].rstrip() + "…"


def fallback_title(context: TitleContext) -> str:
    """The disambiguated title for an empty or duplicated title.

    Duplicated titles keep the raw title as a prefix so the
    result stays recognizable; empty titles have no prefix to
    keep. The course segment is always present ("No course"
    when the record has none), so the core is never empty.
    """
    segments = [context.course or "No course"]
    if context.first_heading:
        segments.append(context.first_heading)
    text_snippet = snippet(context.first_text)
    if text_snippet:
        segments.append(text_snippet)
    core = " · ".join(segments)
    if context.title:
        return f"{context.title} — {core}"
    return core


def resolve_display_titles(
    contexts: Sequence[TitleContext],
) -> dict[str, str]:
    """Display title per item id for one presented window.

    Raw titles are kept when non-empty and unique in the window;
    empty titles and in-window duplicates fall back. Multiple
    rows for one item (several passages of the same item) map to
    the same display title, keyed by item id.
    """
    counts: dict[str, int] = {}
    for context in contexts:
        if context.title:
            counts[context.title] = counts.get(context.title, 0) + 1
    resolved: dict[str, str] = {}
    for context in contexts:
        if not context.title or counts.get(context.title, 0) > 1:
            resolved[context.item_id] = fallback_title(context)
        else:
            resolved[context.item_id] = context.title
    return resolved
