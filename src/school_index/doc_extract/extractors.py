"""Extractors: one strategy per format, dispatched by a registry.

Pattern: Strategy + Registry. An ``Extractor`` turns one file on
disk into raw per-page text; the runner dispatches through an
injected ``Mapping[str, Extractor]`` keyed by the item's stored
format — there is no format if/else anywhere. Adding a format
(html, docx, native slides) means adding a strategy to the Mapping
the composition root builds (``build_extractors``); the runner and
the store do not change. An item whose format has no registered
strategy is not an error — it is the NOT_EXTRACTABLE outcome.

Outcomes are a sum type — ``ExtractedText`` / ``ExtractionFailed`` —
consumed by exhaustive ``match`` in the runner. A corrupt file is
*data* (it becomes the item's stored failure record), never an
exception that could abort the run.

The PDF text contract: pages render in reading order, each opened
by an explicit ``[[page N]]`` boundary marker (see
``render_pages``) — M3's structure-aware chunker splits on those
markers, so the marker form is a cross-milestone contract, not a
formatting choice. Whitespace inside a page is pypdf's, preserved
as-is: extraction does not over-normalize what downstream stages
may need to see verbatim.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

from pypdf import PdfReader


@dataclass(frozen=True, kw_only=True)
class ExtractedText:
    """Raw per-page text in reading order. Pattern: sum-type variant.

    ``pages`` holds one string per page (``""`` for a page with no
    text layer). Page granularity is the structure the corpus
    actually has; rendering to the marked-up stored form is a
    separate pure step (``render_pages``) so the runner can tell
    "valid file, zero text" (all pages blank) from real content.
    """

    pages: tuple[str, ...]

    @property
    def page_count(self) -> int:
        """Number of pages the extractor walked."""
        return len(self.pages)

    @property
    def is_empty(self) -> bool:
        """True when no page carries any non-whitespace text."""
        return all(not page.strip() for page in self.pages)


@dataclass(frozen=True, kw_only=True)
class ExtractionFailed:
    """Extraction was attempted and could not produce text."""

    error: str


#: What an extractor returns for one file.
ExtractOutcome: TypeAlias = ExtractedText | ExtractionFailed


def render_pages(pages: Iterable[str]) -> str:
    """Render per-page text to the stored form with page markers.

    Each page becomes ``[[page N]]`` on its own line followed by the
    page's text verbatim; pages join with a single newline. This is
    the M3 chunker's input contract — change it only with M3.
    """
    return "\n".join(
        f"[[page {number}]]\n{text}"
        for number, text in enumerate(pages, start=1)
    )


class Extractor(Protocol):
    """One format + how to pull its text layer out of a file."""

    name: str

    def extract(self, *, path: Path) -> ExtractOutcome:
        """Extract one file; failures are outcomes, not exceptions."""
        ...


def _error_text(exc: Exception) -> str:
    """One-line, length-capped failure text for the stored record.

    Failure strings are tallied for display, so they must be stable
    and single-line: exception type + first line of its message.
    """
    message = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {message}"[:200]


class PdfExtractor:
    """PDF text layer via pypdf. Pattern: Strategy.

    Pure-Python on purpose (the corpus lives on ARM). Reading is
    total: any pypdf failure — invalid header, broken xref,
    undecryptable content — folds into ``ExtractionFailed`` so one
    corrupt blob can never abort a run.
    """

    name = "pypdf"

    def extract(self, *, path: Path) -> ExtractOutcome:
        try:
            reader = PdfReader(str(path))
            pages = tuple(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # containment is the contract here
            return ExtractionFailed(error=_error_text(exc))
        return ExtractedText(pages=pages)


def build_extractors() -> Mapping[str, Extractor]:
    """The production registry: stored format → strategy.

    v1 has exactly one real format in the corpus — ``pdf`` (the
    harvester's exports and Drive downloads all land as PDF). The
    other formats the design names (html, docx, native slides) slot
    in here as new entries when the corpus actually grows them;
    until then their items resolve to NOT_EXTRACTABLE, which is a
    stored, countable outcome — not a silent gap.
    """
    return {"pdf": PdfExtractor()}
