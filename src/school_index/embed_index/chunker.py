"""Structure-aware chunking over stored item text.

Pattern: pure pipeline (parse → pack → split). No I/O, no state:
the same text + config always yields the same chunks, in the same
order — determinism is a contract, because the runner's incremental
re-embedding keys off "this item's chunks" as a stable set.

Structure comes from M2's stored form: text is a sequence of pages
opened by ``[[page N]]`` marker lines (the cross-milestone contract
in ``doc_extract.extractors.render_pages``). Pages are the atomic
packing unit:

1. **Pack.** Whole pages accumulate into a chunk until adding the
   next would pass ``target_chars`` (~1,000); the chunk's text is
   the pages joined by a newline, its heading the first page's
   ``"page N"``.
2. **Split.** A single page longer than ``max_chars`` (~1,400) is
   split on sentence boundaries into ~target-sized pieces, each
   seeded with a ~``overlap_chars`` (~120) tail of the previous
   piece so a fact straddling the cut survives in both. A sentence
   longer than the target is hard-split at a word boundary (those
   pieces are then ordinary units for packing).

Text with no page markers (a non-PDF extractor's future output) is
one unheaded segment and goes through the same pack/split path.
Split pages have their internal whitespace normalized (sentences
rejoin with single spaces); unsplit pages are stored verbatim.
"""

import re
from dataclasses import dataclass

from school_index.embed_index.domain import Chunk

_PAGE_MARKER = re.compile(r"^\[\[page (\d+)\]\]\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, kw_only=True)
class ChunkConfig:
    """Chunk sizing: targets, not exactitudes (structure wins)."""

    target_chars: int = 1000
    max_chars: int = 1400
    overlap_chars: int = 120


#: The v1 sizing, named so the runner and tests share one value.
DEFAULT_CONFIG = ChunkConfig()


@dataclass(frozen=True, kw_only=True)
class _Segment:
    """One structural unit of stored text: a page, or markerless text."""

    page: int | None
    text: str


def _parse_segments(text: str) -> list[_Segment]:
    """Split stored text on its ``[[page N]]`` marker lines."""
    segments: list[_Segment] = []
    page: int | None = None
    lines: list[str] = []
    for line in text.splitlines():
        marker = _PAGE_MARKER.match(line)
        if marker is not None:
            body = "\n".join(lines).strip()
            if body:
                segments.append(_Segment(page=page, text=body))
            page = int(marker.group(1))
            lines = []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    if body:
        segments.append(_Segment(page=page, text=body))
    return segments


def _heading(page: int | None) -> str | None:
    return f"page {page}" if page is not None else None


def _tail_seed(piece: str, *, config: ChunkConfig) -> str:
    """The ~overlap-char tail of ``piece``, aligned to readable starts.

    Prefers starting right after a sentence end inside the tail;
    falls back to the first word boundary. Empty when the piece is
    too short to be worth echoing.
    """
    if len(piece) <= config.overlap_chars:
        return ""
    tail = piece[-config.overlap_chars :]
    sentence = _SENTENCE_END.search(tail)
    if sentence is not None:
        return tail[sentence.end() :].strip()
    boundary = tail.find(" ")
    return tail[boundary + 1 :].strip() if boundary >= 0 else tail


def _hard_split(sentence: str, *, config: ChunkConfig) -> list[str]:
    """Split one over-long sentence at word boundaries (~target each)."""
    parts: list[str] = []
    rest = sentence
    while len(rest) > config.target_chars:
        cut = rest.rfind(" ", 0, config.target_chars)
        if cut < config.target_chars // 2:
            cut = config.target_chars
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def _units(text: str, *, config: ChunkConfig) -> list[str]:
    """Sentence-sized units of one page; over-long sentences hard-split."""
    units: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= config.target_chars:
            units.append(sentence)
        else:
            units.extend(_hard_split(sentence, config=config))
    return units


def _split_page(text: str, *, config: ChunkConfig) -> list[str]:
    """Split one over-long page into ~target pieces with overlap."""
    pieces: list[str] = []
    current = ""
    for unit in _units(text, config=config):
        candidate = unit if not current else f"{current} {unit}"
        if len(candidate) <= config.target_chars:
            current = candidate
            continue
        pieces.append(current)
        seed = _tail_seed(current, config=config)
        current = f"{seed} {unit}" if seed else unit
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, *, item_id: str, config: ChunkConfig = DEFAULT_CONFIG) -> list[Chunk]:
    """Chunk one item's stored text. Pure and deterministic.

    Empty text (the EMPTY extraction outcome) yields no chunks —
    the item still gets an embed_state record from the runner, with
    ``chunk_count == 0``.
    """
    chunks: list[Chunk] = []
    buffer: list[_Segment] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        joined = "\n".join(segment.text for segment in buffer)
        chunks.append(
            Chunk(
                item_id=item_id,
                ordinal=len(chunks),
                heading=_heading(buffer[0].page),
                text=joined,
            )
        )
        buffer = []
        buffer_len = 0

    for segment in _parse_segments(text):
        if len(segment.text) > config.max_chars:
            flush()
            for piece in _split_page(segment.text, config=config):
                chunks.append(
                    Chunk(
                        item_id=item_id,
                        ordinal=len(chunks),
                        heading=_heading(segment.page),
                        text=piece,
                    )
                )
            continue
        projected = buffer_len + len(segment.text) + (1 if buffer else 0)
        if buffer and projected > config.target_chars:
            flush()
            projected = len(segment.text)
        buffer.append(segment)
        buffer_len = projected
    flush()
    return chunks
