"""Read-only access to the production corpus DB.

Pattern: read-only Repository. Every connection here is opened
with SQLite's ``mode=ro`` URI — the benchmark physically cannot
mutate production tables, which is the M5 constraint ("the ONNX arm
exercises the live corpus read-only; benchmark state lives in its
own store"). Raw SQL rather than the pipeline's repositories for
the same reason: those repositories migrate on open, and a
benchmark must not write schema bookkeeping into a production DB
either. The census counts only status-ok items — the population
the search path can actually serve.
"""

import sqlite3
from pathlib import Path

from school_index.benchmark.domain import CorpusItemRow
from school_index.embed_index.domain import StoredChunk

_ITEM_COLUMNS = (
    "id, source_kind, source_ref, student, course, title, deep_link, "
    "origin_url, format, content_hash, fetched_at, status, error"
)


def _connect(root: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{root / 'corpus.db'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _row_to_item(row: sqlite3.Row) -> CorpusItemRow:
    return CorpusItemRow(
        id=row["id"],
        source_kind=row["source_kind"],
        source_ref=row["source_ref"],
        student=row["student"],
        course=row["course"],
        title=row["title"],
        deep_link=row["deep_link"],
        origin_url=row["origin_url"],
        format=row["format"],
        content_hash=row["content_hash"],
        fetched_at=row["fetched_at"],
        status=row["status"],
        error=row["error"],
    )


def read_items(root: Path) -> list[CorpusItemRow]:
    """Every corpus_items row, id order (all statuses)."""
    connection = _connect(root)
    rows = connection.execute(
        f"SELECT {_ITEM_COLUMNS} FROM corpus_items ORDER BY id"
    ).fetchall()
    return [_row_to_item(row) for row in rows]


def read_chunks(root: Path, *, item_id: str) -> list[StoredChunk]:
    """One item's stored chunks, ordinal order."""
    connection = _connect(root)
    rows = connection.execute(
        "SELECT * FROM chunks WHERE item_id = ? ORDER BY ordinal", (item_id,)
    ).fetchall()
    return [
        StoredChunk(
            id=int(row["id"]),
            item_id=row["item_id"],
            ordinal=int(row["ordinal"]),
            heading=row["heading"],
            text=row["text"],
        )
        for row in rows
    ]


def read_text(root: Path, *, item_id: str) -> str | None:
    """One item's extracted text, or None when it has none."""
    connection = _connect(root)
    row = connection.execute(
        "SELECT text FROM item_text WHERE item_id = ?", (item_id,)
    ).fetchone()
    return str(row["text"]) if row is not None else None


def census(root: Path) -> tuple[int, int]:
    """(items with >= 1 chunk, total chunks) over status-ok items."""
    connection = _connect(root)
    row = connection.execute(
        "SELECT COUNT(DISTINCT c.item_id), COUNT(*) FROM chunks c "
        "JOIN corpus_items i ON i.id = c.item_id WHERE i.status = 'ok'"
    ).fetchone()
    return int(row[0]), int(row[1])
