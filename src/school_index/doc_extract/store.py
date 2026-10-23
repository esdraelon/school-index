"""Extraction store: item_text + extraction_state in the corpus DB.

Pattern: Repository over an injected connection factory, mirroring
``corpus_sync``'s CorpusStore (tests use an in-memory database;
production the same ``corpus.db`` file the harvester writes).

Layering: the corpus database is one store with one migration
chain (owned by ``corpus_sync.store`` — its v2 creates these
tables), but each pipeline stage owns its own repository over its
own tables. doc_extract reads ``corpus_items`` and the blob files
through the CorpusStore (corpus_sync's repository), and writes only
``item_text`` / ``extraction_state`` through this one. Stages never
reach into each other's SQL.

Write semantics — one outcome, one consistent pair of writes:
``record_outcome`` always upserts the state row; when text is
supplied it upserts ``item_text``, and when it is not (FAILED /
NOT_EXTRACTABLE) it deletes any ``item_text`` row, because text
extracted from a *previous* revision of the blob no longer
describes the item and must not be served as if it did.
"""

import sqlite3
from pathlib import Path

from school_index.corpus_sync.lookup import Found, Lookup, Missing
from school_index.corpus_sync.store import ConnectionFactory, connect_file, migrate
from school_index.doc_extract.domain import (
    ExtractionOutcome,
    ExtractionState,
    ItemText,
)


def _row_to_text(row: sqlite3.Row) -> ItemText:
    return ItemText(
        item_id=row["item_id"],
        text=row["text"],
        extractor=row["extractor"],
        extracted_at=row["extracted_at"],
    )


def _row_to_state(row: sqlite3.Row) -> ExtractionState:
    return ExtractionState(
        item_id=row["item_id"],
        outcome=ExtractionOutcome(row["outcome"]),
        extractor=row["extractor"],
        source_hash=row["source_hash"],
        format=row["format"],
        char_count=row["char_count"],
        page_count=row["page_count"],
        error=row["error"],
        extracted_at=row["extracted_at"],
    )


class ExtractStore:
    """Repository for item_text / extraction_state."""

    def __init__(self, *, connection_factory: ConnectionFactory) -> None:
        self._connect = connection_factory
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        migrate(connection)

    @classmethod
    def open(cls, root: Path) -> "ExtractStore":
        """Open (creating if needed) the store rooted at ``root``."""
        root.mkdir(parents=True, exist_ok=True)
        db_path = root / "corpus.db"

        def factory() -> sqlite3.Connection:
            return connect_file(db_path)

        return cls(connection_factory=factory)

    # -- reads ---------------------------------------------------------

    def get_text(self, *, item_id: str) -> Lookup[ItemText]:
        """An item's stored text (Lookup at the boundary)."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM item_text WHERE item_id = ?", (item_id,)
        ).fetchone()
        if row is None:
            return Missing(reason=f"no extracted text for item {item_id!r}")
        return Found(value=_row_to_text(row))

    def get_state(self, *, item_id: str) -> Lookup[ExtractionState]:
        """An item's extraction outcome record, if extraction ran."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM extraction_state WHERE item_id = ?", (item_id,)
        ).fetchone()
        if row is None:
            return Missing(reason=f"no extraction state for item {item_id!r}")
        return Found(value=_row_to_state(row))

    def outcome_counts(self) -> dict[str, int]:
        """Census: extraction_state rows tallied by outcome value.

        This is how "for every corpus item, text? empty? failed?
        not-extractable?" is answered from the store: join these
        counts with corpus_items (items without a state row were
        either never considered or not yet run).
        """
        connection = self._connect()
        rows = connection.execute(
            "SELECT outcome, COUNT(*) AS n FROM extraction_state GROUP BY outcome"
        ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    # -- writes --------------------------------------------------------

    def record_outcome(self, state: ExtractionState, *, text: str | None) -> None:
        """Store one item's outcome (and its text, when it has any).

        ``text=None`` means "no current text" (FAILED /
        NOT_EXTRACTABLE): the state row is upserted and any stale
        ``item_text`` row from a previous blob revision is deleted.
        ``text=""`` is a real write — the EMPTY outcome's row.
        """
        connection = self._connect()
        connection.execute(
            """
            INSERT INTO extraction_state (
                item_id, outcome, extractor, source_hash, format,
                char_count, page_count, error, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                outcome = excluded.outcome,
                extractor = excluded.extractor,
                source_hash = excluded.source_hash,
                format = excluded.format,
                char_count = excluded.char_count,
                page_count = excluded.page_count,
                error = excluded.error,
                extracted_at = excluded.extracted_at
            """,
            (
                state.item_id,
                state.outcome.value,
                state.extractor,
                state.source_hash,
                state.format,
                state.char_count,
                state.page_count,
                state.error,
                state.extracted_at,
            ),
        )
        if text is None:
            connection.execute(
                "DELETE FROM item_text WHERE item_id = ?", (state.item_id,)
            )
        else:
            if state.extractor is None:
                raise ValueError("text without an extractor is not recordable")
            connection.execute(
                """
                INSERT INTO item_text (item_id, text, extractor, extracted_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    text = excluded.text,
                    extractor = excluded.extractor,
                    extracted_at = excluded.extracted_at
                """,
                (state.item_id, text, state.extractor, state.extracted_at),
            )
        connection.commit()
