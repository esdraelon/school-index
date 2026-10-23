"""Corpus store: SQLite metadata + on-disk originals.

Pattern: Repository over an injected connection factory (tests use an
in-memory database; production a file under the corpus root — the
repository cannot tell), plus an ordered-migration runner mirroring
the adapter's store (``migrate`` runs on open, so repositories stay
free of schema bookkeeping).

Layout under the corpus root::

    corpus.db     metadata (this module's tables)
    files/<id>    original bytes per item, named by corpus_items.id

Storage conventions: datetimes are ISO-8601 UTC text; ``ItemStatus``
is stored by value; ``sync_state.counts`` is a JSON object of ints.
Blob writes are temp-file + rename, so a crashed run never leaves a
torn file behind under a good item id.
"""

import json
import os
import sqlite3
from collections.abc import Callable, Collection
from pathlib import Path

from school_index.corpus_sync.domain import CorpusItem, ItemStatus, SyncState
from school_index.corpus_sync.lookup import Found, Lookup, Missing

#: How repositories get connections: injected, per the pattern above.
ConnectionFactory = Callable[[], sqlite3.Connection]

_V1 = """
CREATE TABLE corpus_items (
    id           TEXT PRIMARY KEY,
    source_kind  TEXT NOT NULL,
    source_ref   TEXT NOT NULL UNIQUE,
    student      TEXT NOT NULL,
    course       TEXT,
    title        TEXT NOT NULL,
    deep_link    TEXT NOT NULL,
    origin_url   TEXT NOT NULL,
    format       TEXT NOT NULL,
    content_hash TEXT,
    fetched_at   TEXT NOT NULL,
    status       TEXT NOT NULL,
    error        TEXT
);
CREATE INDEX idx_corpus_items_student ON corpus_items (student);
CREATE INDEX idx_corpus_items_status ON corpus_items (status);

CREATE TABLE sync_state (
    source      TEXT PRIMARY KEY,
    watermark   TEXT,
    last_run_at TEXT NOT NULL,
    last_status TEXT NOT NULL,
    counts      TEXT NOT NULL
);
"""

_V2 = """
CREATE TABLE item_text (
    item_id      TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    extractor    TEXT NOT NULL,
    extracted_at TEXT NOT NULL
);

CREATE TABLE extraction_state (
    item_id      TEXT PRIMARY KEY,
    outcome      TEXT NOT NULL,
    extractor    TEXT,
    source_hash  TEXT,
    format       TEXT NOT NULL,
    char_count   INTEGER NOT NULL,
    page_count   INTEGER,
    error        TEXT,
    extracted_at TEXT NOT NULL
);
CREATE INDEX idx_extraction_state_outcome ON extraction_state (outcome);
"""

_V3 = """
CREATE TABLE chunks (
    id       INTEGER PRIMARY KEY,
    item_id  TEXT NOT NULL,
    ordinal  INTEGER NOT NULL,
    heading  TEXT,
    text     TEXT NOT NULL,
    UNIQUE (item_id, ordinal)
);
CREATE INDEX idx_chunks_item ON chunks (item_id);

CREATE TABLE embed_state (
    item_id     TEXT PRIMARY KEY,
    source_hash TEXT,
    model_id    TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    embedded_at TEXT NOT NULL
);
"""

#: Ordered (version, SQL) migrations. Append-only: never edit a shipped one.
#: One chain for the one database file: v2's tables are doc_extract's
#: (its repository lives there — see doc_extract/store.py), v3's are
#: embed_index's (chunks + embed_state; the vectors themselves live in
#: the vector backend's own table — a vec0 virtual table when
#: sqlite-vec loads, a BLOB table otherwise — created by the backend,
#: not here, because vec0's CREATE needs the extension loaded on the
#: connection, which is a runtime property, not schema history), but
#: the migration mechanics stay here so the file's schema history is a
#: single ordered story instead of two registries racing on one file.
MIGRATIONS: tuple[tuple[int, str], ...] = ((1, _V1), (2, _V2), (3, _V3))


def migrate(connection: sqlite3.Connection) -> None:
    """Apply every pending migration to ``connection`` (idempotent)."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {
        row[0] for row in connection.execute("SELECT version FROM schema_migrations")
    }
    for version, sql in MIGRATIONS:
        if version not in applied:
            connection.executescript(sql)
            connection.execute(
                "INSERT INTO schema_migrations (version, applied_at) "
                "VALUES (?, datetime('now'))",
                (version,),
            )
    connection.commit()


def _row_to_item(row: sqlite3.Row) -> CorpusItem:
    return CorpusItem(
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
        status=ItemStatus(row["status"]),
        error=row["error"],
    )


class CorpusStore:
    """Repository for corpus_items / sync_state + the blob files."""

    def __init__(
        self, *, connection_factory: ConnectionFactory, files_root: Path
    ) -> None:
        self._connect = connection_factory
        self._files_root = files_root
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        migrate(connection)

    @classmethod
    def open(cls, root: Path) -> "CorpusStore":
        """Open (creating if needed) the store rooted at ``root``."""
        root.mkdir(parents=True, exist_ok=True)
        db_path = root / "corpus.db"

        def factory() -> sqlite3.Connection:
            return sqlite3.connect(db_path)

        return cls(connection_factory=factory, files_root=root / "files")

    # -- corpus_items -------------------------------------------------

    def get_item(self, *, source_ref: str) -> Lookup[CorpusItem]:
        """Single-item read by identity key (Lookup at the boundary)."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM corpus_items WHERE source_ref = ?", (source_ref,)
        ).fetchone()
        if row is None:
            return Missing(reason=f"no corpus item for source_ref {source_ref!r}")
        return Found(value=_row_to_item(row))

    def upsert_item(self, item: CorpusItem) -> None:
        """Insert or replace the row for ``item.id``."""
        connection = self._connect()
        connection.execute(
            """
            INSERT INTO corpus_items (
                id, source_kind, source_ref, student, course, title,
                deep_link, origin_url, format, content_hash, fetched_at,
                status, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_kind = excluded.source_kind,
                source_ref = excluded.source_ref,
                student = excluded.student,
                course = excluded.course,
                title = excluded.title,
                deep_link = excluded.deep_link,
                origin_url = excluded.origin_url,
                format = excluded.format,
                content_hash = excluded.content_hash,
                fetched_at = excluded.fetched_at,
                status = excluded.status,
                error = excluded.error
            """,
            (
                item.id,
                item.source_kind,
                item.source_ref,
                item.student,
                item.course,
                item.title,
                item.deep_link,
                item.origin_url,
                item.format,
                item.content_hash,
                item.fetched_at,
                item.status.value,
                item.error,
            ),
        )
        connection.commit()

    def items_for_student(self, *, student: str) -> list[CorpusItem]:
        """All rows for a student (empty list when none — a collection)."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM corpus_items WHERE student = ? ORDER BY source_ref",
            (student,),
        ).fetchall()
        return [_row_to_item(row) for row in rows]

    def all_items(self) -> list[CorpusItem]:
        """Every row in the corpus, any student or status.

        The candidate query for downstream pipeline stages (M2's
        extraction): they scope by status/format themselves, so the
        store stays a dumb collection read.
        """
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM corpus_items ORDER BY source_ref"
        ).fetchall()
        return [_row_to_item(row) for row in rows]

    def tombstone_missing(
        self, *, student: str, seen_refs: Collection[str], at: str
    ) -> int:
        """Mark a student's unseen, still-active items as tombstoned.

        Keeps the rows (the design's tombstone rule: pointers fail
        honestly rather than silently). Returns how many were marked.
        """
        connection = self._connect()
        if seen_refs:
            placeholders = ", ".join("?" for _ in seen_refs)
            cursor = connection.execute(
                "UPDATE corpus_items SET status = ?, fetched_at = ? "
                "WHERE student = ? AND status != ? "
                f"AND source_ref NOT IN ({placeholders})",
                (
                    ItemStatus.TOMBSTONED.value,
                    at,
                    student,
                    ItemStatus.TOMBSTONED.value,
                    *seen_refs,
                ),
            )
        else:
            cursor = connection.execute(
                "UPDATE corpus_items SET status = ?, fetched_at = ? "
                "WHERE student = ? AND status != ?",
                (ItemStatus.TOMBSTONED.value, at, student, ItemStatus.TOMBSTONED.value),
            )
        connection.commit()
        return cursor.rowcount

    # -- blobs ---------------------------------------------------------

    def write_blob(self, *, item_id: str, content: bytes) -> None:
        """Store an item's original bytes (temp file + atomic rename)."""
        self._files_root.mkdir(parents=True, exist_ok=True)
        target = self._files_root / item_id
        temporary = self._files_root / f"{item_id}.tmp"
        temporary.write_bytes(content)
        os.replace(temporary, target)

    def blob_path(self, *, item_id: str) -> Path:
        """Where an item's bytes live (whether or not they exist)."""
        return self._files_root / item_id

    # -- sync_state ----------------------------------------------------

    def record_sync(self, state: SyncState) -> None:
        """Insert or replace the health row for ``state.source``."""
        connection = self._connect()
        connection.execute(
            """
            INSERT INTO sync_state (source, watermark, last_run_at, last_status, counts)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                watermark = excluded.watermark,
                last_run_at = excluded.last_run_at,
                last_status = excluded.last_status,
                counts = excluded.counts
            """,
            (
                state.source,
                state.watermark,
                state.last_run_at,
                state.last_status,
                json.dumps(dict(state.counts)),
            ),
        )
        connection.commit()

    def get_sync_state(self, *, source: str) -> Lookup[SyncState]:
        """Single-item read of a source's health row."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM sync_state WHERE source = ?", (source,)
        ).fetchone()
        if row is None:
            return Missing(reason=f"no sync state recorded for source {source!r}")
        counts = {str(k): int(v) for k, v in json.loads(row["counts"]).items()}
        return Found(
            value=SyncState(
                source=row["source"],
                watermark=row["watermark"],
                last_run_at=row["last_run_at"],
                last_status=row["last_status"],
                counts=counts,
            )
        )
