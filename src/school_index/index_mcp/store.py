"""Read-only index store: vector search + provenance joins.

Pattern: Repository over an injected connection factory, mirroring
the pipeline stores — but read-only by construction: this module
issues SELECTs only. No migrate, no ensure_schema, no writes: the
MCP server must never mutate the corpus the pipeline owns, and a
missing schema is a deployment error surfaced loudly at open, not
papered over by creating empty tables.

The vector backend is the write side's own strategy (reused from
``embed_index.store``), detected from the database artifact itself:
the ``embeddings`` entry in sqlite_master says whether it is a vec0
virtual table or the BLOB fallback table, so the reader always
matches the writer without shared configuration. When the backend
is vec, connections load the sqlite-vec extension — vec0 tables are
unreadable without it (M3's recorded surprise), including for pure
SELECTs.

Concurrency: the corpus DB is in WAL mode (the writers' factories
set it — see ``corpus_sync.connect_file``), which is what lets
these reads run alongside pipeline writes on the one file. Only
items with status ``ok`` are served: a tombstoned or failed item
keeps its chunks until the next embed pass purges them, and the
index must not serve what sync already retracted.
"""

import sqlite3
from pathlib import Path

from school_index.corpus_sync.lookup import Found, Lookup, Missing
from school_index.corpus_sync.store import ConnectionFactory, connect_file
from school_index.embed_index.store import (
    BlobBackend,
    SqliteVecBackend,
    VectorBackend,
    load_vec,
)
from school_index.index_mcp.domain import Passage


class IndexNotReady(RuntimeError):
    """The corpus DB is absent or not in a servable state."""


def _detect_backend(connection: sqlite3.Connection, *, dim: int) -> VectorBackend | None:
    """Which backend wrote this DB's ``embeddings`` table (None: none yet)."""
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'embeddings'"
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return SqliteVecBackend(dim=dim) if "vec0" in str(row[0]).lower() else BlobBackend(dim=dim)


class IndexReadStore:
    """SELECT-only repository over chunks / embeddings / corpus_items."""

    def __init__(
        self, *, connection_factory: ConnectionFactory, backend: VectorBackend | None
    ) -> None:
        self._connect = connection_factory
        self._backend = backend

    @classmethod
    def open(cls, root: Path, *, dim: int) -> "IndexReadStore":
        """Open the corpus DB under ``root`` for reading.

        The directory is NOT created and the DB is NOT initialized:
        a missing corpus.db means the pipeline has never run here,
        which is a deployment error, not an empty index.
        """
        db_path = root / "corpus.db"
        if not db_path.is_file():
            raise IndexNotReady(
                f"no corpus.db under {root} — run the pipeline before serving"
            )
        probe = sqlite3.connect(db_path)
        try:
            backend = _detect_backend(probe, dim=dim)
        finally:
            probe.close()
        is_vec = isinstance(backend, SqliteVecBackend)

        def factory() -> sqlite3.Connection:
            connection = connect_file(db_path)
            if is_vec:
                load_vec(connection)
            return connection

        return cls(connection_factory=factory, backend=backend)

    @property
    def backend_name(self) -> str:
        """Which vector backend this store is reading (or "none")."""
        return self._backend.name if self._backend is not None else "none"

    def index_model_id(self) -> Lookup[str]:
        """The model the stored vectors belong to (from embed_state).

        Missing when nothing is embedded yet. More than one distinct
        model is an invalid index (the writer's runner refuses to
        build one): meeting it here raises, loudly, rather than
        silently searching an incomparable vector space.
        """
        connection = self._connect()
        rows = connection.execute(
            "SELECT DISTINCT model_id FROM embed_state"
        ).fetchall()
        if not rows:
            return Missing(reason="index holds no embedded items yet")
        if len(rows) > 1:
            models = sorted(str(row[0]) for row in rows)
            raise IndexNotReady(f"mixed-model index (models: {models}); rebuild it")
        return Found(value=str(rows[0][0]))

    def search(
        self,
        *,
        query_vector: tuple[float, ...],
        model_id: str,
        course: str | None,
        fetch_limit: int,
    ) -> list[Passage]:
        """Top-``fetch_limit`` passages by cosine score, best first.

        ``course`` filters on the item's course when given. The
        backend ranks chunk ids; one join then attaches each chunk's
        text + item provenance, and the ranking order is restored
        in Python (the join does not preserve it).
        """
        if self._backend is None or fetch_limit < 1:
            return []
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        ranked = self._backend.search(
            connection,
            query_vector=query_vector,
            model_id=model_id,
            limit=fetch_limit,
        )
        if not ranked:
            return []
        placeholders = ", ".join("?" for _ in ranked)
        sql = (
            "SELECT c.id AS chunk_id, c.item_id, c.heading, c.text, "
            "i.title, i.course, i.deep_link, i.source_kind "
            "FROM chunks c JOIN corpus_items i ON i.id = c.item_id "
            f"WHERE c.id IN ({placeholders}) AND i.status = 'ok'"
        )
        params: list[object] = [chunk_id for chunk_id, _score in ranked]
        if course is not None:
            sql += " AND i.course = ?"
            params.append(course)
        rows = connection.execute(sql, params).fetchall()
        by_id = {int(row["chunk_id"]): row for row in rows}
        return [
            Passage(
                chunk_id=chunk_id,
                item_id=str(by_id[chunk_id]["item_id"]),
                text=str(by_id[chunk_id]["text"]),
                heading=by_id[chunk_id]["heading"],
                score=score,
                title=str(by_id[chunk_id]["title"]),
                course=by_id[chunk_id]["course"],
                deep_link=str(by_id[chunk_id]["deep_link"]),
                source_kind=str(by_id[chunk_id]["source_kind"]),
            )
            for chunk_id, score in ranked
            if chunk_id in by_id
        ]
