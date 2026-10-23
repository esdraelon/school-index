"""Embed store: chunks + embed_state + vectors in the corpus DB.

Pattern: Repository over an injected connection factory, mirroring
the earlier stages (tests use an in-memory database; production the
same ``corpus.db`` file), with the vector persistence behind a
``VectorBackend`` strategy so the repository — and everything above
it — cannot tell which backend is in use:

- ``SqliteVecBackend`` (the design's store): a vec0 virtual table,
  cosine distance metric, KNN in the extension.
- ``BlobBackend`` (the fallback): vectors as BLOBs in a plain
  table, brute-force cosine in numpy. Chosen automatically when
  the sqlite-vec extension cannot load on this machine (probed at
  open); at corpus scale (low thousands of chunks) the two are
  indistinguishable in latency, and interface parity is what makes
  the substitution safe. ``store.backend_name`` reports which one
  a given store is actually running.

Layering: the schema history is migration v3 in
``corpus_sync.store`` (chunks + embed_state). The ``embeddings``
table is created by the backend, not the migration chain: a vec0
CREATE needs the extension loaded on the connection, which is a
runtime property, not schema history. Both backends call their
table ``embeddings``; a database is only ever opened with one
backend at a time.

Writes are transactional per item: replacing an item's chunks
deletes its old vectors, inserts the new chunks + vectors, and
upserts its embed_state as one commit — a crash never leaves
chunks without vectors or state without chunks.
"""

import sqlite3
import struct
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from school_index.corpus_sync.lookup import Found, Lookup, Missing
from school_index.corpus_sync.store import ConnectionFactory, migrate
from school_index.embed_index.domain import (
    Chunk,
    EmbedState,
    SearchHit,
    StoredChunk,
)
from school_index.embed_index.providers import Vector


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


class VectorBackend(Protocol):
    """How vectors persist + search. Pattern: Strategy."""

    name: str
    dim: int

    def ensure_schema(self, connection: sqlite3.Connection) -> None:
        """Create the backend's ``embeddings`` table if absent."""
        ...

    def put(
        self,
        connection: sqlite3.Connection,
        *,
        chunk_id: int,
        model_id: str,
        vector: Sequence[float],
    ) -> None:
        """Store one chunk's vector (replacing any existing row)."""
        ...

    def delete(
        self, connection: sqlite3.Connection, *, chunk_ids: Collection[int]
    ) -> None:
        """Drop the vectors for ``chunk_ids`` (missing ids are fine)."""
        ...

    def clear(self, connection: sqlite3.Connection) -> None:
        """Drop every vector (the --rebuild op's first step)."""
        ...

    def search(
        self,
        connection: sqlite3.Connection,
        *,
        query_vector: Sequence[float],
        model_id: str,
        limit: int,
    ) -> list[tuple[int, float]]:
        """Top-``limit`` (chunk_id, cosine score), best first."""
        ...


class SqliteVecBackend:
    """vec0 virtual table, cosine metric. Pattern: Strategy (primary).

    The table has no model_id column — vec0 stores one vector
    space, and the single-model invariant is enforced above this
    layer (embed_state records the model; the runner refuses to
    mix). ``model_id`` is accepted for interface parity and ignored.
    """

    name = "sqlite-vec"

    def __init__(self, *, dim: int) -> None:
        self.dim = dim

    def ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding float[{self.dim}] "
            "distance_metric=cosine)"
        )

    def put(
        self,
        connection: sqlite3.Connection,
        *,
        chunk_id: int,
        model_id: str,
        vector: Sequence[float],
    ) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"vector dim {len(vector)} != index dim {self.dim}")
        connection.execute("DELETE FROM embeddings WHERE chunk_id = ?", (chunk_id,))
        connection.execute(
            "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, _pack(vector)),
        )

    def delete(
        self, connection: sqlite3.Connection, *, chunk_ids: Collection[int]
    ) -> None:
        for chunk_id in chunk_ids:
            connection.execute(
                "DELETE FROM embeddings WHERE chunk_id = ?", (chunk_id,)
            )

    def clear(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM embeddings")

    def search(
        self,
        connection: sqlite3.Connection,
        *,
        query_vector: Sequence[float],
        model_id: str,
        limit: int,
    ) -> list[tuple[int, float]]:
        rows = connection.execute(
            "SELECT chunk_id, distance FROM embeddings "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_pack(query_vector), limit),
        ).fetchall()
        ranked = [(int(row[0]), 1.0 - float(row[1])) for row in rows]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked


class BlobBackend:
    """BLOB vectors + numpy brute-force cosine. Pattern: Strategy (fallback)."""

    name = "blob-numpy"

    def __init__(self, *, dim: int) -> None:
        self.dim = dim

    def ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "chunk_id INTEGER PRIMARY KEY, model_id TEXT NOT NULL, "
            "vector BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings (model_id)"
        )

    def put(
        self,
        connection: sqlite3.Connection,
        *,
        chunk_id: int,
        model_id: str,
        vector: Sequence[float],
    ) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"vector dim {len(vector)} != index dim {self.dim}")
        connection.execute(
            "INSERT OR REPLACE INTO embeddings (chunk_id, model_id, vector) "
            "VALUES (?, ?, ?)",
            (chunk_id, model_id, _pack(vector)),
        )

    def delete(
        self, connection: sqlite3.Connection, *, chunk_ids: Collection[int]
    ) -> None:
        for chunk_id in chunk_ids:
            connection.execute(
                "DELETE FROM embeddings WHERE chunk_id = ?", (chunk_id,)
            )

    def clear(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM embeddings")

    def search(
        self,
        connection: sqlite3.Connection,
        *,
        query_vector: Sequence[float],
        model_id: str,
        limit: int,
    ) -> list[tuple[int, float]]:
        rows = connection.execute(
            "SELECT chunk_id, vector FROM embeddings WHERE model_id = ?",
            (model_id,),
        ).fetchall()
        if not rows:
            return []
        matrix = np.array(
            [np.frombuffer(row[1], dtype="<f4") for row in rows], dtype=np.float64
        )
        query = np.asarray(query_vector, dtype=np.float64)
        denominators = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        denominators[denominators == 0.0] = 1.0
        scores = (matrix @ query) / denominators
        ranked = [(int(rows[i][0]), float(scores[i])) for i in range(len(rows))]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked[:limit]


def load_vec(connection: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into ``connection``."""
    import sqlite_vec

    connection.enable_load_extension(True)
    try:
        sqlite_vec.load(connection)
    finally:
        connection.enable_load_extension(False)


def vec_available(*, dim: int) -> bool:
    """Probe: can this machine load sqlite-vec and build a vec0 table?"""
    try:
        probe = sqlite3.connect(":memory:")
        load_vec(probe)
        probe.execute(
            f"CREATE VIRTUAL TABLE probe USING vec0(embedding float[{dim}])"
        )
        probe.close()
        return True
    except Exception:
        return False


def _row_to_chunk(row: sqlite3.Row) -> StoredChunk:
    return StoredChunk(
        id=row["id"],
        item_id=row["item_id"],
        ordinal=row["ordinal"],
        heading=row["heading"],
        text=row["text"],
    )


def _row_to_state(row: sqlite3.Row) -> EmbedState:
    return EmbedState(
        item_id=row["item_id"],
        source_hash=row["source_hash"],
        model_id=row["model_id"],
        chunk_count=row["chunk_count"],
        embedded_at=row["embedded_at"],
    )


class EmbedStore:
    """Repository for chunks / embed_state / vectors (via a backend)."""

    def __init__(
        self, *, connection_factory: ConnectionFactory, backend: VectorBackend
    ) -> None:
        self._connect = connection_factory
        self._backend = backend
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        migrate(connection)
        backend.ensure_schema(connection)
        connection.commit()

    @classmethod
    def open(cls, root: Path, *, dim: int) -> "EmbedStore":
        """Open the store at ``root``, probing for the vec backend."""
        root.mkdir(parents=True, exist_ok=True)
        db_path = root / "corpus.db"
        backend: VectorBackend = (
            SqliteVecBackend(dim=dim)
            if vec_available(dim=dim)
            else BlobBackend(dim=dim)
        )

        def factory() -> sqlite3.Connection:
            connection = sqlite3.connect(db_path)
            if isinstance(backend, SqliteVecBackend):
                load_vec(connection)
            return connection

        return cls(connection_factory=factory, backend=backend)

    @property
    def backend_name(self) -> str:
        """Which vector backend this store is actually running."""
        return self._backend.name

    # -- writes --------------------------------------------------------

    def replace_item_chunks(
        self,
        *,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        state: EmbedState,
    ) -> None:
        """Atomically replace one item's chunks, vectors, and state."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must pair up")
        if chunks and chunks[0].item_id != state.item_id:
            raise ValueError("chunks and state disagree on the item")
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        try:
            old_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM chunks WHERE item_id = ?", (state.item_id,)
                )
            ]
            self._backend.delete(connection, chunk_ids=old_ids)
            connection.execute(
                "DELETE FROM chunks WHERE item_id = ?", (state.item_id,)
            )
            for chunk, vector in zip(chunks, vectors):
                cursor = connection.execute(
                    "INSERT INTO chunks (item_id, ordinal, heading, text) "
                    "VALUES (?, ?, ?, ?)",
                    (chunk.item_id, chunk.ordinal, chunk.heading, chunk.text),
                )
                self._backend.put(
                    connection,
                    chunk_id=int(cursor.lastrowid),
                    model_id=state.model_id,
                    vector=vector,
                )
            self._upsert_state(connection, state)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def purge_item(self, *, item_id: str) -> None:
        """Drop one item's chunks, vectors, and embed_state."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        try:
            old_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM chunks WHERE item_id = ?", (item_id,)
                )
            ]
            self._backend.delete(connection, chunk_ids=old_ids)
            connection.execute("DELETE FROM chunks WHERE item_id = ?", (item_id,))
            connection.execute(
                "DELETE FROM embed_state WHERE item_id = ?", (item_id,)
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def clear_index(self) -> None:
        """Drop every chunk, vector, and state (the --rebuild op)."""
        connection = self._connect()
        try:
            self._backend.clear(connection)
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM embed_state")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    # -- reads ----------------------------------------------------------

    def get_embed_state(self, *, item_id: str) -> Lookup[EmbedState]:
        """One item's embed record (Lookup at the boundary)."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM embed_state WHERE item_id = ?", (item_id,)
        ).fetchone()
        if row is None:
            return Missing(reason=f"no embed state for item {item_id!r}")
        return Found(value=_row_to_state(row))

    def embed_states(self) -> list[EmbedState]:
        """Every embed_state row (model census + orphan detection)."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM embed_state ORDER BY item_id"
        ).fetchall()
        return [_row_to_state(row) for row in rows]

    def chunk_count_by_item(self) -> dict[str, int]:
        """Census: chunk counts keyed by item_id."""
        connection = self._connect()
        rows = connection.execute(
            "SELECT item_id, COUNT(*) FROM chunks GROUP BY item_id"
        ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def total_chunks(self) -> int:
        """The index's size in chunks."""
        connection = self._connect()
        row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0])

    def chunks_for_item(self, *, item_id: str) -> list[StoredChunk]:
        """One item's chunks in ordinal order."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM chunks WHERE item_id = ? ORDER BY ordinal", (item_id,)
        ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def search(
        self, *, query_vector: Sequence[float], model_id: str, limit: int
    ) -> list[SearchHit]:
        """Top-``limit`` chunks by cosine similarity to the query."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        ranked = self._backend.search(
            connection,
            query_vector=query_vector,
            model_id=model_id,
            limit=limit,
        )
        if not ranked:
            return []
        placeholders = ", ".join("?" for _ in ranked)
        rows = connection.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",
            tuple(chunk_id for chunk_id, _score in ranked),
        ).fetchall()
        by_id = {int(row["id"]): _row_to_chunk(row) for row in rows}
        return [
            SearchHit(chunk=by_id[chunk_id], score=score)
            for chunk_id, score in ranked
            if chunk_id in by_id
        ]

    # -- internals -------------------------------------------------------

    @staticmethod
    def _upsert_state(connection: sqlite3.Connection, state: EmbedState) -> None:
        connection.execute(
            """
            INSERT INTO embed_state (
                item_id, source_hash, model_id, chunk_count, embedded_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                source_hash = excluded.source_hash,
                model_id = excluded.model_id,
                chunk_count = excluded.chunk_count,
                embedded_at = excluded.embedded_at
            """,
            (
                state.item_id,
                state.source_hash,
                state.model_id,
                state.chunk_count,
                state.embedded_at,
            ),
        )
