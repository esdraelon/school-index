"""Benchmark records store: benchmark.db, the benchmark's own file.

Pattern: Repository over an injected connection factory, mirroring
the pipeline stores — but deliberately independent of the corpus
schema: own tables, own file (BENCHMARK_ROOT), and nothing in here
ever opens corpus.db. The design's ``benchmark_runs`` made
concrete, plus the per-query raw outcomes table.

Rankings are stored as JSON: a ranking is an ordered list of
(item, scores) whose only consumers are the metric functions —
there is nothing to join into, and JSON keeps the record complete
(rank order + both scores survive), which is what "reproducible
from the records alone" requires.
"""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from school_index.benchmark.domain import (
    QueryOutcome,
    RankedItem,
    ResourceRank,
    RunRecord,
)
from school_index.corpus_sync.lookup import Found, Lookup, Missing
from school_index.corpus_sync.store import ConnectionFactory, connect_file

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id            TEXT PRIMARY KEY,
    set_version       TEXT NOT NULL,
    arm_key           TEXT NOT NULL,
    provider_model_id TEXT NOT NULL,
    chunking          TEXT NOT NULL,
    status            TEXT NOT NULL,
    blocked_reason    TEXT,
    started_at        TEXT NOT NULL,
    corpus_items      INTEGER NOT NULL,
    corpus_chunks     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_arm
    ON benchmark_runs (arm_key, started_at);

CREATE TABLE IF NOT EXISTS benchmark_query_results (
    run_id              TEXT NOT NULL,
    query_id            TEXT NOT NULL,
    expected_json       TEXT NOT NULL,
    resources_json      TEXT NOT NULL,
    passages_json       TEXT NOT NULL,
    top_chunk_item_id   TEXT,
    latency_resources_ms REAL NOT NULL,
    latency_passages_ms  REAL NOT NULL,
    PRIMARY KEY (run_id, query_id)
);
"""


def _run_from_row(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        set_version=row["set_version"],
        arm_key=row["arm_key"],
        provider_model_id=row["provider_model_id"],
        chunking=row["chunking"],
        status=row["status"],
        blocked_reason=row["blocked_reason"],
        started_at=row["started_at"],
        corpus_items=int(row["corpus_items"]),
        corpus_chunks=int(row["corpus_chunks"]),
    )


def _outcome_from_row(row: sqlite3.Row) -> QueryOutcome:
    resources = json.loads(row["resources_json"])
    passages = json.loads(row["passages_json"])
    return QueryOutcome(
        query_id=row["query_id"],
        expected_item_ids=tuple(json.loads(row["expected_json"])),
        resources=tuple(
            ResourceRank(
                item_id=str(entry["item_id"]),
                best_score=float(entry["best_score"]),
                rank_score=float(entry["rank_score"]),
            )
            for entry in resources
        ),
        passages=tuple(
            RankedItem(item_id=str(entry["item_id"]), score=float(entry["score"]))
            for entry in passages
        ),
        top_chunk_item_id=row["top_chunk_item_id"],
        latency_resources_ms=float(row["latency_resources_ms"]),
        latency_passages_ms=float(row["latency_passages_ms"]),
    )


class BenchmarkStore:
    """Repository for benchmark runs + their raw per-query outcomes."""

    def __init__(self, *, connection_factory: ConnectionFactory) -> None:
        self._connect = connection_factory
        connection = self._connect()
        connection.executescript(_SCHEMA)
        connection.commit()

    @classmethod
    def open(cls, root: Path) -> "BenchmarkStore":
        """Open (creating if needed) benchmark.db under ``root``."""
        root.mkdir(parents=True, exist_ok=True)
        db_path = root / "benchmark.db"

        def factory() -> sqlite3.Connection:
            return connect_file(db_path)

        return cls(connection_factory=factory)

    # -- writes ---------------------------------------------------------

    def record_run(
        self, run: RunRecord, outcomes: Sequence[QueryOutcome]
    ) -> None:
        """Store a run and its outcomes as one transaction."""
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO benchmark_runs (
                    run_id, set_version, arm_key, provider_model_id,
                    chunking, status, blocked_reason, started_at,
                    corpus_items, corpus_chunks
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.set_version,
                    run.arm_key,
                    run.provider_model_id,
                    run.chunking,
                    run.status,
                    run.blocked_reason,
                    run.started_at,
                    run.corpus_items,
                    run.corpus_chunks,
                ),
            )
            for outcome in outcomes:
                connection.execute(
                    """
                    INSERT INTO benchmark_query_results (
                        run_id, query_id, expected_json, resources_json,
                        passages_json, top_chunk_item_id,
                        latency_resources_ms, latency_passages_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.run_id,
                        outcome.query_id,
                        json.dumps(list(outcome.expected_item_ids)),
                        json.dumps(
                            [
                                {
                                    "item_id": r.item_id,
                                    "best_score": r.best_score,
                                    "rank_score": r.rank_score,
                                }
                                for r in outcome.resources
                            ]
                        ),
                        json.dumps(
                            [
                                {"item_id": r.item_id, "score": r.score}
                                for r in outcome.passages
                            ]
                        ),
                        outcome.top_chunk_item_id,
                        outcome.latency_resources_ms,
                        outcome.latency_passages_ms,
                    ),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    # -- reads ----------------------------------------------------------

    def runs(self) -> list[RunRecord]:
        """Every recorded run, oldest first."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM benchmark_runs ORDER BY started_at, run_id"
        ).fetchall()
        return [_run_from_row(row) for row in rows]

    def latest_run(self, *, arm_key: str) -> Lookup[RunRecord]:
        """An arm's most recent run (Lookup at the boundary)."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM benchmark_runs WHERE arm_key = ? "
            "ORDER BY started_at DESC, run_id DESC LIMIT 1",
            (arm_key,),
        ).fetchone()
        if row is None:
            return Missing(reason=f"no runs recorded for arm {arm_key!r}")
        return Found(value=_run_from_row(row))

    def outcomes(self, *, run_id: str) -> list[QueryOutcome]:
        """A run's raw outcomes, in query-id order."""
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM benchmark_query_results WHERE run_id = ? "
            "ORDER BY query_id",
            (run_id,),
        ).fetchall()
        return [_outcome_from_row(row) for row in rows]
