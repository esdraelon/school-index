"""Benchmark runner: execute an arm over the set, record evidence.

Pattern: Service over injected collaborators (the records store;
the arm arrives built). Per query it calls the production search
service exactly as a consumer would — resources_for_topic and
search_passages at the recorded depth — timing each call with a
monotonic clock, and stores the RAW rankings. No metric is
computed here: numbers are derived from records by
benchmark.metrics, here and in the report alike.
"""

import time

from school_index.benchmark.arms import ArmSpec, BuiltArm, chunking_label
from school_index.benchmark.domain import (
    QueryCase,
    QueryOutcome,
    QuerySet,
    ResourceRank,
    RunRecord,
)
from school_index.benchmark.metrics import RANKING_DEPTH, item_ranking
from school_index.benchmark.store import BenchmarkStore
from school_index.index_mcp.queries import SearchService


class BenchmarkRunner:
    """Runs arms and records runs (completed or blocked)."""

    def __init__(self, *, store: BenchmarkStore) -> None:
        self._store = store

    def run_arm(
        self,
        *,
        arm: BuiltArm,
        query_set: QuerySet,
        corpus_items: int,
        corpus_chunks: int,
        run_id: str,
        started_at: str,
    ) -> RunRecord:
        """Execute every query against the arm; record the run."""
        outcomes = [
            self._run_query(arm.service, case) for case in query_set.queries
        ]
        record = RunRecord(
            run_id=run_id,
            set_version=query_set.set_version,
            arm_key=arm.spec.key,
            provider_model_id=arm.provider_model_id,
            chunking=arm.chunking_label,
            status="completed",
            blocked_reason=None,
            started_at=started_at,
            corpus_items=corpus_items,
            corpus_chunks=corpus_chunks,
        )
        self._store.record_run(record, outcomes)
        return record

    def record_blocked(
        self,
        *,
        spec: ArmSpec,
        provider_model_id: str,
        query_set: QuerySet,
        corpus_items: int,
        corpus_chunks: int,
        reason: str,
        run_id: str,
        started_at: str,
    ) -> RunRecord:
        """Record an arm that could not run, with its reason verbatim."""
        record = RunRecord(
            run_id=run_id,
            set_version=query_set.set_version,
            arm_key=spec.key,
            provider_model_id=provider_model_id,
            chunking=chunking_label(spec),
            status="blocked",
            blocked_reason=reason,
            started_at=started_at,
            corpus_items=corpus_items,
            corpus_chunks=corpus_chunks,
        )
        self._store.record_run(record, [])
        return record

    # -- internals -------------------------------------------------------

    @staticmethod
    def _run_query(service: SearchService, case: QueryCase) -> QueryOutcome:
        start = time.perf_counter()
        resources = service.resources_for_topic(
            case.query, course=case.course, limit=RANKING_DEPTH
        )
        mid = time.perf_counter()
        passages = service.search_passages(
            case.query, course=case.course, limit=RANKING_DEPTH
        )
        end = time.perf_counter()
        return QueryOutcome(
            query_id=case.id,
            expected_item_ids=case.expected_item_ids,
            resources=tuple(
                ResourceRank(
                    item_id=hit.item_id,
                    best_score=hit.best_score,
                    rank_score=hit.rank_score,
                )
                for hit in resources
            ),
            passages=item_ranking(passages),
            top_chunk_item_id=passages[0].item_id if passages else None,
            latency_resources_ms=(mid - start) * 1000.0,
            latency_passages_ms=(end - mid) * 1000.0,
        )
