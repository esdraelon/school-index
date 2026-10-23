"""Unit tests: the benchmark records store (round-trips + lookups)."""

from pathlib import Path

from school_index.benchmark.domain import QueryOutcome, RankedItem, ResourceRank, RunRecord
from school_index.benchmark.store import BenchmarkStore
from school_index.corpus_sync.lookup import Found, Missing


def _run(run_id: str, *, arm: str = "arm-a", status: str = "completed") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        set_version="1.0",
        arm_key=arm,
        provider_model_id="model-x",
        chunking="production",
        status=status,
        blocked_reason="NoCredentialsError: no creds" if status == "blocked" else None,
        started_at="2026-10-23T15:00:00Z",
        corpus_items=38,
        corpus_chunks=371,
    )


def _outcome(query_id: str) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        expected_item_ids=("item-a",),
        resources=(
            ResourceRank(item_id="item-a", best_score=0.75, rank_score=0.77),
            ResourceRank(item_id="item-b", best_score=0.5, rank_score=0.5),
        ),
        passages=(RankedItem(item_id="item-b", score=0.81),),
        top_chunk_item_id="item-b",
        latency_resources_ms=12.5,
        latency_passages_ms=9.25,
    )


def test_record_and_read_back_round_trips(tmp_path: Path) -> None:
    store = BenchmarkStore.open(tmp_path)
    outcomes = [_outcome("q02"), _outcome("q01")]
    store.record_run(_run("run-1"), outcomes)
    assert store.runs() == [_run("run-1")]
    # Outcomes come back in query-id order, values intact.
    assert store.outcomes(run_id="run-1") == [outcomes[1], outcomes[0]]


def test_blocked_run_records_without_outcomes(tmp_path: Path) -> None:
    store = BenchmarkStore.open(tmp_path)
    blocked = _run("run-2", arm="bedrock-titan", status="blocked")
    store.record_run(blocked, [])
    latest = store.latest_run(arm_key="bedrock-titan")
    assert isinstance(latest, Found)
    assert latest.value.status == "blocked"
    assert latest.value.blocked_reason == "NoCredentialsError: no creds"
    assert store.outcomes(run_id="run-2") == []


def test_latest_run_picks_the_newest_and_misses_cleanly(tmp_path: Path) -> None:
    store = BenchmarkStore.open(tmp_path)
    older = _run("run-1")
    newer = RunRecord(
        run_id="run-3",
        set_version="1.0",
        arm_key="arm-a",
        provider_model_id="model-x",
        chunking="production",
        status="completed",
        blocked_reason=None,
        started_at="2026-10-23T16:00:00Z",
        corpus_items=38,
        corpus_chunks=371,
    )
    store.record_run(older, [])
    store.record_run(newer, [])
    latest = store.latest_run(arm_key="arm-a")
    assert isinstance(latest, Found)
    assert latest.value.run_id == "run-3"
    assert isinstance(store.latest_run(arm_key="never-ran"), Missing)
