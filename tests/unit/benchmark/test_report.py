"""Unit tests: the report — every number re-derived from records."""

from dataclasses import replace
from pathlib import Path

from school_index.benchmark.domain import QueryOutcome, RankedItem, ResourceRank
from school_index.benchmark.report import render_report
from school_index.benchmark.store import BenchmarkStore
from tests.unit.benchmark.test_store import _outcome, _run


def test_empty_store_reports_emptiness(tmp_path: Path) -> None:
    assert render_report(BenchmarkStore.open(tmp_path)) == (
        "No benchmark runs recorded.\n"
    )


def test_report_renders_latest_runs_misses_and_blocks(tmp_path: Path) -> None:
    store = BenchmarkStore.open(tmp_path)
    # A completed run whose only outcome is a resources top-3 miss.
    miss = QueryOutcome(
        query_id="q99-miss",
        expected_item_ids=("expected-item-0001",),
        resources=(
            ResourceRank(item_id="other-item-0002", best_score=0.8, rank_score=0.8),
            ResourceRank(item_id="other-item-0003", best_score=0.7, rank_score=0.7),
            ResourceRank(item_id="other-item-0004", best_score=0.6, rank_score=0.6),
            ResourceRank(item_id="expected-item-0001", best_score=0.5, rank_score=0.5),
        ),
        passages=(RankedItem(item_id="expected-item-0001", score=0.9),),
        top_chunk_item_id="expected-item-0001",
        latency_resources_ms=10.0,
        latency_passages_ms=10.0,
    )
    store.record_run(_run("run-old", arm="onnx-production"), [_outcome("q01")])
    newer = replace(
        _run("run-new", arm="onnx-production"), started_at="2026-10-23T16:00:00Z"
    )
    store.record_run(newer, [miss])
    store.record_run(_run("run-bed", arm="bedrock-titan", status="blocked"), [])

    report = render_report(store)
    # Only the LATEST production run is summarized (run-new, the miss).
    assert "run-new" in report
    assert "run-old" not in report
    assert "| onnx-production | completed |" in report
    assert "| bedrock-titan | BLOCKED |" in report
    assert "NoCredentialsError: no creds" in report
    # The miss is listed with short ids on both sides.
    assert "q99-miss" in report
    assert "expected" in report and "other-it" in report
    # Passages found it at rank 1 even though resources missed top-3:
    # resources 0.00/0.00/0.250, passages 1.00/1.00/1.000, top-chunk 1.00.
    assert "| 0.00 | 0.00 | 0.250 | 1.00 | 1.00 | 1.000 | 1.00 |" in report


def test_report_notes_a_clean_run(tmp_path: Path) -> None:
    store = BenchmarkStore.open(tmp_path)
    store.record_run(_run("run-1", arm="onnx-production"), [_outcome("q01")])
    report = render_report(store)
    assert "No resources hit@3 misses." in report
