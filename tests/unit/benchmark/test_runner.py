"""Unit tests: the runner — raw outcomes in, records stored.

The service is a duck-typed double returning canned domain objects;
what's under test is the runner's part: rankings converted
faithfully, the expected set snapshotted, blocked runs recorded
with their reason and the arm's intended chunking.
"""

from pathlib import Path

from school_index.benchmark.arms import ARM_SPECS, BuiltArm
from school_index.benchmark.domain import QueryCase, QuerySet
from school_index.benchmark.runner import BenchmarkRunner
from school_index.benchmark.store import BenchmarkStore
from school_index.index_mcp.domain import Passage, TopicResource


class FakeService:
    """Canned SearchService double (same call shape, no store)."""

    def resources_for_topic(
        self, topic: str, *, course: str | None = None, limit: int = 5
    ) -> list[TopicResource]:
        return [
            TopicResource(
                item_id="item-b",
                title="B",
                course=None,
                deep_link="https://example.com/b",
                source_kind="AGENDA",
                best_score=0.6,
                chunk_hits=2,
                rank_score=0.62,
            ),
            TopicResource(
                item_id="item-a",
                title="A",
                course=None,
                deep_link="https://example.com/a",
                source_kind="AGENDA",
                best_score=0.55,
                chunk_hits=1,
                rank_score=0.55,
            ),
        ]

    def search_passages(
        self, query: str, *, course: str | None = None, limit: int = 5
    ) -> list[Passage]:
        def passage(chunk_id: int, item_id: str, score: float) -> Passage:
            return Passage(
                chunk_id=chunk_id,
                item_id=item_id,
                text="text",
                heading=None,
                score=score,
                title="t",
                course=None,
                deep_link="https://example.com/x",
                source_kind="AGENDA",
            )

        return [
            passage(1, "item-a", 0.9),
            passage(2, "item-b", 0.8),
            passage(3, "item-a", 0.7),
        ]


_QUERY_SET = QuerySet(
    set_version="1.0",
    queries=(
        QueryCase(
            id="q01-x",
            query="pythagorean?",
            course=None,
            expected_item_ids=("item-a",),
        ),
        QueryCase(id="q02-y", query="absent?", course=None, expected_item_ids=()),
    ),
)


def test_run_arm_records_raw_outcomes(tmp_path: Path) -> None:
    store = BenchmarkStore.open(tmp_path)
    runner = BenchmarkRunner(store=store)
    arm = BuiltArm(
        spec=ARM_SPECS["onnx-production"],
        service=FakeService(),  # type: ignore[arg-type]
        provider_model_id="fake-hash",
        chunking_label="production (target_chars=1000)",
    )
    record = runner.run_arm(
        arm=arm,
        query_set=_QUERY_SET,
        corpus_items=38,
        corpus_chunks=371,
        run_id="run-1",
        started_at="2026-10-23T15:00:00Z",
    )
    assert record.status == "completed"
    assert record.corpus_chunks == 371
    outcomes = store.outcomes(run_id="run-1")
    assert [o.query_id for o in outcomes] == ["q01-x", "q02-y"]
    first = outcomes[0]
    assert first.expected_item_ids == ("item-a",)
    assert [r.item_id for r in first.resources] == ["item-b", "item-a"]
    assert first.resources[0].rank_score == 0.62
    # Passages dedup to an item ranking by first occurrence.
    assert [(r.item_id, r.score) for r in first.passages] == [
        ("item-a", 0.9),
        ("item-b", 0.8),
    ]
    assert first.top_chunk_item_id == "item-a"
    assert first.latency_resources_ms >= 0.0
    assert first.latency_passages_ms >= 0.0


def test_record_blocked_keeps_reason_and_intended_chunking(tmp_path: Path) -> None:
    store = BenchmarkStore.open(tmp_path)
    runner = BenchmarkRunner(store=store)
    record = runner.record_blocked(
        spec=ARM_SPECS["bedrock-titan"],
        provider_model_id="bedrock-titan-embed-v2",
        query_set=_QUERY_SET,
        corpus_items=38,
        corpus_chunks=371,
        reason="NoCredentialsError: Unable to locate credentials",
        run_id="run-2",
        started_at="2026-10-23T15:01:00Z",
    )
    assert record.status == "blocked"
    assert record.blocked_reason is not None
    assert "NoCredentialsError" in record.blocked_reason
    assert record.chunking.startswith("production (")
    assert store.outcomes(run_id="run-2") == []
