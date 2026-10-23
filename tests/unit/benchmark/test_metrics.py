"""Unit tests: pointer metrics — pure functions over raw outcomes.

The fixture's expected aggregates are computed by hand in the
comments; if a metric definition ever changes on purpose, the
change shows up here as a deliberate test edit, not a drift.
"""

from school_index.benchmark.domain import QueryOutcome, RankedItem, ResourceRank
from school_index.benchmark.metrics import (
    hit_at_k,
    item_ranking,
    reciprocal_rank,
    summarize,
)
from school_index.index_mcp.domain import Passage


def _passage(item_id: str, score: float, chunk_id: int = 1) -> Passage:
    return Passage(
        chunk_id=chunk_id,
        item_id=item_id,
        text="text",
        heading=None,
        score=score,
        title="title",
        course=None,
        deep_link="https://example.com/x",
        source_kind="AGENDA",
    )


def test_item_ranking_dedups_by_first_occurrence() -> None:
    ranking = item_ranking(
        [
            _passage("a", 0.9, chunk_id=1),
            _passage("b", 0.8, chunk_id=2),
            _passage("a", 0.7, chunk_id=3),
        ]
    )
    assert [(r.item_id, r.score) for r in ranking] == [("a", 0.9), ("b", 0.8)]


def test_hit_and_reciprocal_rank() -> None:
    ids = ["x", "a", "y"]
    assert hit_at_k(ids, ("a",), k=3)
    assert not hit_at_k(ids, ("a",), k=1)
    assert not hit_at_k(ids, ("absent",), k=3)
    assert reciprocal_rank(ids, ("a",)) == 0.5
    assert reciprocal_rank(ids, ("x", "a")) == 1.0
    assert reciprocal_rank(ids, ("absent",)) == 0.0


def _outcome(
    query_id: str,
    *,
    expected: tuple[str, ...],
    resources: tuple[ResourceRank, ...],
    passages: tuple[RankedItem, ...],
    top: str | None,
    lat: tuple[float, float] = (0.0, 0.0),
) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        expected_item_ids=expected,
        resources=resources,
        passages=passages,
        top_chunk_item_id=top,
        latency_resources_ms=lat[0],
        latency_passages_ms=lat[1],
    )


# o1: resources rr=1/2, passages rr=1; sep_res=-0.05, sep_pas=+0.20; top hit
# o2: resources rr=1, passages rr=1/2; sep_res=None, sep_pas=-0.10; top miss
# o3: empty label; top resources score 0.42; total latencies 30/10/2
_FIXTURE = [
    _outcome(
        "o1",
        expected=("a", "b"),
        resources=(
            ResourceRank(item_id="x", best_score=0.80, rank_score=0.84),
            ResourceRank(item_id="a", best_score=0.75, rank_score=0.77),
            ResourceRank(item_id="y", best_score=0.50, rank_score=0.50),
        ),
        passages=(RankedItem(item_id="a", score=0.9), RankedItem(item_id="x", score=0.7)),
        top="a",
        lat=(10.0, 20.0),
    ),
    _outcome(
        "o2",
        expected=("c",),
        resources=(ResourceRank(item_id="c", best_score=0.66, rank_score=0.66),),
        passages=(RankedItem(item_id="z", score=0.4), RankedItem(item_id="c", score=0.3)),
        top="z",
        lat=(5.0, 5.0),
    ),
    _outcome(
        "o3",
        expected=(),
        resources=(ResourceRank(item_id="x", best_score=0.42, rank_score=0.42),),
        passages=(RankedItem(item_id="x", score=0.42),),
        top="x",
        lat=(1.0, 1.0),
    ),
]


def test_summarize_matches_hand_computed_metrics() -> None:
    summary = summarize(arm_key="arm-x", outcomes=_FIXTURE)
    assert summary.queries_scored == 2
    assert summary.queries_empty_label == 1
    assert summary.resources_hit1 == 0.5
    assert summary.resources_hit3 == 1.0
    assert summary.resources_mrr == 0.75
    assert summary.passages_hit1 == 0.5
    assert summary.passages_hit3 == 1.0
    assert summary.passages_mrr == 0.75
    assert summary.top_chunk_hit1 == 0.5
    assert summary.separation_resources == -0.05
    assert summary.separation_resources_n == 1
    assert summary.separation_passages == 0.05
    assert summary.separation_passages_n == 2
    assert summary.mean_latency_ms == 14.0
    assert summary.p50_latency_ms == 10.0
    assert summary.max_latency_ms == 30.0
    assert summary.empty_label_mean_top_score == 0.42


def test_summarize_of_nothing_is_zeros_not_a_crash() -> None:
    summary = summarize(arm_key="arm-x", outcomes=[])
    assert summary.queries_scored == 0
    assert summary.resources_hit1 == 0.0
    assert summary.separation_resources is None
    assert summary.empty_label_mean_top_score is None
    assert summary.mean_latency_ms == 0.0
