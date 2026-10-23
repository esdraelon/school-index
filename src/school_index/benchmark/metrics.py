"""Pointer metrics over raw outcomes. Pattern: pure functions.

Nothing here touches a store, a provider, or a clock — outcomes in,
numbers out — which is what makes the benchmark's summaries
reproducible from records alone (the report path calls exactly
these functions over stored outcomes).

Definitions (rankings are the recorded top-10s):
- hit@k: an expected item appears within the first k ranks.
- MRR: mean reciprocal rank of the first expected item (0 when no
  expected item appears in the ranking).
- Separation: per query, the best expected item's score minus the
  best unexpected item's score, averaged over the queries where
  both exist — the calibration signal for tutor thresholds: a
  healthy arm keeps expected material clearly above the noise.
"""

import math
from collections.abc import Collection, Sequence

from school_index.benchmark.domain import ArmSummary, QueryOutcome, RankedItem
from school_index.index_mcp.domain import Passage

#: Outcomes are recorded at this fetch depth (the runner's limit).
RANKING_DEPTH = 10


def item_ranking(passages: Sequence[Passage]) -> tuple[RankedItem, ...]:
    """Dedup chunk-ranked passages to an item ranking.

    First occurrence wins: passages arrive score-ordered, so an
    item's first chunk is its best, and the item ranking inherits
    chunk order — the "which item does the best chunk point at"
    question the pointer metrics ask.
    """
    seen: dict[str, RankedItem] = {}
    for passage in passages:
        if passage.item_id not in seen:
            seen[passage.item_id] = RankedItem(
                item_id=passage.item_id, score=passage.score
            )
    return tuple(seen.values())


def _first_rank(item_ids: Sequence[str], expected: Collection[str]) -> int | None:
    wanted = set(expected)
    for index, item_id in enumerate(item_ids):
        if item_id in wanted:
            return index + 1
    return None


def hit_at_k(item_ids: Sequence[str], expected: Collection[str], *, k: int) -> bool:
    """True when an expected item appears within the first ``k``."""
    rank = _first_rank(item_ids, expected)
    return rank is not None and rank <= k


def reciprocal_rank(item_ids: Sequence[str], expected: Collection[str]) -> float:
    """1/rank of the first expected item; 0 when absent."""
    rank = _first_rank(item_ids, expected)
    return 1.0 / rank if rank is not None else 0.0


def _separation(
    pairs: Sequence[tuple[str, float]], expected: Collection[str]
) -> float | None:
    wanted = set(expected)
    expected_scores = [score for item_id, score in pairs if item_id in wanted]
    unexpected_scores = [score for item_id, score in pairs if item_id not in wanted]
    if not expected_scores or not unexpected_scores:
        return None
    return max(expected_scores) - max(unexpected_scores)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], *, pct: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def summarize(*, arm_key: str, outcomes: Sequence[QueryOutcome]) -> ArmSummary:
    """Aggregate one arm's raw outcomes into the summary metrics."""
    scored = [o for o in outcomes if o.expected_item_ids]
    empty_label = [o for o in outcomes if not o.expected_item_ids]

    def rate(flags: Sequence[bool]) -> float:
        return _mean([1.0 if flag else 0.0 for flag in flags])

    resource_ids = [[r.item_id for r in o.resources] for o in scored]
    passage_ids = [[r.item_id for r in o.passages] for o in scored]
    res_sep = [
        value
        for o in scored
        if (
            value := _separation(
                [(r.item_id, r.best_score) for r in o.resources],
                o.expected_item_ids,
            )
        )
        is not None
    ]
    pas_sep = [
        value
        for o in scored
        if (
            value := _separation(
                [(r.item_id, r.score) for r in o.passages], o.expected_item_ids
            )
        )
        is not None
    ]
    latencies = [o.latency_resources_ms + o.latency_passages_ms for o in outcomes]
    empty_tops = [
        o.resources[0].best_score for o in empty_label if o.resources
    ]
    return ArmSummary(
        arm_key=arm_key,
        queries_scored=len(scored),
        queries_empty_label=len(empty_label),
        resources_hit1=rate(
            [hit_at_k(ids, o.expected_item_ids, k=1) for ids, o in zip(resource_ids, scored)]
        ),
        resources_hit3=rate(
            [hit_at_k(ids, o.expected_item_ids, k=3) for ids, o in zip(resource_ids, scored)]
        ),
        resources_mrr=_mean(
            [reciprocal_rank(ids, o.expected_item_ids) for ids, o in zip(resource_ids, scored)]
        ),
        passages_hit1=rate(
            [hit_at_k(ids, o.expected_item_ids, k=1) for ids, o in zip(passage_ids, scored)]
        ),
        passages_hit3=rate(
            [hit_at_k(ids, o.expected_item_ids, k=3) for ids, o in zip(passage_ids, scored)]
        ),
        passages_mrr=_mean(
            [reciprocal_rank(ids, o.expected_item_ids) for ids, o in zip(passage_ids, scored)]
        ),
        top_chunk_hit1=rate(
            [o.top_chunk_item_id in set(o.expected_item_ids) for o in scored]
        ),
        separation_resources=_mean(res_sep) if res_sep else None,
        separation_passages=_mean(pas_sep) if pas_sep else None,
        separation_resources_n=len(res_sep),
        separation_passages_n=len(pas_sep),
        mean_latency_ms=_mean(latencies),
        p50_latency_ms=_percentile(latencies, pct=50) if latencies else 0.0,
        max_latency_ms=max(latencies) if latencies else 0.0,
        empty_label_mean_top_score=_mean(empty_tops) if empty_tops else None,
    )
