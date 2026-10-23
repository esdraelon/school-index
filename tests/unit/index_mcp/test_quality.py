"""Unit tests: match-quality calibration, over hand-built records.

The shapes mirror the M5 benchmark records the constants were
derived from: absent probes (high top, no standout / few hits),
genuine ties (two relevant items, no margin), and the boundary
values of each arm.
"""

from school_index.index_mcp.domain import Passage, TopicResource
from school_index.index_mcp.quality import assess_passages, assess_resources


def _passage(item_id: str, score: float, *, chunk_id: int = 0) -> Passage:
    return Passage(
        chunk_id=chunk_id,
        item_id=item_id,
        text="text",
        heading=None,
        score=score,
        title=f"Item {item_id}",
        course=None,
        deep_link=f"https://example.com/{item_id}",
        source_kind="AGENDA",
    )


def _resource(item_id: str, best_score: float, *, chunk_hits: int) -> TopicResource:
    return TopicResource(
        item_id=item_id,
        title=f"Item {item_id}",
        course=None,
        deep_link=f"https://example.com/{item_id}",
        source_kind="AGENDA",
        best_score=best_score,
        chunk_hits=chunk_hits,
        rank_score=best_score,
    )


# -- passages -------------------------------------------------------


def test_passages_empty_is_weak() -> None:
    assert assess_passages([]) == "weak"


def test_passages_clear_winner_is_strong() -> None:
    hits = [_passage("a", 0.80, chunk_id=1), _passage("b", 0.60, chunk_id=2)]
    assert assess_passages(hits) == "strong"


def test_passages_single_item_window_is_strong() -> None:
    hits = [_passage("a", 0.75, chunk_id=1), _passage("a", 0.70, chunk_id=2)]
    assert assess_passages(hits) == "strong"


def test_passages_absent_shape_is_weak() -> None:
    # The two-step-equations probe: top 0.701, runner-up item 0.697.
    hits = [_passage("a", 0.701, chunk_id=1), _passage("b", 0.697, chunk_id=2)]
    assert assess_passages(hits) == "weak"


def test_passages_score_gate_beats_a_huge_margin() -> None:
    hits = [_passage("a", 0.66, chunk_id=1), _passage("b", 0.40, chunk_id=2)]
    assert assess_passages(hits) == "weak"


def test_passages_tie_between_genuine_items_is_weak_by_design() -> None:
    # The tutorials shape: two right answers tie at 0.676/0.676 —
    # no standout and no decisive score. Documented under-claim.
    hits = [_passage("a", 0.676, chunk_id=1), _passage("b", 0.676, chunk_id=2)]
    assert assess_passages(hits) == "weak"


def test_passages_high_tie_is_strong_via_the_decisive_arm() -> None:
    # The slope-intercept shape: the KEY and its worksheet tie at
    # 0.810/0.802 (margin 0.008) — at that height the tie is
    # corroboration, not absence. Absent winners stay <= 0.701.
    hits = [_passage("a", 0.810, chunk_id=1), _passage("b", 0.802, chunk_id=2)]
    assert assess_passages(hits) == "strong"


def test_passages_just_below_decisive_with_no_margin_is_weak() -> None:
    hits = [_passage("a", 0.719, chunk_id=1), _passage("b", 0.711, chunk_id=2)]
    assert assess_passages(hits) == "weak"


def test_passages_boundary_values_are_strong() -> None:
    hits = [_passage("a", 0.675, chunk_id=1), _passage("b", 0.655, chunk_id=2)]
    assert assess_passages(hits) == "strong"


# -- resources ------------------------------------------------------


def test_resources_empty_is_weak() -> None:
    assert assess_resources([]) == "weak"


def test_resources_breadth_arm_is_strong() -> None:
    assert assess_resources([_resource("a", 0.681, chunk_hits=11)]) == "strong"


def test_resources_high_score_arm_is_strong() -> None:
    assert assess_resources([_resource("a", 0.748, chunk_hits=2)]) == "strong"


def test_resources_spurious_top_is_weak() -> None:
    # The surface-area probe's winner: 0.689, only 4 candidate hits.
    assert assess_resources([_resource("a", 0.689, chunk_hits=4)]) == "weak"


def test_resources_score_gate_beats_breadth() -> None:
    # The science-units shape: an 11-hit omnibus at 0.662 is still
    # not a pointer.
    assert assess_resources([_resource("a", 0.662, chunk_hits=11)]) == "weak"


def test_resources_moderate_score_without_breadth_is_weak() -> None:
    assert assess_resources([_resource("a", 0.70, chunk_hits=3)]) == "weak"
