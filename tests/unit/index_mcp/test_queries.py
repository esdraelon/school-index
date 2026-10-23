"""Unit tests: SearchService ranking, over the stub geometry.

Geometry (query vector (1, 0), score = vector x): deck-a best 1.0
with 2 hits (rank 1.02); ws-c best 0.9 with 3 hits (rank 0.94);
sci-b best 0.0 with 1 hit (rank 0.0). A breadth-flip pair — dee
best 0.95/1 hit (0.95) vs eee best 0.94/4 hits (1.00) — sits in
ONE band (both 0.92–0.96), so breadth still decides it. The
band-dominance pair — narrow best 0.80/1 hit vs wide best 0.75/6
hits (rank 0.85) — spans bands 20 vs 18, so the better best chunk
wins outright (the M6 fix for the M5 colonial-trade miss).
"""

from pathlib import Path

import pytest

from school_index.index_mcp.queries import SearchService
from school_index.index_mcp.store import IndexReadStore
from tests.unit.index_mcp.fakes import (
    DECK_A,
    SCI_B,
    WS_C,
    SeedChunk,
    SeedItem,
    StubProvider,
    build_index,
)

DEE = SeedItem(
    url="https://example.com/dee",
    title="Dee Deck",
    course="Algebra I",
    chunks=(SeedChunk(text="Dee single.", heading=None, vector=(0.95, 0.3122)),),
)
EEE = SeedItem(
    url="https://example.com/eee",
    title="Eee Deck",
    course="Algebra I",
    chunks=(
        SeedChunk(text="Eee one.", heading=None, vector=(0.94, 0.3412)),
        SeedChunk(text="Eee two.", heading=None, vector=(0.93, 0.3676)),
        SeedChunk(text="Eee three.", heading=None, vector=(0.92, 0.3919)),
        SeedChunk(text="Eee four.", heading=None, vector=(0.91, 0.4146)),
    ),
)


def _service(tmp_path: Path, provider: StubProvider) -> SearchService:
    return SearchService(
        store=IndexReadStore.open(tmp_path, dim=StubProvider.dim), provider=provider
    )


def test_search_passages_ranks_and_trims(tmp_path: Path) -> None:
    provider = StubProvider()
    build_index(tmp_path, [DECK_A, SCI_B, WS_C])
    hits = _service(tmp_path, provider).search_passages("pythagorean", limit=2)
    assert [hit.text for hit in hits] == [
        "Model Pythagorean theorem: a^2 + b^2 = c^2 worked out.",
        "Triangles drill one.",
    ]
    assert provider.queries_seen == ["pythagorean"]
    assert provider.documents_seen == []


def test_search_passages_course_filter_oversamples(tmp_path: Path) -> None:
    provider = StubProvider()
    build_index(tmp_path, [DECK_A, SCI_B, WS_C])
    hits = _service(tmp_path, provider).search_passages(
        "anything", course="Science", limit=5
    )
    assert [hit.title for hit in hits] == ["Photosynthesis Notes"]


def test_resources_for_topic_aggregates_best_and_breadth(tmp_path: Path) -> None:
    provider = StubProvider()
    build_index(tmp_path, [DECK_A, SCI_B, WS_C])
    resources = _service(tmp_path, provider).resources_for_topic("triangles")
    by_title = {r.title: r for r in resources}
    assert resources[0].title == "Pythagorean Deck"
    assert by_title["Pythagorean Deck"].best_score == pytest.approx(1.0)
    assert by_title["Pythagorean Deck"].chunk_hits == 2
    assert by_title["Pythagorean Deck"].rank_score == pytest.approx(1.02)
    assert by_title["Algebra Worksheet"].chunk_hits == 3
    assert by_title["Algebra Worksheet"].rank_score == pytest.approx(0.94, abs=1e-3)
    assert by_title["Pythagorean Deck"].deep_link == "https://example.com/deck-a"


NARROW = SeedItem(
    url="https://example.com/narrow",
    title="Narrow Deck",
    course="Social Studies",
    chunks=(SeedChunk(text="Narrow single.", heading=None, vector=(0.80, 0.6)),),
)
WIDE = SeedItem(
    url="https://example.com/wide",
    title="Wide Omnibus",
    course="Social Studies",
    chunks=tuple(
        SeedChunk(text=f"Wide {n}.", heading=None, vector=(x, y))
        for n, (x, y) in enumerate(
            [
                (0.75, 0.6614),
                (0.74, 0.6726),
                (0.73, 0.6834),
                (0.72, 0.6939),
                (0.71, 0.7042),
                (0.70, 0.7141),
            ]
        )
    ),
)


def test_breadth_bonus_reorders_within_a_band(tmp_path: Path) -> None:
    provider = StubProvider()
    build_index(tmp_path, [DEE, EEE])
    resources = _service(tmp_path, provider).resources_for_topic("topic")
    assert [r.title for r in resources] == ["Eee Deck", "Dee Deck"]


def test_best_chunk_band_dominates_breadth(tmp_path: Path) -> None:
    provider = StubProvider()
    build_index(tmp_path, [NARROW, WIDE])
    resources = _service(tmp_path, provider).resources_for_topic("topic")
    # Wide's rank_score (0.85) beats Narrow's (0.80) on the old
    # aggregate; banding puts the 0.80 best two bands up, and no
    # breadth crosses a band boundary.
    assert [r.title for r in resources] == ["Narrow Deck", "Wide Omnibus"]
    assert resources[0].best_score == pytest.approx(0.80, abs=1e-3)
    assert resources[1].rank_score == pytest.approx(0.85, abs=1e-3)


EDGE = SeedItem(
    url="https://example.com/edge",
    title="Edge Deck",
    course="Math 8",
    chunks=(SeedChunk(text="Edge single.", heading=None, vector=(0.60, 0.8)),),
)
NEAR = SeedItem(
    url="https://example.com/near",
    title="Near Deck",
    course="Math 8",
    chunks=(
        SeedChunk(text="Near one.", heading=None, vector=(0.59, 0.8071)),
        SeedChunk(text="Near two.", heading=None, vector=(0.58, 0.8146)),
        SeedChunk(text="Near three.", heading=None, vector=(0.57, 0.8216)),
    ),
)


def test_band_edge_lands_in_the_upper_band(tmp_path: Path) -> None:
    provider = StubProvider()
    build_index(tmp_path, [EDGE, NEAR])
    # EDGE's best is exactly 0.60, a band edge at width 0.04 — but
    # 0.60/0.04 is 14.999... in float. The band epsilon must put it
    # in band 15, above NEAR (0.59, band 14), whose rank_score
    # (0.63) would otherwise win the shared band on breadth.
    resources = _service(tmp_path, provider).resources_for_topic("topic")
    assert [r.title for r in resources] == ["Edge Deck", "Near Deck"]


def test_worked_examples_formulates_the_query(tmp_path: Path) -> None:
    provider = StubProvider()
    build_index(tmp_path, [DECK_A])
    hits = _service(tmp_path, provider).worked_examples("pythagorean theorem")
    assert provider.queries_seen == ["worked example: pythagorean theorem"]
    assert hits[0].text.startswith("Model Pythagorean theorem")


def test_empty_query_and_bad_limit_are_rejected(tmp_path: Path) -> None:
    provider = StubProvider()
    build_index(tmp_path, [DECK_A])
    service = _service(tmp_path, provider)
    with pytest.raises(ValueError, match="must not be empty"):
        service.search_passages("   ")
    with pytest.raises(ValueError, match="limit"):
        service.search_passages("x", limit=0)
    with pytest.raises(ValueError, match="must not be empty"):
        service.resources_for_topic("")
    with pytest.raises(ValueError, match="must not be empty"):
        service.worked_examples(" ")


def test_limit_is_capped_not_rejected(tmp_path: Path) -> None:
    provider = StubProvider()
    build_index(tmp_path, [DECK_A, SCI_B])
    hits = _service(tmp_path, provider).search_passages("x", limit=999)
    assert len(hits) == 3  # every chunk there is
