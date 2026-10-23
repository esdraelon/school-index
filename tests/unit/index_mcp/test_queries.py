"""Unit tests: SearchService ranking, over the stub geometry.

Geometry (query vector (1, 0), score = vector x): deck-a best 1.0
with 2 hits (rank 1.02); ws-c best 0.9 with 3 hits (rank 0.94);
sci-b best 0.0 with 1 hit (rank 0.0). A breadth-flip pair — dee
best 0.95/1 hit (0.95) vs eee best 0.94/4 hits (1.00) — proves the
breadth bonus can reorder but only within its 0.2 cap.
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


def test_breadth_bonus_reorders_within_its_cap(tmp_path: Path) -> None:
    provider = StubProvider()
    build_index(tmp_path, [DEE, EEE])
    resources = _service(tmp_path, provider).resources_for_topic("topic")
    assert [r.title for r in resources] == ["Eee Deck", "Dee Deck"]


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
