"""Title fallback presentation: empty + duplicated titles, live shapes.

Ground truth (live corpus 2026-10-24): one chunked empty title
and duplicated KEY x5 / Stations x2 / Drive file x2, all first-
chunk-headed "page 1" — so the tests pin course + heading +
first-text snippet, not heading alone, as the disambiguator.
"""

from school_index.index_mcp.domain import Passage, TopicResource
from school_index.index_mcp.server import _window_titles
from school_index.index_mcp.presentation import (
    TitleContext,
    fallback_title,
    resolve_display_titles,
    snippet,
)
from school_index.index_mcp.serializers import (
    passage_to_dict,
    topic_resource_to_dict,
)


def _ctx(**overrides: object) -> TitleContext:
    base: dict[str, object] = {
        "item_id": "aaaabbbbccccdddd",
        "title": "KEY",
        "course": "Algebra I",
        "first_heading": "page 1",
        "first_text": "Writing Equations in standard form practice problems.",
    }
    base.update(overrides)
    return TitleContext(**base)  # type: ignore[arg-type]


def test_snippet_collapses_whitespace_and_marks_cut() -> None:
    assert snippet("  a   b\nc  ") == "a b c"
    long = "word " * 30
    cut = snippet(long.strip())
    assert cut.endswith("…") and len(cut) <= 81
    assert snippet("short") == "short"
    # No word boundary inside the limit: hard cut, still marked.
    assert snippet("x" * 100) == "x" * 80 + "…"


def test_empty_title_falls_back_to_course_heading_snippet() -> None:
    title = fallback_title(_ctx(title=""))
    assert title == (
        "Algebra I · page 1 · Writing Equations in standard form "
        "practice problems."
    )
    assert title  # a fallback never emits ""


def test_duplicated_title_keeps_raw_title_as_prefix() -> None:
    title = fallback_title(_ctx())
    assert title.startswith("KEY — Algebra I · page 1 · Writing Equations")


def test_fallback_without_course_heading_or_snippet_still_names_course() -> None:
    title = fallback_title(
        _ctx(title="", course=None, first_heading=None, first_text="   ")
    )
    # The course segment is always present, so the fallback is
    # never empty even when heading and snippet are both absent.
    assert title == "No course"


def test_resolve_keeps_unique_titles_and_falls_back_on_duplicates() -> None:
    resolved = resolve_display_titles(
        [
            _ctx(item_id="id-1", title="KEY", first_text="First KEY text."),
            _ctx(item_id="id-2", title="KEY", first_text="Second KEY text."),
            _ctx(item_id="id-3", title="Pythagorean Deck", first_text="Other."),
            _ctx(item_id="id-4", title="", first_text="Blank worksheet."),
        ]
    )
    assert resolved["id-1"] != resolved["id-2"]
    assert "First KEY text." in resolved["id-1"]
    assert "Second KEY text." in resolved["id-2"]
    assert resolved["id-3"] == "Pythagorean Deck"
    assert resolved["id-4"].startswith("Algebra I · page 1 · Blank worksheet.")


def test_server_window_helper_resolves_a_duplicated_window() -> None:
    """The server's wiring (not just the pure functions) falls back."""
    first = TopicResource(
        item_id="id-1",
        title="KEY",
        course="Algebra I",
        deep_link="https://example.com/key-1",
        source_kind="AGENDA",
        best_score=0.8,
        chunk_hits=3,
        rank_score=0.84,
        first_heading="page 1",
        first_text="First KEY text.",
    )
    second = TopicResource(
        item_id="id-2",
        title="KEY",
        course="Algebra I",
        deep_link="https://example.com/key-2",
        source_kind="AGENDA",
        best_score=0.79,
        chunk_hits=2,
        rank_score=0.81,
        first_heading="page 1",
        first_text="Second KEY text.",
    )
    titles = _window_titles([first, second])
    assert titles["id-1"] != titles["id-2"]
    assert titles["id-1"].startswith("KEY — Algebra I · page 1 · First KEY text.")
    # Pointers are untouched by the fallback: deep links stay distinct.
    assert first.deep_link != second.deep_link


def test_serializers_use_display_title_without_changing_keys() -> None:
    passage = Passage(
        chunk_id=1,
        item_id="id-1",
        text="chunk text",
        heading="page 3",
        score=0.9,
        title="KEY",
        course="Algebra I",
        deep_link="https://example.com/key-1",
        source_kind="AGENDA",
        first_heading="page 1",
        first_text="First KEY text.",
    )
    wire = passage_to_dict(passage, display_title="KEY — Algebra I · page 1 · First KEY text.")
    assert wire["title"] == "KEY — Algebra I · page 1 · First KEY text."
    assert wire["deep_link"] == "https://example.com/key-1"
    assert set(wire) == {
        "text",
        "score",
        "item_id",
        "title",
        "course",
        "deep_link",
        "page",
        "source_kind",
    }
    resource = TopicResource(
        item_id="id-2",
        title="",
        course=None,
        deep_link="https://example.com/blank",
        source_kind="AGENDA",
        best_score=0.8,
        chunk_hits=3,
        rank_score=0.84,
        first_heading=None,
        first_text="Blank worksheet text.",
    )
    wire = topic_resource_to_dict(resource, display_title="No course · Blank worksheet text.")
    assert wire["title"] == "No course · Blank worksheet text."
    assert wire["deep_link"] == "https://example.com/blank"
