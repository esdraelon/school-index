"""Unit tests: Passage.page derives from the chunker's heading contract."""

from school_index.index_mcp.domain import Passage


def _passage(heading: str | None) -> Passage:
    return Passage(
        chunk_id=1,
        item_id="item-1",
        text="text",
        heading=heading,
        score=0.5,
        title="Title",
        course="Algebra I",
        deep_link="https://example.com/x",
        source_kind="AGENDA",
    )


def test_page_parsed_from_page_heading() -> None:
    assert _passage("page 12").page == 12


def test_page_is_none_without_a_heading() -> None:
    assert _passage(None).page is None


def test_page_is_none_for_a_non_page_heading() -> None:
    assert _passage("introduction").page is None
