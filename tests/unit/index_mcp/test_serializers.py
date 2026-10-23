"""Unit tests: the wire shapes (contract keys, null honesty)."""

from school_index.index_mcp.domain import Passage, TopicResource
from school_index.index_mcp.serializers import passage_to_dict, topic_resource_to_dict


def test_passage_shape() -> None:
    passage = Passage(
        chunk_id=7,
        item_id="item-1",
        text="Body text.",
        heading="page 3",
        score=0.91,
        title="Deck",
        course="Algebra I",
        deep_link="https://example.com/deck",
        source_kind="AGENDA",
    )
    assert passage_to_dict(passage) == {
        "text": "Body text.",
        "score": 0.91,
        "item_id": "item-1",
        "title": "Deck",
        "course": "Algebra I",
        "deep_link": "https://example.com/deck",
        "page": 3,
        "source_kind": "AGENDA",
    }


def test_passage_without_page_reports_null() -> None:
    passage = Passage(
        chunk_id=7,
        item_id="item-1",
        text="Body text.",
        heading=None,
        score=0.5,
        title="Deck",
        course=None,
        deep_link="https://example.com/deck",
        source_kind="AGENDA",
    )
    wire = passage_to_dict(passage)
    assert wire["page"] is None
    assert wire["course"] is None


def test_topic_resource_shape() -> None:
    resource = TopicResource(
        item_id="item-1",
        title="Deck",
        course="Algebra I",
        deep_link="https://example.com/deck",
        source_kind="AGENDA",
        best_score=1.0,
        chunk_hits=2,
        rank_score=1.02,
    )
    assert topic_resource_to_dict(resource) == {
        "item_id": "item-1",
        "title": "Deck",
        "course": "Algebra I",
        "deep_link": "https://example.com/deck",
        "best_score": 1.0,
        "chunk_hits": 2,
        "rank_score": 1.02,
        "source_kind": "AGENDA",
    }
