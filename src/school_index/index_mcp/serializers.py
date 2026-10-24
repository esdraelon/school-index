"""Query records -> JSON-ready dicts for the MCP wire.

Pattern: pure serializer functions (no I/O, no state), mirroring
the adapter's. None stays None (JSON null) — a chunk with no page
anchor reports ``page: null``, never a disguised placeholder.
"""

from school_index.index_mcp.domain import Passage, TopicResource

JsonDict = dict[str, object]


def passage_to_dict(passage: Passage, *, display_title: str | None = None) -> JsonDict:
    """A Passage as a wire dict: text + score + full provenance.

    ``display_title`` (from ``index_mcp.presentation`` over the
    emitted window) replaces the raw title when the raw title is
    empty or duplicated in that window; default is the raw title,
    so single-record callers are unaffected. The wire keys never
    change — the fallback is a presentation of ``title``, not a
    new field (tutor contract v1).
    """
    return {
        "text": passage.text,
        "score": passage.score,
        "item_id": passage.item_id,
        "title": display_title if display_title is not None else passage.title,
        "course": passage.course,
        "deep_link": passage.deep_link,
        "page": passage.page,
        "source_kind": passage.source_kind,
    }


def topic_resource_to_dict(
    resource: TopicResource, *, display_title: str | None = None
) -> JsonDict:
    """A TopicResource as a wire dict (the tutor's pointer shape).

    ``display_title`` as above: the fallback presentation of the
    title for empty/duplicated titles, keys unchanged.
    """
    return {
        "item_id": resource.item_id,
        "title": display_title if display_title is not None else resource.title,
        "course": resource.course,
        "deep_link": resource.deep_link,
        "best_score": resource.best_score,
        "chunk_hits": resource.chunk_hits,
        "rank_score": resource.rank_score,
        "source_kind": resource.source_kind,
    }
