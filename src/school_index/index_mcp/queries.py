"""Query services: the ranking logic behind the three MCP tools.

Pattern: Service layer (mirrors the adapter's QueryServices) — the
MCP handlers translate arguments and serialize; every search and
ranking decision lives here, testable without transport.

Queries are embedded with the provider's ``embed_query`` — never
``embed_documents``: bge-style models are asymmetric, and encoding
a query as a passage silently degrades every result (the
protocol's split exists to make that mistake unrepresentable).

Course filtering is post-filter at the vector layer (KNN has no
WHERE clause), so passage searches oversample when a course is
given, then trim — at corpus scale the oversample is free.

``resources_for_topic`` ranking (the documented aggregate): items
are ordered FIRST by the band of their best chunk's score — bands
are ``_BAND_WIDTH`` (0.04) of cosine wide — and only inside a band
by ``rank_score`` = best score + a breadth bonus (+0.02 per
additional hit chunk, capped at +0.20). The band is the dominance
guarantee: no amount of breadth lifts an item over one whose best
chunk is a full band nearer, while near-ties (best scores within
one band) still reward the item that matches in several places.
M5's benchmark showed why: a 63-chunk omnibus at best 0.752 plus
the full +0.20 breadth bonus outranked the colonial-trade deck
whose best chunk (0.800) was the query's nearest passage — breadth
was compensating for a 0.048 relevance gap. Under banding that
deck wins outright (band 20 vs 18); re-running the full benchmark
set (M6) put resources hit@3 at 1.00 with the deck at rank 1.
Ties break on best score, then item id, so results are stable.
"""

import math

from school_index.embed_index.providers import EmbeddingProvider
from school_index.index_mcp.domain import Passage, TopicResource
from school_index.index_mcp.store import IndexReadStore

MAX_LIMIT = 50
_BAND_WIDTH = 0.04
_BAND_EPSILON = 1e-9
_BREADTH_WEIGHT = 0.02
_BREADTH_CAP = 10
_COURSE_OVERSAMPLE = 8
_TOPIC_FETCH_FLOOR = 100
_TOPIC_FETCH_PER_RESULT = 20
_TOPIC_FETCH_CAP = 500


def _check_query(text: str, *, what: str) -> str:
    query = text.strip()
    if not query:
        raise ValueError(f"{what} must not be empty")
    return query


def _check_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    return min(limit, MAX_LIMIT)


def _band(score: float) -> int:
    """The dominance band of a best-chunk score (see module docstring).

    The epsilon keeps exact band edges (0.80 at width 0.04) in the
    upper band despite binary float dust in the division.
    """
    return math.floor(score / _BAND_WIDTH + _BAND_EPSILON)


class SearchService:
    """The three tool operations over an injected store + provider."""

    def __init__(
        self, *, store: IndexReadStore, provider: EmbeddingProvider
    ) -> None:
        self._store = store
        self._provider = provider

    def search_passages(
        self, query: str, *, course: str | None = None, limit: int = 5
    ) -> list[Passage]:
        """Ranked passages for ``query``, optionally within a course."""
        limit = _check_limit(limit)
        text = _check_query(query, what="query")
        fetch = limit * _COURSE_OVERSAMPLE if course is not None else limit
        hits = self._store.search(
            query_vector=self._provider.embed_query(text),
            model_id=self._provider.model_id,
            course=course,
            fetch_limit=fetch,
        )
        return hits[:limit]

    def resources_for_topic(
        self, topic: str, *, course: str | None = None, limit: int = 5
    ) -> list[TopicResource]:
        """Items ranked as pointers for ``topic`` (see module docstring)."""
        limit = _check_limit(limit)
        text = _check_query(topic, what="topic")
        fetch = min(
            max(limit * _TOPIC_FETCH_PER_RESULT, _TOPIC_FETCH_FLOOR),
            _TOPIC_FETCH_CAP,
        )
        hits = self._store.search(
            query_vector=self._provider.embed_query(text),
            model_id=self._provider.model_id,
            course=course,
            fetch_limit=fetch,
        )
        best: dict[str, Passage] = {}
        counts: dict[str, int] = {}
        for hit in hits:
            counts[hit.item_id] = counts.get(hit.item_id, 0) + 1
            current = best.get(hit.item_id)
            if current is None or hit.score > current.score:
                best[hit.item_id] = hit
        resources = [
            TopicResource(
                item_id=item_id,
                title=hit.title,
                course=hit.course,
                deep_link=hit.deep_link,
                source_kind=hit.source_kind,
                best_score=hit.score,
                chunk_hits=counts[item_id],
                rank_score=hit.score
                + _BREADTH_WEIGHT * min(counts[item_id] - 1, _BREADTH_CAP),
                first_heading=hit.first_heading,
                first_text=hit.first_text,
            )
            for item_id, hit in best.items()
        ]
        resources.sort(
            key=lambda r: (
                -_band(r.best_score),
                -r.rank_score,
                -r.best_score,
                r.item_id,
            )
        )
        return resources[:limit]

    def worked_examples(self, topic: str, *, limit: int = 5) -> list[Passage]:
        """Example-like passages for ``topic`` (honest v1).

        v1 is a passage search whose query is formulated to pull
        worked material ("worked example: <topic>"); results are
        passages, not verified worked examples — no example
        detection exists yet. M6 refines the shape with the tutor.
        """
        text = _check_query(topic, what="topic")
        return self.search_passages(f"worked example: {text}", limit=limit)
