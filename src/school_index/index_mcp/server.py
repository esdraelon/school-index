"""The index MCP server: three tools over the SearchService.

Pattern: thin Adapter (mirrors the school adapter's mcp_server) —
handlers do argument translation and serialization only; every
search decision lives in SearchService. Transport is the SDK's
Streamable HTTP app, stateless (each request self-contained — fits
the gateway's per-request model), wrapped in our bearer-token
middleware (see auth.py).

``build_server`` is the factory tests and the composition root
use: the service and the expected token are injected, nothing is
read from the environment.
"""

from collections.abc import Sequence

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp

from school_index.index_mcp.auth import BearerTokenMiddleware
from school_index.index_mcp.domain import Passage, TopicResource
from school_index.index_mcp.presentation import TitleContext, resolve_display_titles
from school_index.index_mcp.queries import SearchService
from school_index.index_mcp.quality import assess_passages, assess_resources
from school_index.index_mcp.serializers import (
    JsonDict,
    passage_to_dict,
    topic_resource_to_dict,
)


def _window_titles(results: Sequence[Passage | TopicResource]) -> dict[str, str]:
    """Display titles for one emitted window (presentation fallback).

    Every result type carries the same title fields, so one
    helper serves all three tools instead of three copies.
    """
    return resolve_display_titles(
        [
            TitleContext(
                item_id=result.item_id,
                title=result.title,
                course=result.course,
                first_heading=result.first_heading,
                first_text=result.first_text,
            )
            for result in results
        ]
    )


def build_mcp(service: SearchService) -> FastMCP:
    """The FastMCP app with exactly the three index tools registered."""
    mcp = FastMCP(
        "school-index",
        instructions=(
            "School Index: semantic search over Sam's harvested "
            "school resources — passages, topic pointers (resources "
            "with their original deep links), and worked-example "
            "material, all with provenance."
        ),
        stateless_http=True,
        # The endpoint lives at the app root, not the SDK's default
        # /mcp: at the default, POST /mcp/ answers 307 -> /mcp, and a
        # redirect is exactly what breaks behind the gateway. Nginx
        # strips the public /mcp/index/ prefix, so requests arrive
        # at / and are served directly — the sibling MCPs' shape.
        streamable_http_path="/",
        # DNS-rebinding protection with an explicit allowlist. Left
        # to its default, the SDK derives allowed hosts from the
        # bind address — localhost only — and 421s the gateway's
        # public Host (haliffer.ululuth.com, via $http_host). The
        # gateway host is the one public name this app is ever
        # served under; localhost entries keep on-host calls working.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                "haliffer.ululuth.com",
                "haliffer.ululuth.com:*",
                "127.0.0.1:*",
                "localhost:*",
                "[::1]:*",
            ],
            allowed_origins=[
                "https://haliffer.ululuth.com",
                "https://haliffer.ululuth.com:*",
                "http://127.0.0.1:*",
                "http://localhost:*",
            ],
        ),
    )

    @mcp.tool()
    def search_passages(
        query: str, course: str | None = None, limit: int = 5
    ) -> JsonDict:
        """Ranked passages matching ``query``, best first.

        Returns an envelope: ``match_quality`` ("strong" | "weak")
        calibrates the result set as a whole — "weak" means the
        corpus likely has no real material on this and the hits
        must not be presented as on-topic — and ``passages``, each
        carrying the passage text, its cosine score, and
        provenance: item_id, title, course, the original deep_link,
        and the page the passage starts on (null when the source
        has no page structure). ``course`` restricts to one course
        name as the school adapter reports it (e.g. "Algebra I").
        """
        hits = service.search_passages(query, course=course, limit=limit)
        titles = _window_titles(hits)
        return {
            "match_quality": assess_passages(hits),
            "passages": [
                passage_to_dict(passage, display_title=titles[passage.item_id])
                for passage in hits
            ],
        }

    @mcp.tool()
    def resources_for_topic(
        topic: str, course: str | None = None, limit: int = 5
    ) -> JsonDict:
        """Resources ranked as study pointers for ``topic``.

        Returns an envelope: ``match_quality`` ("strong" | "weak")
        for the set as a whole — on "weak" the honest answer is
        "no material on that", not the first resource listed — and
        ``resources``. Ranking: items are ordered by the band of
        their best chunk's score (0.04 of cosine), then by
        rank_score (best score plus a breadth bonus for additional
        matching chunks). Each result carries item_id, title,
        course, the original deep_link (cite this), best_score,
        chunk_hits, and the rank_score used.
        """
        resources = service.resources_for_topic(topic, course=course, limit=limit)
        titles = _window_titles(resources)
        return {
            "match_quality": assess_resources(resources),
            "resources": [
                topic_resource_to_dict(
                    resource, display_title=titles[resource.item_id]
                )
                for resource in resources
            ],
        }

    @mcp.tool()
    def worked_examples(topic: str, limit: int = 5) -> JsonDict:
        """Example-like passages for ``topic`` (v1).

        Same envelope shape as search_passages, including
        ``match_quality`` (the passages calibration, inherited —
        the reformulated query is not separately benchmarked).
        v1 retrieves passages with a worked-example query
        formulation; results are passages that read as worked
        material, not verified worked examples.
        """
        hits = service.worked_examples(topic, limit=limit)
        titles = _window_titles(hits)
        return {
            "match_quality": assess_passages(hits),
            "passages": [
                passage_to_dict(passage, display_title=titles[passage.item_id])
                for passage in hits
            ],
        }

    return mcp


def build_server(service: SearchService, *, expected_token: str) -> ASGIApp:
    """The deployable ASGI app: MCP behind bearer-token middleware."""
    if not expected_token:
        raise ValueError("expected_token must not be empty")
    return BearerTokenMiddleware(
        build_mcp(service).streamable_http_app(),
        expected_token=expected_token,
    )
