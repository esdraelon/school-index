"""Contract fixtures: the frozen tutor-facing wire shapes (v1).

These tests pin the EXACT response shapes the future tutor package
codes against (docs/TUTOR-CONTRACT.md): envelope keys, per-item
keys, and value types for all three tools. A shape change must
fail here first — that failure is the contract's change alarm, and
it is answered by a contract version bump, not by editing the
fixture to match.
"""

import asyncio
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from school_index.index_mcp.queries import SearchService
from school_index.index_mcp.server import build_server
from school_index.index_mcp.store import IndexReadStore
from tests.unit.index_mcp.fakes import DECK_A, SCI_B, WS_C, StubProvider, build_index

TOKEN = "contract-token"

PASSAGE_KEYS = {
    "text",
    "score",
    "item_id",
    "title",
    "course",
    "deep_link",
    "page",
    "source_kind",
}
RESOURCE_KEYS = {
    "item_id",
    "title",
    "course",
    "deep_link",
    "best_score",
    "chunk_hits",
    "rank_score",
    "source_kind",
}


def _httpx_client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout or httpx.Timeout(30.0),
        auth=auth,
        follow_redirects=True,
        trust_env=False,
    )


async def _call(url: str, tool: str, arguments: dict[str, Any]) -> Any:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with streamablehttp_client(
        url, headers=headers, httpx_client_factory=_httpx_client_factory
    ) as (read_stream, write_stream, _session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
    assert not result.isError
    return result.structuredContent


@pytest.fixture
def mcp_url(tmp_path: Path) -> Iterator[str]:
    build_index(tmp_path, [DECK_A, SCI_B, WS_C])
    service = SearchService(
        store=IndexReadStore.open(tmp_path, dim=StubProvider.dim),
        provider=StubProvider(),
    )
    app = build_server(service, expected_token=TOKEN)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("index MCP server did not start")
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True
    thread.join(timeout=5)


def _assert_passage(item: dict[str, Any]) -> None:
    assert set(item) == PASSAGE_KEYS
    assert isinstance(item["text"], str)
    assert isinstance(item["score"], float)
    assert isinstance(item["item_id"], str)
    assert isinstance(item["title"], str)
    assert item["course"] is None or isinstance(item["course"], str)
    assert isinstance(item["deep_link"], str) and item["deep_link"]
    assert item["page"] is None or isinstance(item["page"], int)
    assert isinstance(item["source_kind"], str)


def _assert_passage_envelope(payload: dict[str, Any]) -> None:
    assert set(payload) == {"match_quality", "passages"}
    assert payload["match_quality"] in {"strong", "weak"}
    assert isinstance(payload["passages"], list)
    for item in payload["passages"]:
        _assert_passage(item)


def test_search_passages_shape_is_frozen(mcp_url: str) -> None:
    payload = asyncio.run(
        _call(mcp_url, "search_passages", {"query": "pythagorean", "limit": 5})
    )
    _assert_passage_envelope(payload)
    assert payload["passages"]  # the fixture query has real hits


def test_worked_examples_shape_is_frozen(mcp_url: str) -> None:
    payload = asyncio.run(
        _call(mcp_url, "worked_examples", {"topic": "pythagorean", "limit": 5})
    )
    _assert_passage_envelope(payload)


def test_resources_for_topic_shape_is_frozen(mcp_url: str) -> None:
    payload = asyncio.run(
        _call(mcp_url, "resources_for_topic", {"topic": "triangles", "limit": 5})
    )
    assert set(payload) == {"match_quality", "resources"}
    assert payload["match_quality"] in {"strong", "weak"}
    assert payload["resources"]
    for item in payload["resources"]:
        assert set(item) == RESOURCE_KEYS
        assert isinstance(item["item_id"], str)
        assert isinstance(item["title"], str)
        assert item["course"] is None or isinstance(item["course"], str)
        assert isinstance(item["deep_link"], str) and item["deep_link"]
        assert isinstance(item["best_score"], float)
        assert isinstance(item["chunk_hits"], int)
        assert isinstance(item["rank_score"], float)
        assert isinstance(item["source_kind"], str)
