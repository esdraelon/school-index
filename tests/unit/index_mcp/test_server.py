"""Unit tests: the MCP app booted for real, answered via the SDK client.

Mirrors corpus_sync's round-trip fixture pattern: uvicorn on an
ephemeral port in a thread, the real streamable-HTTP client, a
fixture index underneath. This is the test that pins the deploy
shape — app-root serving, bearer middleware, host allowlist (the
client connects as 127.0.0.1) — before hal9k ever sees the code.
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
from tests.unit.index_mcp.fakes import DECK_A, SCI_B, StubProvider, build_index

TOKEN = "test-token"


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
    build_index(tmp_path, [DECK_A, SCI_B])
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


def test_search_passages_round_trip(mcp_url: str) -> None:
    payload = asyncio.run(_call(mcp_url, "search_passages", {"query": "pythagorean"}))
    results = payload["passages"]
    assert payload["match_quality"] == "strong"
    assert results[0]["text"].startswith("Model Pythagorean theorem")
    assert results[0]["deep_link"] == "https://example.com/deck-a"
    assert results[0]["page"] == 3
    assert results[0]["course"] == "Algebra I"


def test_resources_for_topic_round_trip(mcp_url: str) -> None:
    payload = asyncio.run(_call(mcp_url, "resources_for_topic", {"topic": "triangles"}))
    results = payload["resources"]
    assert payload["match_quality"] == "strong"
    assert results[0]["title"] == "Pythagorean Deck"
    assert results[0]["chunk_hits"] == 2


def test_worked_examples_round_trip(mcp_url: str) -> None:
    payload = asyncio.run(_call(mcp_url, "worked_examples", {"topic": "pythagorean"}))
    assert payload["passages"][0]["text"].startswith("Model Pythagorean theorem")
    assert payload["match_quality"] in {"strong", "weak"}


def test_missing_token_is_401(mcp_url: str) -> None:
    with httpx.Client(trust_env=False) as client:
        response = client.post(
            mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert response.status_code == 401


def test_empty_expected_token_is_rejected(tmp_path: Path) -> None:
    build_index(tmp_path, [])
    service = SearchService(
        store=IndexReadStore.open(tmp_path, dim=StubProvider.dim),
        provider=StubProvider(),
    )
    with pytest.raises(ValueError, match="must not be empty"):
        build_server(service, expected_token="")
