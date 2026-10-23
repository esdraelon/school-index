"""System test: the index MCP serving a real pipeline-built index.

Everything is real — the embed runner builds the index from seeded
M1/M2 records with the deterministic hash provider, the ASGI app
boots under uvicorn, and the tools are called through the real
streamable-HTTP MCP client with the bearer token. Assertions about
read-only behavior read corpus.db with raw SQL, independently of
the repositories: serving must not move a single row.
"""

import asyncio
import hashlib
import math
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from school_index.corpus_sync.domain import CorpusItem, ItemStatus
from school_index.corpus_sync.store import CorpusStore
from school_index.doc_extract.domain import ExtractionOutcome, ExtractionState
from school_index.doc_extract.store import ExtractStore
from school_index.embed_index.providers import Vector
from school_index.embed_index.runner import EmbedRunner
from school_index.embed_index.store import EmbedStore
from school_index.index_mcp.queries import SearchService
from school_index.index_mcp.server import build_server
from school_index.index_mcp.store import IndexReadStore

TOKEN = "system-test-token"


class FakeProvider:
    model_id = "fake-hash"
    dim = 16

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [_vector(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return _vector(text)


def _vector(text: str) -> Vector:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [byte / 127.5 - 1.0 for byte in digest[:16]]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return tuple(x / norm for x in raw)


PYTH_TEXT = (
    "[[page 1]]\nThe Pythagorean theorem: in a right triangle, "
    "a^2 + b^2 = c^2. Model problem: legs 3 and 4 give hypotenuse 5."
)
PHOTO_TEXT = "[[page 1]]\nPhotosynthesis converts sunlight, water, and CO2 into sugars."


def _seed(root: Path, url: str, title: str, course: str, text: str) -> None:
    source_ref = f"sam:{url}"
    item = CorpusItem(
        id=hashlib.sha256(source_ref.encode("utf-8")).hexdigest(),
        source_kind="AGENDA",
        source_ref=source_ref,
        student="sam",
        course=course,
        title=title,
        deep_link=url,
        origin_url=url,
        format="pdf",
        content_hash=f"hash-{title}",
        fetched_at="2026-10-23T13:00:00+00:00",
        status=ItemStatus.OK,
        error=None,
    )
    CorpusStore.open(root).upsert_item(item)
    ExtractStore.open(root).record_outcome(
        ExtractionState(
            item_id=item.id,
            outcome=ExtractionOutcome.EXTRACTED,
            extractor="pypdf",
            source_hash=f"hash-{title}",
            format="pdf",
            char_count=len(text),
            page_count=1,
            error=None,
            extracted_at="2026-10-23T14:00:00+00:00",
        ),
        text=text,
    )


def _census(db_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(db_path)
    try:
        return {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in ("corpus_items", "chunks", "embed_state", "item_text")
        }
    finally:
        connection.close()


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
def served(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    root = tmp_path / "corpus"
    _seed(root, "https://example.com/pyth", "Pythagorean Deck", "Algebra I", PYTH_TEXT)
    _seed(root, "https://example.com/photo", "Photosynthesis", "Science", PHOTO_TEXT)
    provider = FakeProvider()
    runner = EmbedRunner(
        corpus_store=CorpusStore.open(root),
        extract_store=ExtractStore.open(root),
        embed_store=EmbedStore.open(root, dim=provider.dim),
        provider=provider,
    )
    report = runner.run()
    assert report.embedded == 2

    service = SearchService(
        store=IndexReadStore.open(root, dim=provider.dim), provider=provider
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
    yield f"http://127.0.0.1:{port}/", root
    server.should_exit = True
    thread.join(timeout=5)


def test_all_three_tools_answer(served: tuple[str, Path]) -> None:
    url, _root = served
    # The hash provider makes an item's own text its nearest query:
    # the query below is the seeded deck's distinctive content.
    passages = asyncio.run(
        _call(url, "search_passages", {"query": PYTH_TEXT.split("\n", 1)[1]})
    )
    assert passages["match_quality"] == "strong"
    assert passages["passages"][0]["title"] == "Pythagorean Deck"
    assert passages["passages"][0]["deep_link"] == "https://example.com/pyth"
    assert passages["passages"][0]["page"] == 1

    resources = asyncio.run(
        _call(url, "resources_for_topic", {"topic": PYTH_TEXT.split("\n", 1)[1]})
    )
    assert resources["match_quality"] == "strong"
    assert resources["resources"][0]["title"] == "Pythagorean Deck"
    assert resources["resources"][0]["chunk_hits"] >= 1

    examples = asyncio.run(
        _call(url, "worked_examples", {"topic": PYTH_TEXT.split("\n", 1)[1]})
    )
    assert examples["passages"]  # passages come back, labeled as passages
    assert examples["match_quality"] in {"strong", "weak"}


def test_serving_is_read_only(served: tuple[str, Path]) -> None:
    url, root = served
    before = _census(root / "corpus.db")
    asyncio.run(_call(url, "search_passages", {"query": "anything"}))
    asyncio.run(_call(url, "resources_for_topic", {"topic": "anything"}))
    assert _census(root / "corpus.db") == before


def test_no_token_no_service(served: tuple[str, Path]) -> None:
    url, _root = served
    with httpx.Client(trust_env=False) as client:
        response = client.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert response.status_code == 401
