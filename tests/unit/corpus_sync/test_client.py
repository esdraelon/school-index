"""Unit tests: the MCP client / anti-corruption layer.

Two seams are exercised: an injected fixture transport (the network
edge — everything downstream of it, the wire-shape translation, is
real code) and a genuine round trip: an in-process FastMCP server
serving the fixture payload, driven by McpResourceSource's default
SDK transport over localhost HTTP.
"""

import threading
import time
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP

from mcp.types import CallToolResult, ImageContent, TextContent

from school_index.corpus_sync.client import (
    McpResourceSource,
    PayloadError,
    _unwrap_result,
    parse_resources,
)

# Modeled on the dispatcher's 2026-10-23 live probe of the adapter:
# envelope {"result": [...]}, nested student/course, blank titles and
# null courses both occur in the real corpus.
FIXTURE_ITEMS: list[dict[str, Any]] = [
    {
        "student": {"id": "sam", "display_name": "Sam"},
        "course": {"name": "Algebra I", "period": None, "teacher": None},
        "title": "Pythagorean Theorem slides",
        "kind": "DOCUMENT",
        "url": "https://docs.google.com/presentation/d/fixtureDeck/pub",
        "source_kind": "AGENDA",
        "scraped_at": "2026-10-23T05:15:00+00:00",
    },
    {
        "student": {"id": "sam", "display_name": "Sam"},
        "course": None,
        "title": "   ",
        "kind": "TUTORIAL",
        "url": "https://drive.google.com/file/d/fixtureFile/view",
        "source_kind": "HAC",
        "scraped_at": "2026-10-23T05:15:00+00:00",
    },
    {
        "student": {"id": "sam", "display_name": "Sam"},
        "course": {"name": "Science", "period": "3", "teacher": "Ng"},
        "title": "IXL: cell structure",
        "kind": "TUTORIAL",
        "url": "https://www.ixl.com/science/grade-8/cells",
        "source_kind": "TEAM_SITE",
        "scraped_at": "2026-10-23T05:15:00+00:00",
    },
]


def test_parse_envelope_payload() -> None:
    resources = parse_resources({"result": FIXTURE_ITEMS})
    assert len(resources) == 3
    first = resources[0]
    assert first.student_id == "sam"
    assert first.course_name == "Algebra I"
    assert first.title == "Pythagorean Theorem slides"
    assert first.source_ref == f"sam:{first.url}"


def test_parse_bare_list_payload() -> None:
    assert len(parse_resources(FIXTURE_ITEMS)) == 3


def test_parse_blank_title_and_null_course() -> None:
    resources = parse_resources({"result": FIXTURE_ITEMS})
    second = resources[1]
    assert second.title == ""  # whitespace-only titles normalize to ""
    assert second.course_name is None


def test_parse_rejects_payload_without_a_list() -> None:
    with pytest.raises(PayloadError):
        parse_resources({"result": 42})
    with pytest.raises(PayloadError):
        parse_resources("not a payload")


def test_parse_rejects_non_object_item() -> None:
    with pytest.raises(PayloadError, match="expected an object"):
        parse_resources({"result": [42]})


def test_parse_rejects_non_object_student() -> None:
    bad = [{"student": "sam", "url": "https://example.com"}]
    with pytest.raises(PayloadError, match="student"):
        parse_resources(bad)


def test_parse_rejects_item_without_url() -> None:
    bad = [{"student": {"id": "sam"}, "title": "x"}]
    with pytest.raises(PayloadError, match="url"):
        parse_resources(bad)


def test_parse_rejects_item_without_student_id() -> None:
    bad = [{"student": {"display_name": "Sam"}, "url": "https://example.com"}]
    with pytest.raises(PayloadError, match="id"):
        parse_resources(bad)


def test_source_calls_get_resources_through_injected_transport() -> None:
    calls: list[tuple[str, Mapping[str, Any]]] = []

    def transport(tool: str, arguments: Mapping[str, Any]) -> object:
        calls.append((tool, arguments))
        return {"result": FIXTURE_ITEMS}

    source = McpResourceSource(url="http://unused", token="t", transport=transport)
    resources = source.list_resources(student="sam")
    assert calls == [("get_resources", {"student": "sam"})]
    assert len(resources) == 3


def test_unwrap_prefers_structured_content() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text='{"result": [1]}')],
        structuredContent={"result": [1, 2]},
    )
    assert _unwrap_result(result, tool="get_resources") == {"result": [1, 2]}


def test_unwrap_falls_back_to_text_json() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text='{"result": []}')]
    )
    assert _unwrap_result(result, tool="get_resources") == {"result": []}


def test_unwrap_rejects_error_results() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="boom")], isError=True
    )
    with pytest.raises(PayloadError, match="error result"):
        _unwrap_result(result, tool="get_resources")


def test_unwrap_rejects_content_without_text() -> None:
    image = ImageContent(type="image", data="aGk=", mimeType="image/png")
    result = CallToolResult(content=[image])
    with pytest.raises(PayloadError, match="no usable content"):
        _unwrap_result(result, tool="get_resources")


@pytest.fixture(scope="module")
def fixture_mcp_url() -> Iterator[str]:
    server = FastMCP("fixture-school")

    @server.tool()
    def get_resources(student: str) -> dict[str, Any]:
        return {"result": FIXTURE_ITEMS}

    config = uvicorn.Config(
        server.streamable_http_app(), host="127.0.0.1", port=0, log_level="error"
    )
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not uvicorn_server.started:
        if time.time() > deadline:
            raise RuntimeError("fixture MCP server did not start")
        time.sleep(0.05)
    port = uvicorn_server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/mcp"
    uvicorn_server.should_exit = True
    thread.join(timeout=5)


def test_source_round_trips_through_real_mcp(fixture_mcp_url: str) -> None:
    source = McpResourceSource(url=fixture_mcp_url, token="fixture-token")
    resources = source.list_resources(student="sam")
    assert [r.kind for r in resources] == ["DOCUMENT", "TUTORIAL", "TUTORIAL"]
    assert resources[2].course_name == "Science"
