"""The school adapter's MCP contract, consumed.

Pattern: Anti-Corruption Layer + Port. ``ResourceSource`` is the port
the harvester depends on. ``McpResourceSource`` translates the
adapter's ``get_resources`` wire shape into domain ``ResourceRef``
records exactly once, here at the boundary — nested student/course
objects, the ``{"result": [...]}`` envelope, blank titles, null
courses are all normalized in this module and nowhere else.

The SDK transport (``_sdk_call_tool``) is the only network-touching
code; the source takes it as an injected callable so unit tests can
substitute a fixture transport — the network edge is the only
substitution, and the translation logic is exercised for real. (The
default transport is also covered, by a round-trip test against an
in-process FastMCP server: see tests/unit/corpus_sync/test_client.py.)

One shape the live data taught us (first harvest, 2026-10-23): a
resource's ``student`` may be **null** — shared material (the team
site's, mostly) that the adapter reports inside every student's
result set. Those items are attributed to the *queried* student:
the request's scope is their scope.

A payload that breaks the contract raises ``PayloadError``: that is
a systemic failure (the harvest cannot know what the world looks
like), unlike a per-item download failure, which is data.
"""

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent

from school_index.corpus_sync.domain import ResourceRef

#: The transport seam: (tool name, arguments) -> decoded tool payload.
Transport = Callable[[str, Mapping[str, Any]], object]


class PayloadError(Exception):
    """The adapter's response does not match the get_resources contract."""


class ResourceSource(Protocol):
    """Where resources come from, as a port. Impl: McpResourceSource."""

    def list_resources(self, *, student: str) -> list[ResourceRef]:
        """All resources the adapter reports for ``student``."""
        ...


def _require_text(item: Mapping[str, Any], key: str, *, where: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PayloadError(f"{where}: required text field {key!r} is missing/blank")
    return value


def _optional_text(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    return value if isinstance(value, str) else ""


def _parse_resource(item: object, *, index: int, default_student: str) -> ResourceRef:
    where = f"resource[{index}]"
    if not isinstance(item, Mapping):
        raise PayloadError(f"{where}: expected an object, got {type(item).__name__}")
    student = item.get("student")
    if student is None:
        # Shared material (see module docstring): the query's student.
        student_id, student_name = default_student, ""
    elif isinstance(student, Mapping):
        student_id = _require_text(student, "id", where=f"{where}.student")
        student_name = _optional_text(student, "display_name")
    else:
        raise PayloadError(f"{where}: 'student' is neither an object nor null")
    course = item.get("course")
    course_name: str | None = None
    if isinstance(course, Mapping):
        name = course.get("name")
        course_name = name if isinstance(name, str) else None
    return ResourceRef(
        student_id=student_id,
        student_name=student_name,
        course_name=course_name,
        title=_optional_text(item, "title").strip(),
        kind=_optional_text(item, "kind"),
        url=_require_text(item, "url", where=where),
        source_kind=_optional_text(item, "source_kind"),
        scraped_at=_optional_text(item, "scraped_at"),
    )


def parse_resources(payload: object, *, default_student: str) -> list[ResourceRef]:
    """Translate a decoded get_resources payload into ResourceRefs.

    Accepts the probed envelope (``{"result": [...]}``) or a bare
    list — FastMCP produces the former for list-returning tools, and
    tolerating both costs nothing at the boundary. ``default_student``
    attributes resources whose ``student`` is null (shared material).
    """
    items: object = payload.get("result") if isinstance(payload, Mapping) else payload
    if not isinstance(items, list):
        raise PayloadError(
            f"get_resources payload has no resource list (got {type(items).__name__})"
        )
    return [
        _parse_resource(item, index=i, default_student=default_student)
        for i, item in enumerate(items)
    ]


def _httpx_client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """The SDK's client factory, pinned to ``trust_env=False``.

    Same reasoning as HttpDownloader's default client (see fetch.py):
    env proxy variables are runtime-specific and can make a default
    httpx client unconstructable; this bearer-token client talks to
    the estate gateway directly. (The SDK's own factory honors env
    proxies, which is the wrong default for a system-of-record tool.)
    """
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout or httpx.Timeout(30.0),
        auth=auth,
        follow_redirects=True,
        trust_env=False,
    )


async def _sdk_call_tool_async(
    *, url: str, token: str, tool: str, arguments: Mapping[str, Any]
) -> object:
    headers = {"Authorization": f"Bearer {token}"}
    async with streamablehttp_client(
        url, headers=headers, httpx_client_factory=_httpx_client_factory
    ) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool, dict(arguments))
    return _unwrap_result(result, tool=tool)


def _unwrap_result(result: CallToolResult, *, tool: str) -> object:
    """Reduce a CallToolResult to its decoded payload (pure; unit-tested).

    Structured content wins (the probed envelope shape); otherwise the
    first text block is parsed as JSON. An error result, or a result
    with no usable content, breaks the contract: PayloadError.
    """
    if result.isError:
        raise PayloadError(f"MCP tool {tool!r} returned an error result")
    if result.structuredContent is not None:
        return result.structuredContent
    for block in result.content:
        if isinstance(block, TextContent):
            return json.loads(block.text)
    raise PayloadError(f"MCP tool {tool!r} returned no usable content")


def _sdk_call_tool(
    *, url: str, token: str, tool: str, arguments: Mapping[str, Any]
) -> object:
    """Sync facade over the SDK's async client (the default transport)."""
    return asyncio.run(
        _sdk_call_tool_async(url=url, token=token, tool=tool, arguments=arguments)
    )


class McpResourceSource:
    """ResourceSource over the school adapter's MCP (streamable HTTP)."""

    def __init__(
        self, *, url: str, token: str, transport: Transport | None = None
    ) -> None:
        self._url = url
        self._token = token
        self._transport: Transport = transport or (
            lambda tool, args: _sdk_call_tool(
                url=url, token=token, tool=tool, arguments=args
            )
        )

    def list_resources(self, *, student: str) -> list[ResourceRef]:
        payload = self._transport("get_resources", {"student": student})
        return parse_resources(payload, default_student=student)
