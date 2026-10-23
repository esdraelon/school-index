"""Unit tests: the bearer middleware at the ASGI boundary."""

import asyncio

import httpx

from school_index.index_mcp.auth import BearerTokenMiddleware
from starlette.types import Receive, Scope, Send


async def _ok_app(scope: Scope, receive: Receive, send: Send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})


def _get(headers: dict[str, str]) -> httpx.Response:
    app = BearerTokenMiddleware(_ok_app, expected_token="s3cret")

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.get("/", headers=headers)

    return asyncio.run(run())


def test_missing_token_is_401() -> None:
    response = _get({})
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}


def test_wrong_token_is_401() -> None:
    response = _get({"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_correct_token_passes_through() -> None:
    response = _get({"Authorization": "Bearer s3cret"})
    assert response.status_code == 200
    assert response.text == "ok"
