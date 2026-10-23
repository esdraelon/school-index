"""Bearer-token auth for the MCP endpoint.

Pattern: ASGI middleware — our own code at the transport boundary,
not framework internals (mirrors the adapter's). The gateway
expects a shared bearer token; every HTTP request must carry
``Authorization: Bearer <token>`` before any MCP machinery (and
therefore any tool) is reached. The comparison is constant-time
(``hmac.compare_digest``) so the token can't be probed byte-by-byte
through response timing. Rejections are a bare 401 with a JSON
error body — no tool surface, no hints.

Non-HTTP scopes (lifespan, websocket) pass through untouched:
lifespan must run for the app to start at all, and it exposes
nothing.
"""

import hmac

from starlette.types import ASGIApp, Receive, Scope, Send

_UNAUTHORIZED_BODY = b'{"error": "unauthorized"}'


class BearerTokenMiddleware:
    """Reject HTTP requests without the expected bearer token."""

    def __init__(self, app: ASGIApp, *, expected_token: str) -> None:
        self._app = app
        self._expected = f"Bearer {expected_token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._authorized(scope):
            await self._app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})

    def _authorized(self, scope: Scope) -> bool:
        headers = {
            key.decode("latin-1").lower(): value
            for key, value in scope.get("headers", [])
        }
        provided = headers.get("authorization", b"")
        return hmac.compare_digest(provided, self._expected)
