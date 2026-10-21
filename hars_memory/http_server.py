"""Native Streamable HTTP(S) MCP server for the long-term memory.

One process, one low-level ``Server`` instance, one loaded index. The stateless
``StreamableHTTPSessionManager`` creates a fresh transport per request but
reuses the same ``Server`` object — and therefore the module-global
``_rag_instance`` / project registry caches — so the corpus is loaded once per
*process* instead of once per MCP *session* (the supergateway ``--stateful``
model).

The endpoint is the MCP **resource server**. When
``HARS_MEMORY_MCP_OAUTH_ENABLED=1`` it also hosts the OAuth 2.1 authorization
server described in :mod:`hars_memory.mcp_oauth`: discovery documents,
dynamic client registration, authorization-code + PKCE, refresh tokens and
revocation. Without it the endpoint is unauthenticated — intended only for a
loopback or WireGuard-only bind (the previous supergateway deployment).

TLS is first-class: point ``HARS_MEMORY_MCP_TLS_CERTFILE`` and
``HARS_MEMORY_MCP_TLS_KEYFILE`` at a cert/key pair and the same endpoint
serves HTTPS; optionally require a client certificate with ``..._TLS_CA`` +
``..._TLS_REQUIRE_CLIENT_CERT=1``.

Environment (see docs/http-mcp.md for the full table):
    HARS_MEMORY_MCP_HTTP_HOST          bind host (default 127.0.0.1)
    HARS_MEMORY_MCP_HTTP_PORT          bind port (default 8099)
    HARS_MEMORY_MCP_HTTP_PATH          MCP path (default /mcp)
    HARS_MEMORY_MCP_HTTP_STATELESS     1 (default) = no MCP session state
    HARS_MEMORY_MCP_HTTP_JSON_RESPONSE 1 (default) = JSON instead of SSE replies
    HARS_MEMORY_MCP_HTTP_WARMUP        1 = load the index at startup, not on first call
    HARS_MEMORY_MCP_TLS_CERTFILE       PEM certificate chain -> enables HTTPS
    HARS_MEMORY_MCP_TLS_KEYFILE        PEM private key
    HARS_MEMORY_MCP_TLS_CA             PEM CA bundle (client-cert verification)
    HARS_MEMORY_MCP_TLS_REQUIRE_CLIENT_CERT  1/true -> require a client cert
    HARS_MEMORY_MCP_TLS_MIN_VERSION    TLSv1.2 (default) or TLSv1.3
    HARS_MEMORY_MCP_CORS_ORIGINS       space/comma list, "*" (default) or "none"
    HARS_MEMORY_MCP_OAUTH_*            see hars_memory/mcp_oauth.py
    HARS_MEMORY_TRACE_HTTP_*           the read-only trace SSE listener (sibling port)

Entry point: ``hars-longterm-memory-mcp-https``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
import time
from typing import Any

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from hars_memory import mcp_server
from hars_memory.mcp_oauth import (
    ALL_SCOPES,
    GRANT_SCOPE_KEY,
    OAuthProvider,
    OAuthSettings,
    build_oauth_app,
)
from hars_memory.trace import start_trace_server, trace_config_from_env

logger = logging.getLogger("memory.http")

HOST_ENV = "HARS_MEMORY_MCP_HTTP_HOST"
PORT_ENV = "HARS_MEMORY_MCP_HTTP_PORT"
PATH_ENV = "HARS_MEMORY_MCP_HTTP_PATH"
STATELESS_ENV = "HARS_MEMORY_MCP_HTTP_STATELESS"
JSON_RESPONSE_ENV = "HARS_MEMORY_MCP_HTTP_JSON_RESPONSE"
WARMUP_ENV = "HARS_MEMORY_MCP_HTTP_WARMUP"
WARMUP_PROJECT_ENV = "HARS_MEMORY_MCP_HTTP_WARMUP_PROJECT"
TLS_CERT_ENV = "HARS_MEMORY_MCP_TLS_CERTFILE"
TLS_KEY_ENV = "HARS_MEMORY_MCP_TLS_KEYFILE"
TLS_CA_ENV = "HARS_MEMORY_MCP_TLS_CA"
TLS_CLIENT_CERT_ENV = "HARS_MEMORY_MCP_TLS_REQUIRE_CLIENT_CERT"
TLS_MIN_VERSION_ENV = "HARS_MEMORY_MCP_TLS_MIN_VERSION"

STARTED_AT = time.monotonic()
WARM_STATE: dict[str, Any] = {"warm": False, "warmup_seconds": None, "error": None}


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bearer_token(headers: Headers) -> str:
    raw = headers.get("authorization", "").strip()
    if raw[:7].lower() == "bearer ":
        return raw[7:].strip()
    return ""


def _json(status: int, body: dict[str, Any], extra: dict[str, str] | None = None) -> Response:
    headers = {"Cache-Control": "no-store", **(extra or {})}
    return JSONResponse(body, status_code=status, headers=headers)


class MCPGateway:
    """Route ``/mcp`` to the session manager behind the bearer-token gate.

    Everything else goes to the Starlette sub-app (OAuth + ``/healthz``).
    The validated :class:`~hars_memory.mcp_oauth.Grant` is attached to the ASGI
    scope before the manager sees the request; ``mcp_server.call_tool`` reads it
    back from ``app.request_context.request.scope`` to enforce per-tool scopes.
    """

    def __init__(
        self, inner: Any, manager: StreamableHTTPSessionManager, provider: OAuthProvider, path: str
    ) -> None:
        self.inner = inner
        self.manager = manager
        self.provider = provider
        self.path = path

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("path") != self.path:
            await self.inner(scope, receive, send)
            return

        if not self.provider.enabled:
            await self.manager.handle_request(scope, receive, send)
            return

        request = Request(scope, receive)
        token = _bearer_token(request.headers)
        metadata_url = (
            f"{self.provider.issuer_for(self.provider.base_url(request))}"
            "/.well-known/oauth-protected-resource"
        )
        challenge = f'Bearer resource_metadata="{metadata_url}"'

        if not token:
            response = _json(
                401,
                {
                    "error": "unauthorized",
                    "error_description": "Missing Authorization: Bearer <access token>",
                    "resource_metadata": metadata_url,
                },
                {"WWW-Authenticate": challenge},
            )
            await response(scope, receive, send)
            return

        grant, error = self.provider.validate_access_token(token, self.provider.base_url(request))
        if grant is None:
            response = _json(
                401,
                {
                    "error": error or "invalid_token",
                    "error_description": "The access token is invalid, expired or revoked",
                    "resource_metadata": metadata_url,
                },
                {"WWW-Authenticate": f'{challenge}, error="{error or "invalid_token"}"'},
            )
            await response(scope, receive, send)
            return

        if not grant.scopes.intersection(ALL_SCOPES):
            response = _json(
                403,
                {
                    "error": "insufficient_scope",
                    "error_description": (
                        f"Token has no memory scope; granted: {sorted(grant.scopes)}"
                    ),
                },
                {
                    "WWW-Authenticate": (
                        f'{challenge}, error="insufficient_scope", scope="{" ".join(ALL_SCOPES)}"'
                    )
                },
            )
            await response(scope, receive, send)
            return

        scope[GRANT_SCOPE_KEY] = grant
        await self.manager.handle_request(scope, receive, send)


class AppWithLifespan:
    """Owns the session-manager, trace-listener and warm-up lifecycles."""

    def __init__(
        self,
        app: Any,
        manager: StreamableHTTPSessionManager,
        *,
        trace: tuple[str, int, float] | None = None,
        warmup_project: str | None = None,
    ) -> None:
        self.app = app
        self.manager = manager
        self.trace = trace
        self.warmup_project = warmup_project
        self._trace_server: asyncio.Server | None = None
        self._warmup_task: asyncio.Task[None] | None = None
        # Set by _serve() so warm-up waits for the listening socket instead of
        # competing with uvicorn's startup for the event loop.
        self.server: Any = None

    async def _start_trace(self) -> None:
        if self.trace is None:
            return
        try:
            self._trace_server = await start_trace_server(*self.trace)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Live trace SSE endpoint disabled (bind failed): %s", exc)

    async def _warmup(self) -> None:
        assert self.warmup_project is not None
        while self.server is not None and not getattr(self.server, "started", False):
            await asyncio.sleep(0.05)
        started = time.monotonic()
        try:
            await mcp_server._get_rag(self.warmup_project)
            WARM_STATE.update(
                warm=True, warmup_seconds=round(time.monotonic() - started, 1), error=None
            )
            logger.info(
                "Index warm-up complete in %.1fs (project=%s)",
                WARM_STATE["warmup_seconds"],
                self.warmup_project,
            )
        except Exception as exc:
            WARM_STATE["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Index warm-up failed (will retry on first request): %s", exc)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return

        message = await receive()
        if message["type"] != "lifespan.startup":
            return
        try:
            async with self.manager.run():
                await self._start_trace()
                if self.warmup_project:
                    # Do not block the bind: uvicorn has not started listening
                    # yet inside the lifespan handler.
                    self._warmup_task = asyncio.create_task(self._warmup())
                await send({"type": "lifespan.startup.complete"})
                while True:
                    incoming = await receive()
                    if incoming["type"] == "lifespan.shutdown":
                        if self._warmup_task is not None:
                            self._warmup_task.cancel()
                            await asyncio.gather(self._warmup_task, return_exceptions=True)
                        if self._trace_server is not None:
                            self._trace_server.close()
                            try:
                                await self._trace_server.wait_closed()
                            except Exception:  # pragma: no cover
                                pass
                        await send({"type": "lifespan.shutdown.complete"})
                        return
        except Exception as exc:
            await send({"type": "lifespan.startup.failed", "message": repr(exc)})
            raise


def _health_payload(provider: OAuthProvider) -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - STARTED_AT, 1),
        "index_shared": True,
        "index_warm": bool(WARM_STATE["warm"]),
        "warmup_seconds": WARM_STATE["warmup_seconds"],
        "oauth": provider.enabled,
    }


def create_app(
    provider: OAuthProvider,
    *,
    mcp_path: str = "/mcp",
    stateless: bool = True,
    json_response: bool = True,
    trace: tuple[str, int, float] | None = None,
    warmup_project: str | None = None,
) -> Any:
    """Build the full ASGI app (MCP + OAuth + health) for uvicorn or tests."""
    manager = StreamableHTTPSessionManager(
        app=mcp_server.app,
        stateless=stateless,
        json_response=json_response,
    )
    inner = build_oauth_app(provider, health=lambda: _health_payload(provider))
    gateway = MCPGateway(inner, manager, provider, mcp_path)
    cors = provider.settings.cors_origins
    if cors and cors != ("none",):
        app: Any = CORSMiddleware(
            gateway,
            allow_origins=list(cors),
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "Accept",
                "Mcp-Session-Id",
                "Mcp-Protocol-Version",
                "Last-Event-ID",
            ],
            expose_headers=["WWW-Authenticate", "Mcp-Session-Id"],
            max_age=600,
        )
    else:
        app = gateway
    return AppWithLifespan(app, manager, trace=trace, warmup_project=warmup_project)


# ---------------------------------------------------------------------------
# TLS
# ---------------------------------------------------------------------------


def _tls_files() -> tuple[str, str] | None:
    cert = os.environ.get(TLS_CERT_ENV, "").strip()
    key = os.environ.get(TLS_KEY_ENV, "").strip()
    if not cert or not key:
        return None
    return cert, key


def _min_tls_version() -> ssl.TLSVersion:
    raw = os.environ.get(TLS_MIN_VERSION_ENV, "").strip().lower()
    if raw in {"1.3", "tlsv1.3", "tls1.3"}:
        return ssl.TLSVersion.TLSv1_3
    return ssl.TLSVersion.TLSv1_2


def _ssl_context_factory(config: Any, default_factory: Any) -> ssl.SSLContext:
    context = default_factory()
    context.minimum_version = _min_tls_version()
    ca = os.environ.get(TLS_CA_ENV, "").strip()
    if ca:
        context.load_verify_locations(ca)
    if _flag(TLS_CLIENT_CERT_ENV):
        if not ca:
            raise RuntimeError(
                f"{TLS_CLIENT_CERT_ENV} requires {TLS_CA_ENV} to verify client certificates"
            )
        context.verify_mode = ssl.CERT_REQUIRED
    return context


def _uvicorn_config(app: Any, host: str, port: int) -> uvicorn.Config:
    kwargs: dict[str, Any] = {}
    tls = _tls_files()
    if tls is not None:
        kwargs["ssl_certfile"], kwargs["ssl_keyfile"] = tls
        kwargs["ssl_context_factory"] = _ssl_context_factory
    return uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        lifespan="on",
        access_log=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _serve() -> None:
    host = os.environ.get(HOST_ENV, "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get(PORT_ENV, "8099"))
    path = os.environ.get(PATH_ENV, "/mcp").strip() or "/mcp"
    stateless = _flag(STATELESS_ENV, default=True)
    json_response = _flag(JSON_RESPONSE_ENV, default=True)

    settings = OAuthSettings.from_env()
    provider = OAuthProvider(settings)

    warmup_project = os.environ.get(WARMUP_PROJECT_ENV, "default").strip() or "default"
    if not _flag(WARMUP_ENV):
        warmup_project = None

    trace_cfg = trace_config_from_env()
    app = create_app(
        provider,
        mcp_path=path,
        stateless=stateless,
        json_response=json_response,
        trace=trace_cfg,
        warmup_project=warmup_project,
    )

    scheme = "https" if _tls_files() else "http"
    issuer = settings.issuer or f"{scheme}://{host}:{port}"
    logger.info(
        "MCP %s endpoint: %s (index shared per process, stateless=%s, oauth=%s)",
        scheme,
        f"{scheme}://{host}:{port}{path}",
        stateless,
        provider.enabled,
    )
    if provider.enabled:
        logger.info("OAuth issuer=%s resource=%s", issuer, settings.resource or f"{issuer}/mcp")
        if scheme == "http" and host not in {"127.0.0.1", "localhost", "::1"}:
            logger.warning(
                "OAuth is enabled on a plain-HTTP listener. Bearer tokens and the "
                "authorization code are only as private as the network path; keep "
                "this bind on WireGuard/loopback or terminate TLS (%s / %s).",
                TLS_CERT_ENV,
                TLS_KEY_ENV,
            )

    config = _uvicorn_config(app, host, port)
    server = uvicorn.Server(config)
    app.server = server
    await server.serve()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
