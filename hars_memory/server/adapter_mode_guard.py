"""Refuses to run against a KnotGate agent-adapter that is not in `local` mode.

hars-longterm-memory's LLM and embedding calls, when pointed at a KnotGate
agent-adapter's own loopback bridges (`/internal/llm/*`,
`/internal/embeddings/*` — see that repo's `packages/agent-adapter/src/app.ts`),
only work against an adapter running in `local` mode: a plain, credential-free
proxy to the colocated Processor with no chat-turn context required. Every
other transport (`server`, the default) negotiates a turn-scoped credential
grant keyed to a real conversation's chatId — a standalone tool invocation
like `memory_consolidate` has no chat turn to scope that grant to, and the
adapter refuses with a 503 (`processor_llm_context_unavailable`).

Extending this to non-`local` modes needs a new background/tool-scoped
credential grant type on the adapter side — a deliberate future product
decision, not something this guard should route around or paper over.

This check is deliberately OPT-IN, not inferred from the base URL shape:
hars-longterm-memory predates KnotGate and remains usable standalone against
any OpenAI-compatible endpoint (a bare `llama-server`, for instance), which
has no `/health` endpoint and must never be forced through this probe. It
activates only when HARS_MEMORY_ADAPTER_HEALTH_URL is explicitly set —
i.e., only when this deployment is deliberately wired to a KnotGate adapter.
"""

from __future__ import annotations

import os
import sys

import httpx

ADAPTER_HEALTH_URL_ENV = "HARS_MEMORY_ADAPTER_HEALTH_URL"
REQUIRED_LLM_PROVIDER = "local"


def assert_local_mode_adapter_if_configured(timeout_seconds: float = 5.0) -> None:
    """Exit the process with a clear message if the configured adapter is not
    running in `local` mode. No-op entirely when
    HARS_MEMORY_ADAPTER_HEALTH_URL is unset (standalone/non-adapter use).
    """
    health_url = os.environ.get(ADAPTER_HEALTH_URL_ENV, "").strip()

    if not health_url:
        return

    try:
        response = httpx.get(health_url, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - any failure here means "cannot verify", not "is local"
        print(
            f"ERROR: {ADAPTER_HEALTH_URL_ENV} is set to '{health_url}' but it could not be "
            f"reached or did not return a valid health payload ({exc}). Refusing to start: "
            "hars-longterm-memory v1 only supports adapters confirmed to be running in "
            "'local' mode, and this cannot be confirmed right now.",
            file=sys.stderr,
        )
        sys.exit(1)

    llm_provider = payload.get("llmProvider") if isinstance(payload, dict) else None

    if llm_provider != REQUIRED_LLM_PROVIDER:
        print(
            f"ERROR: the adapter at '{health_url}' reports llmProvider="
            f"'{llm_provider}', not '{REQUIRED_LLM_PROVIDER}'. hars-longterm-memory v1 "
            "only supports 'local'-mode adapters — every other transport negotiates a "
            "turn-scoped credential grant this tool has no chat turn to scope to, and "
            "its LLM/embedding calls would 503. Switch this environment's LLM_PROVIDER "
            "to 'local', or do not enable this MCP server on this environment.",
            file=sys.stderr,
        )
        sys.exit(1)


__all__ = ["assert_local_mode_adapter_if_configured", "ADAPTER_HEALTH_URL_ENV"]
