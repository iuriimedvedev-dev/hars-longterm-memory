"""OpenAI-embeddings-shaped bridge onto a KnotGate agent-adapter's own
`EmbeddingProvider` abstraction, for use in place of `embedder.py`'s local
`sentence-transformers` model.

This is a deliberately SEPARATE module from `embedder.py`, not a modification
of it: the local-model path stays exactly as it was (still the default, still
what a standalone/non-adapter deployment uses), and this module is an
alternate implementation `lightrag_init.py` selects via
`HARS_MEMORY_EMBED_BACKEND=adapter`. Choosing this backend removes the local
ML runtime requirement entirely — no torch, no sentence-transformers, no
downloaded model weights — at the cost of needing a KnotGate agent-adapter
running in `local` mode nearby (see `adapter_mode_guard.py`; every other
adapter transport 503s on `/internal/embeddings/*` calls the same way it does
on `/internal/llm/*`, since both share the same loopback-only network guard,
though embeddings themselves need no chat-turn scoping at all — see that
route's own comment in the adapter's `app.ts`).

Required env vars when this backend is selected:
    HARS_MEMORY_EMBED_ADAPTER_URL
        Full URL of the adapter's embeddings bridge, e.g.
        http://127.0.0.1:4243/internal/embeddings/v1/embeddings
    HARS_MEMORY_EMBED_MODEL
        Passed through as the OpenAI-shaped request's `model` field. The
        adapter's own EmbeddingProviderFactory may ignore it (e.g. the
        `local`/Ollama-compatible provider uses its own EMBEDDING_MODEL
        config) — it exists for providers that route on it.
"""

from __future__ import annotations

import os

import httpx


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required when HARS_MEMORY_EMBED_BACKEND=adapter. "
            "Set it to the KnotGate agent-adapter's embeddings bridge URL, e.g. "
            "http://127.0.0.1:4243/internal/embeddings/v1/embeddings."
        )
    return value


def adapter_embedding_dimension(timeout_seconds: float = 30.0) -> int:
    """Probe the adapter's embeddings bridge once, synchronously, to learn its
    real output dimension. There is no separate metadata endpoint — the
    dimension is whatever the adapter's configured EmbeddingProvider actually
    produces, which this tool cannot know without asking it. Called once at
    LightRAG init time (create_lightrag() itself is synchronous), not on
    every embed call.
    """
    url = _require_env("HARS_MEMORY_EMBED_ADAPTER_URL")
    model = os.environ.get("HARS_MEMORY_EMBED_MODEL", "")

    response = httpx.post(
        url,
        json={"input": "dimension probe", "model": model},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    embedding = payload["data"][0]["embedding"]

    return len(embedding)


def make_adapter_embedding_func(batch_size: int = 32, timeout_seconds: float = 60.0) -> object:
    """Return an async embedding function compatible with LightRAG, backed by
    the adapter's embeddings bridge instead of a local model.

    Returns
    -------
    async callable:
        ``async (texts: list[str]) -> numpy.ndarray`` of shape (n_texts, dim), dtype float32.
    """
    url = _require_env("HARS_MEMORY_EMBED_ADAPTER_URL")
    model = os.environ.get("HARS_MEMORY_EMBED_MODEL", "")

    async def embed(texts: list[str], context: str | None = None, **_kwargs: object) -> "np.ndarray":  # noqa: F821
        import numpy as np

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            vectors: list[list[float]] = []

            # The adapter's LocalEmbeddingProvider (the only provider this v1
            # actually points 'local' mode at) embeds one text per upstream
            # call — batching here just means "send `batch_size` requests
            # concurrently", not "one request for many texts", so the
            # response shape (`data` ordered 1:1 with a single-item `input`)
            # stays simple and correct regardless of which provider is
            # actually configured behind the bridge.
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                response = await client.post(url, json={"input": chunk, "model": model})
                response.raise_for_status()
                payload = response.json()
                vectors.extend(item["embedding"] for item in payload["data"])

        return np.array(vectors, dtype=np.float32)

    return embed


__all__ = ["adapter_embedding_dimension", "make_adapter_embedding_func"]
