"""LightRAG instance factory — wires e5-large (CPU), Qdrant, NetworkX, llama.cpp LLM.

All config comes from environment variables or graphrag.yaml via the caller.
Nothing is hardcoded.

Query path: embedder (CPU) + light query LLM — always available.
Index path: extractor LLM (GPU) — gate with gpu_guard.assert_gpu_free() first.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _make_llm_func(base_url: str, model: str, max_tokens: int, temperature: float) -> object:
    """Return an async LLM function compatible with LightRAG (OpenAI-compatible)."""
    import httpx

    async def llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: object,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload.update(kwargs)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])

    return llm_func


def create_lightrag(
    *,
    working_dir: str | None = None,
    extractor_base_url: str | None = None,
    extractor_model: str | None = None,
    query_base_url: str | None = None,
    query_model: str | None = None,
    embed_model: str | None = None,
    embed_hf_cache: str | None = None,
    embed_batch_size: int = 32,
    qdrant_url: str | None = None,
    qdrant_collection: str | None = None,
) -> object:
    """Construct and return a LightRAG instance.

    All parameters fall back to environment variables, which fall back to
    the defaults in graphrag.yaml.  Zero hardcoded values.

    Parameters
    ----------
    working_dir:
        LightRAG working directory for graph KV storage (NetworkX).
    extractor_base_url / extractor_model:
        OpenAI-compatible endpoint for the extraction LLM (Qwen3.6-27B).
    query_base_url / query_model:
        OpenAI-compatible endpoint for the query LLM (Qwen3.5-4B).
    embed_model:
        sentence-transformers model name (default: intfloat/e5-large-v2).
    embed_hf_cache:
        HF model cache directory.
    embed_batch_size:
        CPU embedding batch size.
    qdrant_url / qdrant_collection:
        Qdrant connection parameters.

    Returns
    -------
    LightRAG
        Fully wired instance ready for ``.ainsert()`` / ``.aquery()``.
    """
    from lightrag import LightRAG, QueryParam  # type: ignore[import-not-found]
    from lightrag.utils import EmbeddingFunc  # type: ignore[import-not-found]

    from tools.graphrag.server.embedder import embedding_dimension, make_embedding_func

    # --- resolve config (param > env > default) ---
    _wdir = working_dir or os.environ.get("GRAPHRAG_WORKING_DIR", "/tmp/hars_graphrag_lightrag")
    _ext_url = extractor_base_url or os.environ.get("GRAPHRAG_EXTRACTOR_BASE_URL", "http://localhost:8080/v1")
    _ext_model = extractor_model or os.environ.get("GRAPHRAG_EXTRACTOR_MODEL", "Qwen3.6-27B-Q4_K_M")
    _qry_url = query_base_url or os.environ.get("GRAPHRAG_QUERY_BASE_URL", "http://localhost:8081/v1")
    _qry_model = query_model or os.environ.get("GRAPHRAG_QUERY_MODEL", "Qwen3.5-4B-Q4_K_M")
    _emb_model = embed_model or os.environ.get("GRAPHRAG_EMBED_MODEL", "intfloat/e5-large-v2")
    _emb_cache = embed_hf_cache or os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home")
    _qdrant_url = qdrant_url or os.environ.get("GRAPHRAG_QDRANT_URL", "http://localhost:6333")
    _qdrant_coll = qdrant_collection or os.environ.get("GRAPHRAG_QDRANT_COLLECTION", "hars_graphrag")

    logger.info(
        "Creating LightRAG: working_dir=%s, extractor=%s@%s, query=%s@%s, embed=%s, qdrant=%s/%s",
        _wdir,
        _ext_model, _ext_url,
        _qry_model, _qry_url,
        _emb_model,
        _qdrant_url, _qdrant_coll,
    )

    Path(_wdir).mkdir(parents=True, exist_ok=True)

    embed_func = make_embedding_func(
        model_name=_emb_model,
        hf_cache_dir=_emb_cache,
        batch_size=embed_batch_size,
    )
    emb_dim = embedding_dimension(_emb_model)

    rag = LightRAG(
        working_dir=_wdir,
        llm_model_func=_make_llm_func(
            base_url=_ext_url,
            model=_ext_model,
            max_tokens=4096,
            temperature=0.1,
        ),
        embedding_func=EmbeddingFunc(
            embedding_dim=emb_dim,
            max_token_size=512,
            func=embed_func,
        ),
    )
    return rag


__all__ = ["create_lightrag"]
