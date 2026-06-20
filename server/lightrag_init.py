"""LightRAG instance factory — wires CPU embeddings, file-backed graph/vector storage, and llama.cpp LLMs.

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


class _ByteTokenizer:
    """Small offline tokenizer compatible with LightRAG's Tokenizer wrapper.

    It counts UTF-8 bytes instead of model tokens. That is conservative enough
    for local smoke/indexing and avoids tiktoken's runtime asset download.
    """

    def encode(self, content: str) -> list[int]:
        return list((content or "").encode("utf-8", errors="ignore"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(max(0, min(255, int(token))) for token in tokens).decode(
            "utf-8",
            errors="ignore",
        )


def make_llm_func(base_url: str, model: str, max_tokens: int, temperature: float) -> object:
    """Return an async LLM function compatible with LightRAG (OpenAI-compatible)."""
    import httpx

    endpoint = base_url.rstrip("/")
    # Default 300 s: a 27B model at 8192-ctx context under 2-slot contention can
    # legitimately take >120 s.  Set GRAPHRAG_LLM_TIMEOUT_SECONDS to override.
    # NOTE: LightRAG's internal worker timeout is separate and not exposed by
    # the library's public API.  If chunks are still timing out under very heavy
    # load, lower GRAPHRAG_MAX_PARALLEL_INSERT (default 2) to reduce contention.
    timeout_seconds = float(os.environ.get("GRAPHRAG_LLM_TIMEOUT_SECONDS", "300"))

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

        _JSON_PRIMITIVES = (str, int, float, bool, type(None))
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # LightRAG may pass internal storage objects via **kwargs; only forward
        # JSON-serializable primitives to avoid TypeError in httpx serialization.
        payload.update({k: v for k, v in kwargs.items() if isinstance(v, _JSON_PRIMITIVES)})

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                resp = await client.post(
                    f"{endpoint}/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return str(data["choices"][0]["message"]["content"])
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"LLM endpoint unavailable for model '{model}' at {endpoint}: {exc}"
            ) from exc

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
    vector_storage: str | None = None,
    graph_storage: str | None = None,
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
    vector_storage / graph_storage:
        LightRAG storage backend names. The current PoC default is
        NanoVectorDBStorage + NetworkXStorage.

    Returns
    -------
    LightRAG
        Fully wired instance ready for ``.ainsert()`` / ``.aquery()``.
    """
    from lightrag import LightRAG  # type: ignore[import-not-found]
    from lightrag.utils import EmbeddingFunc, Tokenizer  # type: ignore[import-not-found]

    from tools.graphrag.server.embedder import embedding_dimension, make_embedding_func
    from tools.graphrag.schema.entity_types import EntityType

    # --- resolve config (param > env > default) ---
    _wdir = working_dir or os.environ.get("GRAPHRAG_WORKING_DIR", "/tmp/hars_graphrag_lightrag")
    _ext_url = extractor_base_url or os.environ.get("GRAPHRAG_EXTRACTOR_BASE_URL", "http://localhost:8080/v1")
    _ext_model = extractor_model or os.environ.get("GRAPHRAG_EXTRACTOR_MODEL", "Qwen3.6-27B-Q4_K_M")
    _qry_url = query_base_url or os.environ.get("GRAPHRAG_QUERY_BASE_URL", "http://localhost:8081/v1")
    _qry_model = query_model or os.environ.get("GRAPHRAG_QUERY_MODEL", "Qwen3.5-4B-Q4_K_M")
    _emb_model = embed_model or os.environ.get("GRAPHRAG_EMBED_MODEL", "intfloat/e5-large-v2")
    _emb_cache = embed_hf_cache or os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home")
    _qdrant_url = qdrant_url or os.environ.get("GRAPHRAG_QDRANT_URL", "http://localhost:6335")
    _qdrant_coll = qdrant_collection or os.environ.get("GRAPHRAG_QDRANT_COLLECTION", "hars_graphrag")
    _vector_storage = vector_storage or os.environ.get("GRAPHRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
    _graph_storage = graph_storage or os.environ.get("GRAPHRAG_GRAPH_STORAGE", "NetworkXStorage")

    # Chunking: LightRAG is the single source of truth for token-based chunking.
    # Default 512 tokens/chunk so each extraction prompt stays well within the
    # server's per-slot context.  Override via env to tune without code changes.
    # Invariant: max_extract_input_tokens MUST be ≤ server per-slot context
    #   (llama-server -c / --parallel).  With -c 65536 --parallel 2 → 32768/slot;
    #   default 30720 leaves ~2 KB headroom for the system prompt.
    _chunk_token_size = int(os.environ.get("GRAPHRAG_CHUNK_TOKEN_SIZE", "512"))
    _chunk_overlap_tokens = int(os.environ.get("GRAPHRAG_CHUNK_OVERLAP_TOKENS", "64"))
    _llm_max_extract_tokens = int(os.environ.get("GRAPHRAG_LLM_MAX_TOKEN_SIZE", "30720"))
    # Max output tokens for the extractor LLM.  Dense 1024-token chunks require
    # more output than 4096 to list all entities + relations + descriptions + the
    # mandatory <|COMPLETE|> delimiter without truncation.  8192 is safe:
    # input 1024 + prompt ~1600 + output 8192 ≈ 10.8k ≪ 32768/slot server context.
    # INVARIANT: GRAPHRAG_EXTRACTOR_MAX_TOKENS ≤ server per-slot context minus input overhead.
    _ext_max_output_tokens = int(os.environ.get("GRAPHRAG_EXTRACTOR_MAX_TOKENS", "8192"))

    logger.info(
        "Creating LightRAG: working_dir=%s, extractor=%s@%s, query=%s@%s, embed=%s, vector_storage=%s, graph_storage=%s",
        _wdir,
        _ext_model, _ext_url,
        _qry_model, _qry_url,
        _emb_model,
        _vector_storage,
        _graph_storage,
    )
    if "qdrant" in _vector_storage.lower():
        logger.info("Qdrant target: %s (workspace=%s)", _qdrant_url, _qdrant_coll)
        # QdrantVectorDBStorage.initialize() reads QDRANT_URL directly from os.environ;
        # it does not accept the URL as a constructor argument.  Bridge the gap here
        # so that GRAPHRAG_QDRANT_URL controls the connection without leaking the
        # low-level env var into the rest of the process by default.
        # We only set it if the caller has not already set QDRANT_URL explicitly.
        if not os.environ.get("QDRANT_URL"):
            os.environ["QDRANT_URL"] = _qdrant_url
        # QDRANT_WORKSPACE isolates data within a shared collection; map our
        # collection config to it so multi-tenant separation is preserved.
        if not os.environ.get("QDRANT_WORKSPACE"):
            os.environ["QDRANT_WORKSPACE"] = _qdrant_coll

    Path(_wdir).mkdir(parents=True, exist_ok=True)

    embed_func = make_embedding_func(
        model_name=_emb_model,
        hf_cache_dir=_emb_cache,
        batch_size=embed_batch_size,
    )
    emb_dim = embedding_dimension(_emb_model)

    logger.info(
        "LightRAG chunking: chunk_token_size=%d, chunk_overlap_token_size=%d, "
        "max_extract_input_tokens=%d, extractor_max_output_tokens=%d",
        _chunk_token_size,
        _chunk_overlap_tokens,
        _llm_max_extract_tokens,
        _ext_max_output_tokens,
    )

    rag = LightRAG(
        working_dir=_wdir,
        vector_storage=_vector_storage,
        graph_storage=_graph_storage,
        tokenizer=Tokenizer("utf8-byte", _ByteTokenizer()),
        llm_model_func=make_llm_func(
            base_url=_ext_url,
            model=_ext_model,
            max_tokens=_ext_max_output_tokens,
            temperature=0.1,
        ),
        llm_model_name=_ext_model,
        embedding_func=EmbeddingFunc(
            embedding_dim=emb_dim,
            # Match the chunk size so LightRAG does not re-split chunks before
            # embedding.  Override explicitly with GRAPHRAG_EMBED_MAX_TOKENS if
            # your model has a stricter limit than GRAPHRAG_CHUNK_TOKEN_SIZE.
            max_token_size=int(os.environ.get("GRAPHRAG_EMBED_MAX_TOKENS", str(_chunk_token_size))),
            func=embed_func,
        ),
        # Chunking — LightRAG is the sole chunker; index.py passes whole docs.
        chunk_token_size=_chunk_token_size,
        chunk_overlap_token_size=_chunk_overlap_tokens,
        # No-truncation guarantee: extraction prompt input is capped here.
        # This MUST be ≤ the llama-server per-slot context (-c / --parallel).
        # With -c 65536 --parallel 2 → 32768 tokens/slot; default 30720 is safe.
        # NOTE: renamed from max_extract_input_tokens in lightrag-hku >= 1.5.0.
        max_total_tokens=_llm_max_extract_tokens,
        addon_params={
            "language": os.environ.get("GRAPHRAG_EXTRACTION_LANGUAGE", "English"),
            "entity_types": [entity_type.value for entity_type in EntityType],
        },
        max_parallel_insert=int(os.environ.get("GRAPHRAG_MAX_PARALLEL_INSERT", "2")),
    )
    return rag


def create_query_model_func() -> object:
    """Return the configured query LLM function.

    Indexing uses the extractor LLM wired into ``LightRAG``. Query/admin paths
    pass this function via ``QueryParam.model_func`` so normal MCP lookups do
    not hit the GPU-exclusive extractor by default.
    """
    return make_llm_func(
        base_url=os.environ.get("GRAPHRAG_QUERY_BASE_URL", "http://localhost:8081/v1"),
        model=os.environ.get("GRAPHRAG_QUERY_MODEL", "Qwen3.5-4B-Q4_K_M"),
        max_tokens=int(os.environ.get("GRAPHRAG_QUERY_MAX_TOKENS", "2048")),
        temperature=float(os.environ.get("GRAPHRAG_QUERY_TEMPERATURE", "0.1")),
    )


__all__ = ["create_lightrag", "create_query_model_func", "make_llm_func"]
