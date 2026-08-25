"""Local CPU cross-encoder reranker for LightRAG.

LightRAG 1.4.16 (`lightrag/base.py`) defaults `QueryParam.enable_rerank` to
``True`` (``os.getenv("RERANK_BY_DEFAULT", "true").lower() == "true"``), but
`LightRAG.rerank_model_func` (`lightrag/lightrag.py`) defaults to ``None``.
When rerank is enabled but no function is configured,
`lightrag/utils.py::apply_rerank_if_enabled` logs::

    "Rerank is enabled but no rerank model is configured. Please set up a
    rerank model or set enable_rerank=False in query parameters."

and returns the retrieved documents unchanged — i.e. reranking silently does
nothing today. This module supplies the missing in-process function.

All four of LightRAG's built-in rerank bindings (`cohere_rerank`,
`jina_rerank`, `ali_rerank`, `generic_rerank_api`, see `lightrag/rerank.py`)
are HTTP calls to a remote rerank API. There is no in-process local
cross-encoder binding in the library — this module is one.

Chosen model: cross-encoder/ettin-reranker-68m-v1 (Apache-2.0, 68M params,
8K token context — no chunk-splitting workaround needed for our
512-1024-token chunks). Deliberately NOT bge-reranker-v2-m3 (512-token
context, ~6 pairs/s on CPU, and weak on code retrieval per MTEB-Code).

Runs on CPU — same rationale as `embedder.py`: the GPU is reserved for
training, and queries (including reranking) must work at any time.

Optional env vars
------------------
HARS_MEMORY_RERANK_MODEL
    HuggingFace cross-encoder model ID. Unset (default) => reranking is
    disabled entirely: `lightrag_init.create_lightrag()` passes
    `rerank_model_func=None`, preserving today's behaviour, and no model is
    ever loaded.
HARS_MEMORY_RERANK_DEVICE
    Torch device string passed to CrossEncoder (e.g. "cpu", "cuda").
    Default: "cpu".
HARS_MEMORY_RERANK_LOCAL_FILES_ONLY
    Mirrors HARS_MEMORY_EMBED_LOCAL_FILES_ONLY's convention. Default: "1"
    (offline; do not hit the network / HF Hub at query time).
HARS_MEMORY_RERANK_BATCH_SIZE
    CPU batch size for CrossEncoder.predict(). Default: 1.
    Measured on this corpus (tools/memory chunks average ~540 tokens, up to
    ~900 in a 50-doc random sample — much longer than the short passages the
    published "~31.2 pairs/sec" ettin-reranker CPU figure almost certainly
    used): batch_size=1 measured ~150ms/pair, batch_size=32 measured
    ~330ms/pair — 2x SLOWER, not faster. Cause: sentence-transformers pads
    every item in a batch to that batch's longest sequence; with this
    corpus's high length variance, padding waste outweighs batching's
    parallelism benefit on CPU. Raise this only if your corpus has short,
    length-homogeneous chunks.

Score scale: cross-encoder/ettin-reranker-68m-v1's final Dense layer uses an
Identity activation (see its `README.md` / `config_sentence_transformers.json`
— NOT a sigmoid), so `relevance_score` is a raw, unbounded logit (the model's
own example gives ~11.5 for a strong match), not a [0, 1] probability. Do not
assume 0.5 is a meaningful cutoff — observe the score distribution on real
queries before setting `HARS_MEMORY_MIN_RERANK_SCORE`/`MIN_RERANK_SCORE` above 0.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# LightRAG's required rerank function contract (lightrag/utils.py::apply_rerank_if_enabled):
#   async def rerank_func(query: str, documents: list, top_n: int | None = None, **kwargs)
#       -> list[{"index": int, "relevance_score": float}]


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str, hf_cache_dir: str, local_files_only: bool) -> object:
    """Load and cache the CrossEncoder model. Only called on first actual rerank call."""
    from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

    logger.info(
        "Loading reranker: %s on %s (cache: %s, local_files_only=%s)",
        model_name,
        device,
        hf_cache_dir,
        local_files_only,
    )
    if hf_cache_dir:
        os.environ.setdefault("HF_HOME", hf_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)

    # Same dual-cache-root probing as embedder.py::_load_model — `hf download`
    # lands models under <HF_HOME>/hub/, legacy sentence-transformers downloads
    # land under <HF_HOME>/ directly.
    cache_candidates: list[str | None] = [hf_cache_dir or None]
    hub_dir = os.path.join(hf_cache_dir, "hub") if hf_cache_dir else ""
    if hub_dir and os.path.isdir(hub_dir):
        cache_candidates.append(hub_dir)

    model = None
    last_exc: Exception | None = None
    for cache_folder in cache_candidates:
        try:
            model = CrossEncoder(
                model_name,
                device=device,
                cache_folder=cache_folder,
                local_files_only=local_files_only,
            )
            break
        except Exception as exc:
            last_exc = exc
    if model is None:
        mode = "local cache" if local_files_only else "local cache or Hugging Face"
        raise RuntimeError(
            f"Could not load reranker model '{model_name}' from {mode}. "
            "Set HARS_MEMORY_RERANK_MODEL/HF_HOME to an available local model, or set "
            "HARS_MEMORY_RERANK_LOCAL_FILES_ONLY=0 to allow downloads."
        ) from last_exc
    logger.info("Reranker loaded: %s", model_name)
    return model


def make_rerank_func(
    model_name: str,
    device: str = "cpu",
    hf_cache_dir: str = "",
    batch_size: int = 1,
    local_files_only: bool = True,
) -> object:
    """Return an async LightRAG-compatible ``rerank_model_func``.

    The returned closure does NOT load the model. The model is lazily loaded
    on the first actual call (inside the ``asyncio.to_thread`` worker), so
    constructing this function (e.g. because ``HARS_MEMORY_RERANK_MODEL`` is
    set) costs nothing until reranking is actually invoked.

    Parameters
    ----------
    model_name:
        HuggingFace cross-encoder model ID.
    device:
        Torch device for CrossEncoder. Default: "cpu".
    hf_cache_dir:
        Override HF_HOME for offline operation.
    batch_size:
        CPU batch size for CrossEncoder.predict().
    local_files_only:
        Forbid network access; require the model to already be cached.

    Returns
    -------
    async callable matching LightRAG's rerank_model_func contract:
        ``async (query, documents, top_n=None, **kwargs) -> list[{"index", "relevance_score"}]``
    """

    async def rerank(
        query: str,
        documents: list[Any],
        top_n: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        import asyncio

        if not documents:
            return []

        def _predict() -> list[float]:
            model = _load_model(model_name, device, hf_cache_dir, local_files_only)
            # LightRAG's apply_rerank_if_enabled() always passes plain strings
            # (already extracted from the "content"/"text"/"chunk_content"/
            # "document" fields of each retrieved chunk dict) — see
            # lightrag/utils.py. Coerce defensively in case a caller (e.g. a
            # test) passes something else.
            pairs = [(query, doc if isinstance(doc, str) else str(doc)) for doc in documents]
            scores = model.predict(  # type: ignore[attr-defined]
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [float(s) for s in scores]

        scores = await asyncio.to_thread(_predict)

        results = [
            {"index": i, "relevance_score": score} for i, score in enumerate(scores)
        ]
        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results

    return rerank


__all__ = ["make_rerank_func"]
