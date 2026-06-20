"""Sentence-transformers embedder for LightRAG.

Device defaults to CPU so queries and status checks work even while training
runs.  Set GRAPHRAG_EMBED_DEVICE=cuda (or "cuda:0") to offload to GPU — useful
for larger models like Qwen3-Embedding-0.6B that time-out on CPU during
LightRAG's 60-second embedding flush.

Swap to a different model by setting GRAPHRAG_EMBED_MODEL in .env — no code change.

Optional env vars
-----------------
GRAPHRAG_EMBED_DEVICE
    Torch device string passed to SentenceTransformer (e.g. "cpu", "cuda", "cuda:0").
    Default: "cpu".
GRAPHRAG_EMBED_DOC_PREFIX
    String prepended to every text before encoding (e.g. "passage: " for e5 models).
    Default: "" (no prefix).
GRAPHRAG_EMBED_PROMPT_NAME
    sentence-transformers prompt_name passed to model.encode() for models that
    support named prompts (e.g. embeddinggemma supports "document").
    Default: "" (not passed).  Silently ignored on older ST versions that lack it.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# LightRAG expects an async embedding function:
#   async def embed_func(texts: list[str]) -> numpy.ndarray  (shape: n_texts × dim, dtype float32)


@lru_cache(maxsize=1)
def _load_model(model_name: str, hf_cache_dir: str, batch_size: int) -> object:
    """Load and cache the SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    device: str = os.environ.get("GRAPHRAG_EMBED_DEVICE", "cpu")
    logger.info("Loading embedder: %s on %s (cache: %s)", model_name, device, hf_cache_dir)
    if hf_cache_dir:
        os.environ.setdefault("HF_HOME", hf_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)

    local_files_only = os.environ.get("GRAPHRAG_EMBED_LOCAL_FILES_ONLY", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    try:
        model = SentenceTransformer(
            model_name,
            device=os.environ.get("GRAPHRAG_EMBED_DEVICE", "cpu"),
            cache_folder=hf_cache_dir or None,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        mode = "local cache" if local_files_only else "local cache or Hugging Face"
        raise RuntimeError(
            f"Could not load embedding model '{model_name}' from {mode}. "
            "Set GRAPHRAG_EMBED_MODEL/HF_HOME to an available local model, or set "
            "GRAPHRAG_EMBED_LOCAL_FILES_ONLY=0 to allow downloads."
        ) from exc
    model._batch_size = batch_size  # type: ignore[attr-defined]
    _dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    logger.info("Embedder loaded: %s, embedding_dim=%d", model_name, _dim_fn())
    return model


def make_embedding_func(
    model_name: str = "intfloat/e5-large-v2",
    hf_cache_dir: str = "",
    batch_size: int = 32,
) -> object:
    """Return an async embedding function compatible with LightRAG.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Default: e5-large-v2 (dim=1024).
    hf_cache_dir:
        Override HF_HOME for offline operation (e.g. /mnt/datasets/models/.hf_home).
    batch_size:
        Batch size for CPU inference.

    Returns
    -------
    async callable:
        ``async (texts: list[str]) -> numpy.ndarray`` of shape (n_texts, dim), dtype float32.
    """
    model = _load_model(model_name, hf_cache_dir, batch_size)

    _doc_prefix: str = os.environ.get("GRAPHRAG_EMBED_DOC_PREFIX", "")
    _prompt_name: str = os.environ.get("GRAPHRAG_EMBED_PROMPT_NAME", "")

    async def embed(texts: list[str]) -> "np.ndarray":
        import asyncio
        import numpy as np

        def _encode() -> "np.ndarray":
            inputs = [_doc_prefix + t for t in texts] if _doc_prefix else texts
            encode_kwargs: dict[str, object] = {
                "batch_size": model._batch_size,  # type: ignore[attr-defined]
                "show_progress_bar": False,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
            }
            if _prompt_name:
                try:
                    vecs = model.encode(inputs, prompt_name=_prompt_name, **encode_kwargs)  # type: ignore[attr-defined]
                except TypeError:
                    # Older sentence-transformers versions don't support prompt_name
                    logger.warning(
                        "GRAPHRAG_EMBED_PROMPT_NAME=%r ignored: model.encode() does not "
                        "accept prompt_name (upgrade sentence-transformers>=5.0.0)",
                        _prompt_name,
                    )
                    vecs = model.encode(inputs, **encode_kwargs)  # type: ignore[attr-defined]
            else:
                vecs = model.encode(inputs, **encode_kwargs)  # type: ignore[attr-defined]
            return vecs.astype(np.float32)

        return await asyncio.to_thread(_encode)

    return embed


def embedding_dimension(model_name: str = "intfloat/e5-large-v2") -> int:
    """Return the output dimension for a given sentence-transformers model.

    Resolution order:
    1. Try to derive the dimension from the already-loaded (or loadable) model via
       ``get_sentence_embedding_dimension()`` — always authoritative.
    2. Fall back to ``_KNOWN_DIMS`` for offline/cold-start fast-path.
    3. Return 1024 as a last resort.
    """
    # Fast-path offline fallback — covers models that haven't been loaded yet and
    # models where loading might be undesired (e.g. during config validation).
    _KNOWN_DIMS: dict[str, int] = {
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-small-v2": 384,
        "BAAI/bge-m3": 1024,
        "nomic-ai/nomic-embed-text-v1": 768,
        "Qwen/Qwen3-Embedding-0.6B": 1024,
        "google/embeddinggemma-300m": 768,
    }

    # Prefer the model-derived dimension when the model is already cached in
    # _load_model (lru_cache).  This correctly handles any model not listed above.
    try:
        hf_cache_dir = os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home")
        batch_size = int(os.environ.get("GRAPHRAG_EMBED_BATCH_SIZE", "32"))
        model = _load_model(model_name, hf_cache_dir, batch_size)
        _dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(_dim_fn())
    except Exception:
        pass

    return _KNOWN_DIMS.get(model_name, 1024)
