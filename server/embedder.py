"""CPU-only e5-large-v2 embedder for LightRAG.

This module is always GPU-free: device is forced to "cpu" regardless of what
the config says, so queries and status checks work even while training runs.

Swap to a different model by setting GRAPHRAG_EMBED_MODEL in .env — no code change.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# LightRAG expects an async embedding function:
#   async def embed_func(texts: list[str]) -> list[list[float]]


@lru_cache(maxsize=1)
def _load_model(model_name: str, hf_cache_dir: str, batch_size: int) -> object:
    """Load and cache the SentenceTransformer model (CPU only)."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    logger.info("Loading embedder: %s on CPU (cache: %s)", model_name, hf_cache_dir)
    if hf_cache_dir:
        os.environ.setdefault("HF_HOME", hf_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)

    model = SentenceTransformer(model_name, device="cpu", cache_folder=hf_cache_dir or None)
    model._batch_size = batch_size  # type: ignore[attr-defined]
    logger.info("Embedder loaded: %s, embedding_dim=%d", model_name, model.get_sentence_embedding_dimension())
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
        ``async (texts: list[str]) -> list[list[float]]``
    """
    model = _load_model(model_name, hf_cache_dir, batch_size)

    async def embed(texts: list[str]) -> list[list[float]]:
        import asyncio

        def _encode() -> list[list[float]]:
            vecs = model.encode(  # type: ignore[attr-defined]
                texts,
                batch_size=model._batch_size,  # type: ignore[attr-defined]
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return [v.tolist() for v in vecs]

        return await asyncio.to_thread(_encode)

    return embed


def embedding_dimension(model_name: str = "intfloat/e5-large-v2") -> int:
    """Return the output dimension for a given sentence-transformers model."""
    # Known dimensions without loading the model
    _KNOWN_DIMS: dict[str, int] = {
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-small-v2": 384,
        "BAAI/bge-m3": 1024,
        "nomic-ai/nomic-embed-text-v1": 768,
    }
    return _KNOWN_DIMS.get(model_name, 1024)
