"""Sentence-transformers embedder for LightRAG.

Device defaults to CPU so queries and status checks work even while training
runs.  Set HARS_MEMORY_EMBED_DEVICE=cuda (or "cuda:0") to offload to GPU — useful
for larger models like Qwen3-Embedding-0.6B that time-out on CPU during
LightRAG's 60-second embedding flush.

Swap to a different model by setting HARS_MEMORY_EMBED_MODEL in .env — no code change.

Optional env vars
-----------------
HARS_MEMORY_EMBED_DEVICE
    Torch device string passed to SentenceTransformer (e.g. "cpu", "cuda", "cuda:0").
    Default: "cpu".
HARS_MEMORY_EMBED_DOC_PREFIX
    String prepended to every text before encoding (e.g. "passage: " for e5 models).
    Default: "" (no prefix).
HARS_MEMORY_EMBED_PROMPT_NAME
    sentence-transformers prompt_name passed to model.encode() for models that
    support named prompts (e.g. embeddinggemma supports "document").
    Default: "" (not passed).  Silently ignored on older ST versions that lack it.
    Applied unconditionally to every call (document AND query) for backward
    compatibility — see HARS_MEMORY_EMBED_QUERY_PROMPT_NAME below for the
    context-aware, query-only alternative.
HARS_MEMORY_EMBED_QUERY_PROMPT_NAME
    sentence-transformers prompt_name applied ONLY when LightRAG calls the
    embedding function with context="query" (i.e. embedding a search query,
    not a document being indexed) — see `lightrag.utils.EmbeddingFunc`'s
    `supports_asymmetric` flag, which `lightrag_init.create_lightrag()` sets
    to True so this context is actually delivered.
    Default: "query" — this matches embeddinggemma's built-in prompts dict
    (`config_sentence_transformers.json`: "query" -> "task: search result |
    query: "). Safe to change/enable at any time: it only affects freshly
    computed query embeddings, never the stored document vectors, so it does
    NOT require re-embedding an existing index.
    Document-side embedding (context="document") is UNAFFECTED by this var —
    it keeps using HARS_MEMORY_EMBED_PROMPT_NAME/HARS_MEMORY_EMBED_DOC_PREFIX above
    (default: no prefix at all), which is what the existing index at
    /home/user/.local/share/hars-graphrag/index_gemma_v4 (relocated
    2026-07-29 from /mnt/datasets/graphrag/index_gemma_v4) was actually
    built with. Changing
    the document-side prompt WOULD require a full re-embed — see
    embeddinggemma's "document" prompt ("title: none | text: ") for the
    theoretically-correct (but re-embed-requiring) setting.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# LightRAG expects an async embedding function:
#   async def embed_func(texts: list[str]) -> numpy.ndarray  (shape: n_texts × dim, dtype float32)


@lru_cache(maxsize=1)
def _load_model(model_name: str, hf_cache_dir: str, batch_size: int) -> object:
    """Load and cache the SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    device: str = os.environ.get("HARS_MEMORY_EMBED_DEVICE", "cpu")
    logger.info("Loading embedder: %s on %s (cache: %s)", model_name, device, hf_cache_dir)
    if hf_cache_dir:
        os.environ.setdefault("HF_HOME", hf_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)

    local_files_only = os.environ.get("HARS_MEMORY_EMBED_LOCAL_FILES_ONLY", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    # HF CLI downloads land under <HF_HOME>/hub/, legacy sentence-transformers
    # downloads under <HF_HOME>/ directly — try both cache roots.
    cache_candidates = [hf_cache_dir or None]
    hub_dir = os.path.join(hf_cache_dir, "hub") if hf_cache_dir else ""
    if hub_dir and os.path.isdir(hub_dir):
        cache_candidates.append(hub_dir)
    model = None
    last_exc: Exception | None = None
    for cache_folder in cache_candidates:
        try:
            model = SentenceTransformer(
                model_name,
                device=os.environ.get("HARS_MEMORY_EMBED_DEVICE", "cpu"),
                cache_folder=cache_folder,
                local_files_only=local_files_only,
            )
            break
        except Exception as exc:
            last_exc = exc
    if model is None:
        exc = last_exc  # type: ignore[assignment]
        mode = "local cache" if local_files_only else "local cache or Hugging Face"
        raise RuntimeError(
            f"Could not load embedding model '{model_name}' from {mode}. "
            "Set HARS_MEMORY_EMBED_MODEL/HF_HOME to an available local model, or set "
            "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY=0 to allow downloads."
        ) from exc
    model._batch_size = batch_size  # type: ignore[attr-defined]
    _dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    logger.info("Embedder loaded: %s, embedding_dim=%d", model_name, _dim_fn())
    return model


def make_embedding_func(
    model_name: str = "unsloth/embeddinggemma-300m",
    hf_cache_dir: str = "",
    batch_size: int = 32,
) -> object:
    """Return an async embedding function compatible with LightRAG.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Default: unsloth/embeddinggemma-300m (dim=768) —
        this matches the model the deployed index at
        /home/user/.local/share/hars-graphrag/index_gemma_v4 (relocated
        2026-07-29 from /mnt/datasets/graphrag/index_gemma_v4) was actually built with.
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

    _doc_prefix: str = os.environ.get("HARS_MEMORY_EMBED_DOC_PREFIX", "")
    _prompt_name: str = os.environ.get("HARS_MEMORY_EMBED_PROMPT_NAME", "")
    _query_prompt_name: str = os.environ.get("HARS_MEMORY_EMBED_QUERY_PROMPT_NAME", "query")

    async def embed(texts: list[str], context: str | None = None, **_kwargs: object) -> "np.ndarray":
        import asyncio
        import numpy as np

        # LightRAG (>=1.4.16) calls embedding_func(batch, context="document")
        # at insert time and embedding_func([query], context="query", ...) at
        # query time — see lightrag/kg/nano_vector_db_impl.py — but only
        # forwards `context` when EmbeddingFunc.supports_asymmetric=True
        # (lightrag_init.create_lightrag() sets this). context is None for
        # any older/legacy caller, which falls back to document-side
        # behaviour (today's default: no prefix), matching the existing
        # index exactly.
        effective_prompt_name = _query_prompt_name if context == "query" else _prompt_name

        def _encode() -> "np.ndarray":
            inputs = [_doc_prefix + t for t in texts] if _doc_prefix else texts
            encode_kwargs: dict[str, object] = {
                "batch_size": model._batch_size,  # type: ignore[attr-defined]
                "show_progress_bar": False,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
            }
            if effective_prompt_name:
                try:
                    vecs = model.encode(inputs, prompt_name=effective_prompt_name, **encode_kwargs)  # type: ignore[attr-defined]
                except TypeError:
                    # Older sentence-transformers versions don't support prompt_name
                    logger.warning(
                        "prompt_name=%r ignored: model.encode() does not accept "
                        "prompt_name (upgrade sentence-transformers>=5.0.0)",
                        effective_prompt_name,
                    )
                    vecs = model.encode(inputs, **encode_kwargs)  # type: ignore[attr-defined]
            else:
                vecs = model.encode(inputs, **encode_kwargs)  # type: ignore[attr-defined]
            return vecs.astype(np.float32)

        return await asyncio.to_thread(_encode)

    return embed


def embedding_dimension(model_name: str = "unsloth/embeddinggemma-300m") -> int:
    """Return the output dimension for a given sentence-transformers model.

    Resolution order:
    1. Try to derive the dimension from the already-loaded (or loadable) model via
       ``get_sentence_embedding_dimension()`` — always authoritative.
    2. Fall back to ``_KNOWN_DIMS`` for offline/cold-start fast-path.
    3. Raise — a silent wrong-dimension guess is exactly the failure mode this
       module exists to eliminate (see EmbeddingDimensionMismatchError below).
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
        "unsloth/embeddinggemma-300m": 768,
    }

    # Prefer the model-derived dimension when the model is already cached in
    # _load_model (lru_cache).  This correctly handles any model not listed above.
    try:
        hf_cache_dir = os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home")
        batch_size = int(os.environ.get("HARS_MEMORY_EMBED_BATCH_SIZE", "32"))
        model = _load_model(model_name, hf_cache_dir, batch_size)
        _dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(_dim_fn())
    except Exception:
        pass

    if model_name in _KNOWN_DIMS:
        return _KNOWN_DIMS[model_name]

    raise RuntimeError(
        f"Cannot resolve embedding dimension for model '{model_name}': the model "
        "could not be loaded and it is not listed in embedder._KNOWN_DIMS. "
        "Add it to _KNOWN_DIMS, or fix HARS_MEMORY_EMBED_MODEL/HF_HOME so the model "
        "can be loaded. Guessing a dimension here would silently produce "
        "wrong-dimension vectors — see EmbeddingDimensionMismatchError."
    )


class EmbeddingDimensionMismatchError(RuntimeError):
    """Raised when the configured embedder's output dimension does not match the
    dimension recorded in an existing LightRAG vector index at ``working_dir``.

    This is the fail-fast guard for the exact bug class it is named after: if the
    embedder config silently disagrees with the index, queries get embedded into
    the wrong vector space and retrieval returns garbage with no error. Never
    warn-and-continue here — always raise.
    """

    def __init__(self, working_dir: str, model_name: str, configured_dim: int, index_dim: int) -> None:
        super().__init__(
            f"Embedder/index dimension mismatch at working_dir={working_dir!r}: "
            f"configured embedder HARS_MEMORY_EMBED_MODEL='{model_name}' produces "
            f"{configured_dim}-dim vectors, but the existing index recorded "
            f"embedding_dim={index_dim}. Rebuild the index with the configured "
            "embedder, or set HARS_MEMORY_EMBED_MODEL to the model the index was "
            "actually built with."
        )
        self.working_dir = working_dir
        self.model_name = model_name
        self.configured_dim = configured_dim
        self.index_dim = index_dim


# LightRAG/NanoVectorDBStorage writes `{"embedding_dim": <int>, "data": [...]}` with
# `embedding_dim` as the first key of the JSON object — reading a small prefix of the
# file is enough, no need to parse the full (tens-to-hundreds of MB) vector store.
_VDB_FILENAMES: tuple[str, ...] = ("vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json")
_VDB_PROBE_BYTES = 256
_EMBEDDING_DIM_RE = re.compile(r'"embedding_dim"\s*:\s*(\d+)')


def resolve_index_embedding_dim(working_dir: str) -> int | None:
    """Return the ``embedding_dim`` recorded in an existing vector index, or ``None``.

    Checks LightRAG's ``vdb_*.json`` sibling files in ``working_dir`` in priority
    order (chunks first — it is both the smallest file and the one used by
    local/naive retrieval) and reads only the first bytes of whichever file is
    found first, since NanoVectorDBStorage writes ``embedding_dim`` as the first
    key. Returns ``None`` when none of the vdb files exist yet (brand-new,
    not-yet-indexed working dir) — callers must treat that as "nothing to
    validate against", not as an error.
    """
    for filename in _VDB_FILENAMES:
        path = Path(working_dir) / filename
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                head = fh.read(_VDB_PROBE_BYTES)
        except OSError as exc:
            raise RuntimeError(f"Could not read vector index file {path}: {exc}") from exc
        match = _EMBEDDING_DIM_RE.search(head)
        if match is None:
            raise RuntimeError(
                f"Vector index file {path} exists but its first {_VDB_PROBE_BYTES} bytes do "
                "not contain an 'embedding_dim' field — index format may have changed or the "
                "file is corrupt."
            )
        return int(match.group(1))
    return None


def validate_embedder_against_index(working_dir: str, model_name: str, configured_dim: int) -> None:
    """Fail fast if the configured embedder disagrees with an existing index.

    No-op when ``working_dir`` has no vector index yet (brand-new working dir) —
    there is nothing to validate against and a fresh index will simply be built
    with ``configured_dim``.
    """
    index_dim = resolve_index_embedding_dim(working_dir)
    if index_dim is not None and index_dim != configured_dim:
        raise EmbeddingDimensionMismatchError(working_dir, model_name, configured_dim, index_dim)
