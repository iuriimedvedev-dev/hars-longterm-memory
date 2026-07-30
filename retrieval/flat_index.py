"""Flat (chunk-level) dense retrieval index — no graph, no LLM extraction.

WHY this channel exists: the system's retrieval portfolio has a gap between
BM25 (exact tokens, index-bound freshness) and the LightRAG graph (multi-hop
relations, ~3h47m GPU build for 241 documents). Graph extraction is ~100% of
that build cost — it needs an LLM to produce entities/relationships. Plain
chunk embedding needs none of that: chunk text already exists on disk
(`kv_store_text_chunks.json`), and CPU sentence-embedding is the only cost.
That makes this channel rebuildable on a much shorter cycle (minutes, not
hours) than the graph, at the price of the graph's multi-hop edge (measured
elsewhere: `naive` matches `hybrid` on overall ndcg, but loses specifically
on multihop queries) — see the module's companion report for the measured
tradeoff on this corpus's labeled query set.

WHY reuse `kv_store_text_chunks.json` chunks, not re-chunk from source docs:
id-compatibility with every other channel. `retrieval/bm25_index.py` and
LightRAG's own dense channel (`rag.chunks_vdb`) both key their hits by this
same chunk `_id`; `retrieval/fusion.py::fuse()` merges dense_hits/sparse_hits
dicts on that key. Re-chunking from source documents with a retrieval-suited
strategy (sentence/paragraph-aware, not LightRAG's 2048-BYTE/256-overlap
byte-tokenizer split that has been observed to cut chunks mid-sentence)
would very likely improve standalone quality, but it breaks that shared id
space: a flat index built from different chunk boundaries cannot be fused
against BM25 at the chunk level without a separate id-remapping layer this
module does not build. This module ships the id-compatible (a) choice;
the re-chunked (b) alternative was measured separately (read-only, outside
this module — see the companion report) specifically to quantify what (a)
gives up, not implemented as a second on-disk index here.

WHY the same embedder.py doc/query asymmetric convention (no document
prefix; `HARS_MEMORY_EMBED_QUERY_PROMPT_NAME` on the query side, default
"query" -> "task: search result | query: "): this module never loads its
own embedding model — it takes an already-constructed embed function
(the same async `embed(texts, context=...)` callable
`server/embedder.py::make_embedding_func()` returns, and that
`server/lightrag_init.py` already wires into the live LightRAG instance) as
a parameter. Reusing that function, not a second model, is what keeps this
channel's vectors in the exact same space as `dense_only`'s (LightRAG's
`rag.chunks_vdb`) vectors automatically, including its asymmetry: calling
`embed_func(contents, context="document")` to build the index and
`embed_func([query], context="query")` to search reproduces exactly what
`server/lightrag_init.py` already does for the production dense channel, no
separate prefix constant to keep in sync by hand.

WHY plain numpy dot product, no ANN library (faiss/hnswlib/annoy): 7,184 x
768 float32 is ~22 MB, trivially resident in RAM; scoring one query is a
single (N, D) @ (D,) matmul, sub-millisecond on any CPU. ANN indexes exist
to avoid an O(N) scan over corpora many orders of magnitude larger than
this one, at the cost of approximate recall and real index-build/tuning
complexity. Revisit only if this corpus's chunk count grows into the
hundreds of thousands, or once a shared vector store (Qdrant, out of scope
for this module — a concurrent migration owns that) makes a matrix-in-numpy
index redundant rather than complementary.

WHY persisted (`.npy` vectors + JSON sidecar) + mtime-invalidated: mirrors
`retrieval/bm25_index.py`'s cache exactly (see that module's docstring for
the rationale) — the expensive part here is N CPU embedding calls, worth
paying once per source-file version, not once per process.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from tools.memory.retrieval.bm25_index import CHUNKS_FILENAME

logger = logging.getLogger(__name__)

_CACHE_META_FILENAME = "flat_dense_cache_meta.json"
_CACHE_VECTORS_FILENAME = "flat_dense_vectors.npy"

# `embed_func(texts, context=None, **kwargs) -> np.ndarray` — the exact shape
# of the async callable `server/embedder.py::make_embedding_func()` returns.
# Typed as `Any` return (not `np.ndarray`) to avoid a hard top-level `numpy`
# type-only import cycle; the real return is always coerced via
# `np.asarray(...)` before use below.
EmbedFunc = Callable[..., Awaitable[Any]]


class FlatIndexUnavailableError(RuntimeError):
    """Raised when the source `kv_store_text_chunks.json` does not exist yet.

    Same "nothing to do yet, not a real error" distinction as
    `retrieval.bm25_index.BM25IndexUnavailableError` — the LightRAG index
    itself hasn't been built.
    """


@dataclass(frozen=True)
class FlatSearchHit:
    chunk_id: str
    score: float
    content: str
    file_path: str


@dataclass(frozen=True)
class FlatBuildStats:
    chunk_count: int
    build_seconds: float
    source_mtime: float
    cache_hit: bool
    cache_dir: str
    embed_model: str


@dataclass
class FlatDenseIndex:
    """In-memory flat dense index: a normalized (chunk_count, dim) float32
    matrix + chunk metadata. No ANN structure — see module docstring.
    """

    chunk_ids: list[str]
    vectors: np.ndarray  # (chunk_count, dim), float32, L2-normalized rows
    chunk_meta: dict[str, dict[str, str]]
    source_mtime: float
    embed_model: str

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ids)

    async def search(self, query: str, embed_func: EmbedFunc, top_k: int) -> list[FlatSearchHit]:
        """Rank indexed chunks against `query`; up to `top_k` hits.

        `context="query"` is passed to `embed_func` deliberately — see module
        docstring for why this must match the query-side convention
        `server/embedder.py` applies for the production dense channel.

        `top_k` is clamped to the corpus size (mirrors
        `BM25SparseIndex.search`'s clamping — asking for more results than
        exist is a caller convenience, not a real request for padding).
        """
        if not query or self.chunk_count == 0:
            return []
        k = max(1, min(top_k, self.chunk_count))

        raw = await embed_func([query], context="query")
        query_vec = np.asarray(raw, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query_vec))
        if norm > 0.0:
            query_vec = query_vec / norm

        sims = self.vectors @ query_vec
        # argpartition for the top-k candidate set, then exact sort within
        # it — avoids a full O(N log N) sort for a corpus this size, though
        # at 7,184 rows either is sub-millisecond; kept for headroom as the
        # corpus grows toward the "revisit ANN" threshold noted above.
        if k >= len(sims):
            top_idx = np.argsort(-sims)
        else:
            candidate_idx = np.argpartition(-sims, k - 1)[:k]
            top_idx = candidate_idx[np.argsort(-sims[candidate_idx])]

        hits: list[FlatSearchHit] = []
        for idx in top_idx[:k]:
            chunk_id = self.chunk_ids[int(idx)]
            meta = self.chunk_meta.get(chunk_id, {})
            hits.append(
                FlatSearchHit(
                    chunk_id=chunk_id,
                    score=float(sims[int(idx)]),
                    content=str(meta.get("content", "")),
                    file_path=str(meta.get("file_path", "")),
                )
            )
        return hits


def _chunks_file_path(working_dir: str) -> Path:
    return Path(working_dir) / CHUNKS_FILENAME


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / _CACHE_META_FILENAME


def _vectors_path(cache_dir: Path) -> Path:
    return cache_dir / _CACHE_VECTORS_FILENAME


async def build_index(
    working_dir: str, embed_func: EmbedFunc, *, embed_model: str = ""
) -> tuple[FlatDenseIndex, float]:
    """Build a fresh `FlatDenseIndex` from `working_dir/kv_store_text_chunks.json`.

    Returns `(index, build_seconds)`. Raises `FlatIndexUnavailableError` if
    the source file does not exist. `embed_model` is stored alongside the
    vectors purely as a label for cache-invalidation / diagnostics (see
    `get_or_build_index`) — this module never loads a model itself, so it
    cannot independently verify the vectors' dimensionality matches the
    label; that is the caller's responsibility (same division of
    responsibility as `server/embedder.py::validate_embedder_against_index`
    uses for the production LightRAG index).
    """
    chunks_file = _chunks_file_path(working_dir)
    if not chunks_file.is_file():
        raise FlatIndexUnavailableError(
            f"No text-chunk store at {chunks_file} — build the LightRAG index first "
            "(python tools/memory/server/index.py)."
        )

    start = time.monotonic()
    with chunks_file.open("r", encoding="utf-8") as fh:
        raw_chunks: dict[str, dict[str, Any]] = json.load(fh)

    chunk_ids = sorted(raw_chunks.keys())  # deterministic positional order, matches bm25_index.py
    contents: list[str] = []
    chunk_meta: dict[str, dict[str, str]] = {}
    for chunk_id in chunk_ids:
        entry = raw_chunks[chunk_id]
        content = str(entry.get("content", ""))
        contents.append(content)
        chunk_meta[chunk_id] = {
            "content": content,
            "file_path": str(entry.get("file_path", "")),
        }

    if contents:
        raw_vectors = await embed_func(contents, context="document")
        vectors = np.asarray(raw_vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(contents):
            raise RuntimeError(
                f"embed_func returned shape {vectors.shape}, expected ({len(contents)}, dim)"
            )
        # Defensive re-normalization: server/embedder.py's own encode() call
        # already sets normalize_embeddings=True, but this module takes
        # embed_func as an opaque parameter and must not silently trust an
        # unnormalized caller — cosine similarity via plain dot product (see
        # FlatDenseIndex.search) is only correct on unit-norm rows.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vectors = (vectors / norms).astype(np.float32)
    else:
        vectors = np.zeros((0, 0), dtype=np.float32)
    build_seconds = time.monotonic() - start

    source_mtime = chunks_file.stat().st_mtime
    index = FlatDenseIndex(
        chunk_ids=chunk_ids,
        vectors=vectors,
        chunk_meta=chunk_meta,
        source_mtime=source_mtime,
        embed_model=embed_model,
    )
    logger.info(
        "Built flat dense index: %d chunks in %.3fs (source=%s, embed_model=%s)",
        len(chunk_ids), build_seconds, chunks_file, embed_model or "<unspecified>",
    )
    return index, build_seconds


def save_index(index: FlatDenseIndex, cache_dir: str) -> None:
    """Persist `index` to `cache_dir`: vectors as `.npy`, everything else
    (chunk_ids/chunk_meta/source_mtime/embed_model) as a JSON sidecar —
    mirrors `retrieval/bm25_index.py::save_index`'s split between a
    library-native binary format and a hand-written metadata file.
    """
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(_vectors_path(out_dir), index.vectors)
    meta = {
        "chunk_ids": index.chunk_ids,
        "chunk_meta": index.chunk_meta,
        "source_mtime": index.source_mtime,
        "embed_model": index.embed_model,
    }
    _meta_path(out_dir).write_text(json.dumps(meta), encoding="utf-8")


def load_index(cache_dir: str) -> FlatDenseIndex | None:
    """Load a previously persisted index from `cache_dir`, or `None` if absent.

    `None` (not an exception) on a missing cache mirrors
    `retrieval/bm25_index.py::load_index` — "no cache yet" is expected
    first-run state, not an error; `get_or_build_index` decides what to do.

    Vectors are loaded with `mmap_mode="r"` — cheap process startup (mirrors
    bm25s' own `mmap=True` used by `retrieval/bm25_index.py::load_index`);
    the returned array is read-only, which is correct here since nothing
    mutates `FlatDenseIndex.vectors` after a build.
    """
    out_dir = Path(cache_dir)
    meta_file = _meta_path(out_dir)
    vectors_file = _vectors_path(out_dir)
    if not meta_file.is_file() or not vectors_file.is_file():
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    vectors = np.load(vectors_file, mmap_mode="r")
    return FlatDenseIndex(
        chunk_ids=list(meta["chunk_ids"]),
        vectors=vectors,
        chunk_meta=dict(meta["chunk_meta"]),
        source_mtime=float(meta["source_mtime"]),
        embed_model=str(meta.get("embed_model", "")),
    )


async def get_or_build_index(
    working_dir: str,
    cache_dir: str,
    embed_func: EmbedFunc,
    *,
    embed_model: str = "",
) -> tuple[FlatDenseIndex, FlatBuildStats]:
    """Return a ready-to-query `FlatDenseIndex`, rebuilding only when the
    source `kv_store_text_chunks.json` mtime has changed since the on-disk
    cache was written, no cache exists yet, or the cached `embed_model`
    label disagrees with the one requested now (guards against silently
    querying stale vectors from a different embedding space after an
    embedder swap — same failure class
    `server/embedder.py::EmbeddingDimensionMismatchError` exists to catch
    for the production LightRAG index, applied here as a cache-miss instead
    of a hard error since rebuilding is cheap and always safe).

    Raises `FlatIndexUnavailableError` if the source chunk store itself does
    not exist (LightRAG index not built yet).
    """
    chunks_file = _chunks_file_path(working_dir)
    if not chunks_file.is_file():
        raise FlatIndexUnavailableError(
            f"No text-chunk store at {chunks_file} — build the LightRAG index first "
            "(python tools/memory/server/index.py)."
        )
    current_mtime = chunks_file.stat().st_mtime

    cached = load_index(cache_dir)
    embed_model_matches = not embed_model or not cached or cached.embed_model in ("", embed_model)
    if cached is not None and cached.source_mtime == current_mtime and embed_model_matches:
        return cached, FlatBuildStats(
            chunk_count=cached.chunk_count,
            build_seconds=0.0,
            source_mtime=current_mtime,
            cache_hit=True,
            cache_dir=cache_dir,
            embed_model=cached.embed_model,
        )

    index, build_seconds = await build_index(working_dir, embed_func, embed_model=embed_model)
    save_index(index, cache_dir)
    return index, FlatBuildStats(
        chunk_count=index.chunk_count,
        build_seconds=build_seconds,
        source_mtime=current_mtime,
        cache_hit=False,
        cache_dir=cache_dir,
        embed_model=embed_model,
    )


__all__ = [
    "CHUNKS_FILENAME",
    "EmbedFunc",
    "FlatIndexUnavailableError",
    "FlatSearchHit",
    "FlatBuildStats",
    "FlatDenseIndex",
    "build_index",
    "save_index",
    "load_index",
    "get_or_build_index",
]
