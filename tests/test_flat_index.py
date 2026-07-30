"""Unit tests for tools/memory/retrieval/flat_index.py — the flat (chunk-level)
dense retrieval index. GPU-free, no real embedding model, no network.

Uses a small deterministic fake `embed_func` (hand-computed 2D unit vectors,
same async `embed(texts, context=...) -> np.ndarray` signature
`server/embedder.py::make_embedding_func()` returns) instead of loading the
real `unsloth/embeddinggemma-300m` model — the point of these tests is the
build/persist/cache-invalidation/search-shape logic in flat_index.py itself,
not embedding quality (that is measured separately, against the real model,
in the companion eval report). Mirrors `test_bm25_retrieval.py`'s
`TestBM25IndexCache` structure and fixtures as closely as the dense case
allows.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from tools.memory.retrieval.flat_index import (
    FlatIndexUnavailableError,
    build_index,
    get_or_build_index,
    load_index,
    save_index,
)

# ---------------------------------------------------------------------------
# Fake embedder: deterministic 2D unit vectors placed at known angles, so
# cosine-similarity ranking is hand-verifiable. Content strings below are
# mapped to fixed vectors keyed by the FIRST WORD (so it also works for
# "query"-context questions that aren't verbatim chunk content).
# ---------------------------------------------------------------------------

_VECTOR_BY_KEYWORD: dict[str, tuple[float, float]] = {
    "alpha": (1.0, 0.0),
    "beta": (0.0, 1.0),
    "gamma": (-1.0, 0.0),
}


def _vector_for(text: str) -> tuple[float, float]:
    lower = text.casefold()
    for keyword, vec in _VECTOR_BY_KEYWORD.items():
        if keyword in lower:
            return vec
    return (0.6, 0.8)  # arbitrary off-axis default, still unit norm


async def _fake_embed(texts: list[str], context: str | None = None, **_kwargs: object) -> np.ndarray:
    return np.array([_vector_for(t) for t in texts], dtype=np.float32)


class _RecordingEmbed:
    """Wraps `_fake_embed`, recording every `context` it was called with —
    used to assert the query-side prefix/context convention is actually
    applied (not silently dropped).
    """

    def __init__(self) -> None:
        self.contexts: list[str | None] = []

    async def __call__(self, texts: list[str], context: str | None = None, **kwargs: object) -> np.ndarray:
        self.contexts.append(context)
        return await _fake_embed(texts, context=context, **kwargs)


def _write_chunks(working_dir: Path, chunks: dict[str, dict]) -> None:
    (working_dir / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


def _sample_chunks() -> dict[str, dict]:
    return {
        "chunk-1": {"content": "alpha document about gate stabilization.", "file_path": "a.md"},
        "chunk-2": {"content": "beta document about lora training.", "file_path": "b.md"},
        "chunk-3": {"content": "gamma document about vea native mode.", "file_path": "c.md"},
    }


class TestBuildIndex:
    def test_missing_source_raises_unavailable(self, tmp_path: Path) -> None:
        with pytest.raises(FlatIndexUnavailableError):
            asyncio.run(build_index(str(tmp_path), _fake_embed))

    def test_build_index_indexes_expected_chunk_count(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, _sample_chunks())
        index, build_seconds = asyncio.run(build_index(str(tmp_path), _fake_embed))
        assert index.chunk_count == 3
        assert build_seconds >= 0.0
        assert index.vectors.shape == (3, 2)

    def test_build_index_normalizes_vectors(self, tmp_path: Path) -> None:
        # Fake embedder already returns unit vectors, so this exercises the
        # defensive re-normalization path is a no-op for already-normalized
        # input (not that it breaks it).
        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        norms = np.linalg.norm(index.vectors, axis=1)
        assert norms == pytest.approx([1.0, 1.0, 1.0])

    def test_build_index_renormalizes_unnormalized_embed_func(self, tmp_path: Path) -> None:
        async def unnormalized_embed(texts: list[str], context: str | None = None, **_: object) -> np.ndarray:
            # Deliberately NOT unit-norm (magnitude 5), to prove build_index
            # does not blindly trust the caller's embed_func.
            return np.array([[5.0, 0.0] for _ in texts], dtype=np.float32)

        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), unnormalized_embed))
        norms = np.linalg.norm(index.vectors, axis=1)
        assert norms == pytest.approx([1.0, 1.0, 1.0])


class TestEmptyAndSingleChunkEdgeCases:
    def test_empty_corpus_builds_zero_row_index(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, {})
        index, build_seconds = asyncio.run(build_index(str(tmp_path), _fake_embed))
        assert index.chunk_count == 0
        assert build_seconds >= 0.0
        hits = asyncio.run(index.search("alpha query", _fake_embed, top_k=5))
        assert hits == []

    def test_single_chunk_corpus_returns_that_one_hit(self, tmp_path: Path) -> None:
        _write_chunks(
            tmp_path, {"chunk-only": {"content": "alpha only document.", "file_path": "only.md"}}
        )
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        assert index.chunk_count == 1
        # top_k requested larger than corpus size must clamp, not raise.
        hits = asyncio.run(index.search("alpha query", _fake_embed, top_k=50))
        assert len(hits) == 1
        assert hits[0].chunk_id == "chunk-only"
        assert hits[0].file_path == "only.md"

    def test_empty_query_returns_no_hits(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        assert asyncio.run(index.search("", _fake_embed, top_k=5)) == []


class TestSearchShapeAndRanking:
    def test_search_returns_expected_shape(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        hits = asyncio.run(index.search("alpha query", _fake_embed, top_k=2))
        assert len(hits) == 2
        for hit in hits:
            assert isinstance(hit.chunk_id, str)
            assert isinstance(hit.score, float)
            assert isinstance(hit.content, str)
            assert isinstance(hit.file_path, str)

    def test_search_ranks_closest_vector_first(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        hits = asyncio.run(index.search("beta query", _fake_embed, top_k=3))
        assert hits[0].chunk_id == "chunk-2"  # beta vector is the exact match
        assert hits[0].score == pytest.approx(1.0)
        # gamma is anti-parallel to alpha and orthogonal to beta -> must rank
        # strictly below the exact beta match.
        assert hits[0].score > hits[-1].score

    def test_top_k_clamped_to_corpus_size(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        hits = asyncio.run(index.search("alpha query", _fake_embed, top_k=100))
        assert len(hits) == 3

    def test_query_context_is_applied(self, tmp_path: Path) -> None:
        """Item: the query-side prefix/context convention must actually be
        applied on every query call, consistently (not silently dropped)."""
        _write_chunks(tmp_path, _sample_chunks())
        recorder = _RecordingEmbed()
        index, _ = asyncio.run(build_index(str(tmp_path), recorder))
        assert recorder.contexts == ["document"]  # build-time call

        recorder.contexts.clear()
        asyncio.run(index.search("alpha query", recorder, top_k=1))
        asyncio.run(index.search("beta query", recorder, top_k=1))
        assert recorder.contexts == ["query", "query"], (
            "every search() call must pass context='query' to embed_func, matching "
            "server/embedder.py's asymmetric document/query convention"
        )


class TestPersistenceAndCacheInvalidation:
    def test_get_or_build_index_caches_on_disk(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _write_chunks(working_dir, _sample_chunks())

        _index1, stats1 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats1.cache_hit is False
        assert stats1.chunk_count == 3
        assert (cache_dir / "flat_dense_cache_meta.json").is_file()
        assert (cache_dir / "flat_dense_vectors.npy").is_file()

        _index2, stats2 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats2.cache_hit is True
        assert stats2.chunk_count == 3

    def test_cache_invalidates_on_mtime_change(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"
        _write_chunks(working_dir, _sample_chunks())

        _index1, stats1 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats1.cache_hit is False

        richer_chunks = _sample_chunks()
        richer_chunks["chunk-4"] = {"content": "alpha new chunk added after re-ingest.", "file_path": "new.md"}
        _write_chunks(working_dir, richer_chunks)
        future = time.time() + 5
        os.utime(chunks_file, (future, future))

        index2, stats2 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats2.cache_hit is False, "mtime changed — must rebuild, not reuse stale cache"
        assert index2.chunk_count == 4

    def test_reloaded_index_still_searches_correctly(self, tmp_path: Path) -> None:
        """The disk-persisted (cache_hit=True) path must return identical
        results to a fresh in-memory build — proves save/load round-trips
        the vector matrix + chunk metadata correctly."""
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _write_chunks(working_dir, _sample_chunks())

        index1, stats1 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats1.cache_hit is False
        hits1 = asyncio.run(index1.search("beta query", _fake_embed, top_k=3))

        index2, stats2 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats2.cache_hit is True
        hits2 = asyncio.run(index2.search("beta query", _fake_embed, top_k=3))

        assert [h.chunk_id for h in hits1] == [h.chunk_id for h in hits2]
        assert [h.score for h in hits1] == pytest.approx([h.score for h in hits2])

    def test_cache_invalidates_on_embed_model_change(self, tmp_path: Path) -> None:
        """A cache built under one embed_model label must not be silently
        reused when a different embed_model is requested — guards against
        querying stale vectors from a different embedding space after an
        embedder swap (see flat_index.py::get_or_build_index docstring)."""
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _write_chunks(working_dir, _sample_chunks())

        _index1, stats1 = asyncio.run(
            get_or_build_index(
                str(working_dir), str(cache_dir), _fake_embed, embed_model="model-a"
            )
        )
        assert stats1.cache_hit is False
        assert stats1.embed_model == "model-a"

        _index2, stats2 = asyncio.run(
            get_or_build_index(
                str(working_dir), str(cache_dir), _fake_embed, embed_model="model-b"
            )
        )
        assert stats2.cache_hit is False, "different embed_model label must force a rebuild"
        assert stats2.embed_model == "model-b"

    def test_save_and_load_round_trip_directly(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _write_chunks(working_dir, _sample_chunks())

        index, _ = asyncio.run(build_index(str(working_dir), _fake_embed, embed_model="fake-2d"))
        save_index(index, str(cache_dir))

        loaded = load_index(str(cache_dir))
        assert loaded is not None
        assert loaded.chunk_ids == index.chunk_ids
        assert loaded.embed_model == "fake-2d"
        assert loaded.source_mtime == index.source_mtime
        np.testing.assert_allclose(np.asarray(loaded.vectors), index.vectors)

    def test_load_index_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert load_index(str(tmp_path / "no-such-cache")) is None
