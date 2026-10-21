"""Apply the learned channel-fusion ranker to a candidate pool (fail-soft).

This is the ONE function the recall path calls. It replaces the cross-encoder
rerank stage on the default path: instead of a transformer forward pass per
candidate it is a LightGBM/MLP evaluation over already-computed channel scores,
so it costs microseconds per candidate on CPU.

Contract (deliberately identical to code-diver's H-91a ranker):

* No artifact / unloadable artifact / any scoring error -> the pool is returned
  UNCHANGED (truncated to ``top_k``), so enabling the ranker can never break a
  working search.
* The pool is the FULL pre-truncation candidate list (``fuse()`` output), so the
  ranker can improve recall@k, not just reorder inside an already-truncated
  window.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from hars_memory.ranking.artifact import LtrRankerModel
from hars_memory.ranking.features import (
    LtrCorpusContext,
    LtrFeatureRow,
    extract_pool_features,
    load_cached_context,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hars_memory.retrieval.fusion import FusedChunk

logger = logging.getLogger(__name__)

HARS_MEMORY_LTR_RANKER_ENV = "HARS_MEMORY_LTR_RANKER"

# Shipped artifact (small, committed). The DEFAULT must be the RETRAINED
# md_struct artifact, not the older synthetic-query one: a plain deployment
# that sets no env vars must serve the model the retrieval pipeline is tuned
# for (alpha=0.15 on cleaned md_struct chunks). Before this change the default
# was `ranker.json` (trained on the 223-query synthetic set, alpha from the
# pre-mdstruct corpus), so every deployment without an explicit
# HARS_MEMORY_LTR_RANKER silently served the WRONG model. The older artifact
# is still shipped next to it for measurement (`ranker.json`).
#
# Override with HARS_MEMORY_LTR_RANKER=<manifest path>, or set it to `0` to
# disable the ranker entirely (see _resolve_manifest_path).
_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent / "artifacts" / "ranker.mdstruct.json"
)

_cache: dict[str, LtrRankerModel | None] = {}


def _resolve_manifest_path(explicit: str | Path | None) -> Path | None:
    raw: str | Path | None = explicit
    if raw is None:
        raw = os.environ.get(HARS_MEMORY_LTR_RANKER_ENV)
    if raw is None:
        return _DEFAULT_MANIFEST
    text = str(raw).strip()
    if text == "" or text.lower() in {"0", "off", "false", "no", "none", "disabled"}:
        return None
    return Path(text)


def load_default_ranker(explicit: str | Path | None = None) -> LtrRankerModel | None:
    """Load (and memoize) the configured ranker, or ``None`` when disabled/absent.

    ``explicit`` overrides ``HARS_MEMORY_LTR_RANKER``; the env value ``0``
    disables the ranker entirely.
    """
    path = _resolve_manifest_path(explicit)
    if path is None:
        return None
    key = str(path)
    if key in _cache:
        return _cache[key]
    model = LtrRankerModel.load(path)
    _cache[key] = model
    return model


def clear_ranker_cache() -> None:
    """Drop the memoized models (tests / after retraining)."""
    _cache.clear()


def ltr_ranker_enabled(explicit: str | Path | None = None) -> bool:
    """True when a ranker manifest is configured AND loads successfully."""
    return load_default_ranker(explicit) is not None


def _resolve_context(context: LtrCorpusContext | None, *, index_dir: str | Path | None) -> LtrCorpusContext | None:
    if context is not None:
        return context
    base = index_dir
    if base is None:
        base = os.environ.get("HARS_MEMORY_INDEX_DIR")
    if not base:
        return None
    return load_cached_context(base)


def apply_ltr_ranker(
    fused: Sequence[FusedChunk],
    query: str,
    *,
    top_k: int | None = None,
    model: LtrRankerModel | None = None,
    model_path: str | Path | None = None,
    context: LtrCorpusContext | None = None,
    index_dir: str | Path | None = None,
    return_rows: bool = False,
) -> list[FusedChunk] | tuple[list[FusedChunk], list[LtrFeatureRow]]:
    """Reorder ``fused`` by the learned ranker's score, best first.

    Returns the (optionally ``top_k``-truncated) reordered pool. When no ranker
    is available -- or scoring raises -- returns ``fused[:top_k]`` unchanged so
    the caller's existing hand-tuned order is preserved exactly.
    """
    limit = top_k if top_k is not None else len(fused)
    fallback = list(fused[:limit])
    if not fused:
        return (fallback, []) if return_rows else fallback

    ranker = model if model is not None else load_default_ranker(model_path)
    if ranker is None:
        return (fallback, []) if return_rows else fallback

    corpus_context = _resolve_context(context, index_dir=index_dir)
    rows = extract_pool_features(list(fused), query, context=corpus_context)
    scores = ranker.score(rows)
    if scores is None:
        return (fallback, []) if return_rows else fallback

    order = sorted(
        range(len(fused)),
        key=lambda i: (-scores[i], fused[i].chunk_id),
    )
    ranked = [fused[i] for i in order][:limit]

    # Rewrite `fused_score` with the model's score so the ranker's order is
    # AUTHORITATIVE for every downstream stage. Without this, the ranker only
    # reorders the list, and a later stage that re-sorts on `-fused_score`
    # (mcp_server._select_diverse_fused_chunks -> diversity.select_diverse_
    # fused_chunks, which does `sorted(fused, key=-fused_score)`) silently
    # RESTORES the pre-ranker order -- measured on the live MCP path: the
    # ranker changed c1..c5 -> c1,c3,c5,c2,c4 and the diversity pass reverted
    # it to c1..c5. Scoring the reorder into `fused_score` is the same contract
    # `select_diverse_fused_chunks` itself uses ("the returned list is ordered
    # by S_div and carries the adjusted score on fused_score, so order and
    # reported score agree").
    #
    # The model score is an unbounded LightGBM/LambdaRank value, so it is
    # min-max mapped onto the pool's own fused range: rank-preserving, keeps
    # `fused_score` on the scale the diversity penalty and the LightRAG score
    # band are calibrated for, and leaves a degenerate (all-equal) score
    # vector untouched so ties keep `fuse()`'s own tie-break.
    ranked = _rescore_with_model(ranked, scores, order, fused)

    if return_rows:
        rows_by_id = {row.item_id: row for row in rows}
        return ranked, [rows_by_id[chunk.chunk_id] for chunk in ranked if chunk.chunk_id in rows_by_id]
    return ranked


def _rescore_with_model(
    ranked: list[FusedChunk],
    scores: list[float],
    order: list[int],
    fused: Sequence[FusedChunk],
) -> list[FusedChunk]:
    """Return ``ranked`` with ``fused_score`` replaced by a rank-preserving
    min-max mapping of the model scores onto the pool's own fused range.

    See the call site for why the score must be written (not just the order).
    A single-element or degenerate score vector is returned unchanged.
    """
    from dataclasses import replace

    model_scores = [scores[i] for i in order][: len(ranked)]
    if not model_scores:
        return ranked
    lo = min(model_scores)
    hi = max(model_scores)
    pool_scores = [c.fused_score for c in fused]
    pool_lo = min(pool_scores)
    pool_hi = max(pool_scores)
    span = hi - lo
    if span <= 0.0:
        # Every candidate scored identically: keep the pool's own scores so the
        # downstream sort falls back to fuse()'s tie-break rather than a flat
        # constant that would erase it.
        return ranked
    out: list[FusedChunk] = []
    for chunk, score in zip(ranked, model_scores, strict=True):
        mapped = pool_lo + (score - lo) / span * (pool_hi - pool_lo)
        out.append(replace(chunk, fused_score=mapped))
    return out


__all__ = [
    "HARS_MEMORY_LTR_RANKER_ENV",
    "apply_ltr_ranker",
    "clear_ranker_cache",
    "load_default_ranker",
    "ltr_ranker_enabled",
]
