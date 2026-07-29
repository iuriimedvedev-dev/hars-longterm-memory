"""Tuned convex-combination fusion of a dense channel and the BM25 sparse channel.

WHY tuned convex combination (`score = alpha*dense + (1-alpha)*sparse`), not RRF:

- CORE-Bench (arXiv:2606.11864): dense embedding retrieval drops 71.7 -> 20.3
  NDCG@10 moving to repo-level/identifier-bearing retrieval; the paper names
  BM25 as the fix for exact identifiers, APIs, filenames, and config keys.
- LIMIT (arXiv:2605.29384): BM25 Recall@20 = 0.949 vs a dense retriever's 0.027
  on the same identifier-heavy queries.
- A 2026-04-12 production hybrid-search writeup measured a *tuned* convex
  combination giving +7.5% NDCG over dense-only, vs only +1.3% for plain
  Reciprocal Rank Fusion (RRF) on the same corpus, citing Bruch et al. 2022:
  tuning a single alpha on a small labeled set (~40 pairs) beats RRF, which
  discards score magnitude entirely and only uses rank position.

RRF is attractive because it needs no score normalization, but that is also
its weakness here: it throws away exactly the magnitude information that lets
a channel "win" decisively when it is confident (e.g. BM25 finding an exact
`A2S32` match) rather than being diluted by rank alone.

WHY min-max normalization (not z-score, not a fixed global scale):

Dense (cosine similarity, roughly bounded) and BM25 (Lucene-variant term-
weighted sum, unbounded, magnitude depends on corpus IDF and query length)
live on incomparable scales that also drift with the ACTUAL candidate pool
returned for a given query (a query matching many strong hits produces a
different score distribution than one matching few weak ones). Min-max
rescales each channel's *own returned candidates* to [0, 1] per query, with no
corpus-wide calibration constant to keep in sync as the corpus changes. This
is the standard choice in production hybrid search (e.g. Weaviate's
relativeScoreFusion, Azure AI Search's normalized-score fusion) for exactly
this reason. Z-score was considered and rejected: it is sensitive to the
(here, often small — a handful of hits) candidate-pool size and produces
unbounded output that would need re-clamping before combination anyway.

Degenerate case (all scores in a channel tied, including the single-hit case):
min-max's `(v - lo) / (hi - lo)` divides by zero. Treated as "no discriminative
signal from this channel for this query" -> normalized to a neutral 0.5,
rather than raising or defaulting to 0/1 (either of which would arbitrarily
favor or penalize a channel that legitimately found exactly one candidate).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from tools.memory.retrieval.supersession import apply_supersession_scoring

# Guards the supersession-aware re-scoring pass (retrieval/supersession.py)
# behind an explicit opt-in, following the HARS_MEMORY_* env convention used
# elsewhere in this package (e.g. HARS_MEMORY_HYBRID_ALPHA). Default OFF: the
# mechanism must be measured against the eval harness
# (tools/memory/eval/ab_bench.py `ab`) before it changes shipped ranking
# behaviour for any caller of `fuse()` — see retrieval/supersession.py's
# module docstring for what it does and why each of its two signals is
# scoped the way it is.
HARS_MEMORY_SUPERSESSION_SCORING_ENV = "HARS_MEMORY_SUPERSESSION_SCORING"


_TRUTHY = {"1", "true", "yes"}


def _supersession_scoring_enabled() -> bool:
    return os.environ.get(HARS_MEMORY_SUPERSESSION_SCORING_ENV, "0").strip().lower() in _TRUTHY


# Per-signal overrides for retrieval/supersession.py's two independent
# signals, measured in isolation against the eval harness
# (tools/memory/eval/ab_bench.py `ab`, 46-query labeled set,
# tools/memory/eval/retrieval_queries.yaml). Summary of the isolated runs
# (reproducible across PYTHONHASHSEED=0/7/42):
#
#   marker_penalty alone:  supersession_error_rate 0.333 -> 0.167, EVERY other
#                           metric unchanged or slightly improved (identical
#                           recall@1 across 3 different PYTHONHASHSEED runs).
#   recency_discount alone: supersession_error_rate UNCHANGED at 0.333 (zero
#                           queries fixed), identifier recall@1 regresses
#                           0.602 -> 0.560 reproducibly (not tie-order noise
#                           — confirmed deterministic across 3 seeds):  it
#                           demotes the exactly-correct OLDER doc for two
#                           identifier lookups (id04, id05) below a newer but
#                           merely topically-adjacent doc — precisely the
#                           "old-but-valid" failure mode called out as a risk
#                           for any recency prior on this corpus.
#
# Net: recency_discount defaults OFF even when the master flag is on (real,
# reproducible cost; zero measured benefit on this corpus's labeled set).
# marker_penalty defaults ON — it is the one signal that measured a clean win
# with no offsetting regression. Both remain independently overridable via
# their own env var for future re-measurement as the corpus's marker
# convention (name: DEPRECATED -, **THIS MEMORY IS DEPRECATED.**) gets used
# more, or if a smarter, topically-conditioned recency signal replaces this
# one — see retrieval/supersession.py's module docstring "not attempted" note.
HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV = "HARS_MEMORY_SUPERSESSION_MARKER_PENALTY"
HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV = "HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT"


def _sub_flag_enabled(env_name: str, *, default: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY

# UNTUNED DEFAULT. Bruch et al. 2022 (and the production writeup cited above)
# tune alpha on a labeled query/relevance set; no such set exists yet for this
# corpus. 0.5 is a neutral midpoint, not a measured optimum. Override via
# HARS_MEMORY_HYBRID_ALPHA once a labeled eval set justifies a different value —
# see tools/memory/eval/gold_questions.yaml as the natural home for that set.
DEFAULT_HYBRID_ALPHA = 0.5


@dataclass(frozen=True)
class ChannelHit:
    """One channel's raw hit for a chunk (dense OR sparse)."""

    score: float
    content: str
    file_path: str


@dataclass(frozen=True)
class FusedChunk:
    chunk_id: str
    fused_score: float
    dense_score: float | None  # raw (pre-normalization); None if dense had no hit
    sparse_score: float | None  # raw; None if sparse had no hit
    dense_norm: float  # 0.0 if dense had no hit for this chunk
    sparse_norm: float  # 0.0 if sparse had no hit for this chunk
    content: str
    file_path: str


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        # Tied (or single) candidate — see module docstring.
        return {chunk_id: 0.5 for chunk_id in scores}
    return {chunk_id: (v - lo) / (hi - lo) for chunk_id, v in scores.items()}


def fuse(
    dense_hits: dict[str, ChannelHit],
    sparse_hits: dict[str, ChannelHit],
    alpha: float,
) -> list[FusedChunk]:
    """Fuse dense and sparse per-chunk hits into one ranked list.

    `alpha=1.0` reduces to pure dense ranking (chunks with no dense hit get
    `dense_norm=0.0` and rank at the bottom); `alpha=0.0` reduces to pure
    sparse ranking, symmetrically.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

    dense_norm = _min_max_normalize({cid: hit.score for cid, hit in dense_hits.items()})
    sparse_norm = _min_max_normalize({cid: hit.score for cid, hit in sparse_hits.items()})

    all_chunk_ids = set(dense_hits) | set(sparse_hits)
    fused: list[FusedChunk] = []
    for chunk_id in all_chunk_ids:
        dense_hit = dense_hits.get(chunk_id)
        sparse_hit = sparse_hits.get(chunk_id)
        d_norm = dense_norm.get(chunk_id, 0.0)
        s_norm = sparse_norm.get(chunk_id, 0.0)
        source_hit = dense_hit or sparse_hit
        assert source_hit is not None  # chunk_id came from one of the two dicts
        fused.append(
            FusedChunk(
                chunk_id=chunk_id,
                fused_score=alpha * d_norm + (1.0 - alpha) * s_norm,
                dense_score=dense_hit.score if dense_hit else None,
                sparse_score=sparse_hit.score if sparse_hit else None,
                dense_norm=d_norm,
                sparse_norm=s_norm,
                content=source_hit.content,
                file_path=source_hit.file_path,
            )
        )
    fused.sort(key=lambda c: c.fused_score, reverse=True)

    if _supersession_scoring_enabled():
        fused = apply_supersession_scoring(
            fused,
            enable_marker_penalty=_sub_flag_enabled(
                HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV, default=True
            ),
            enable_recency_discount=_sub_flag_enabled(
                HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV, default=False
            ),
        )

    return fused


__all__ = [
    "DEFAULT_HYBRID_ALPHA",
    "HARS_MEMORY_SUPERSESSION_SCORING_ENV",
    "HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV",
    "HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV",
    "ChannelHit",
    "FusedChunk",
    "fuse",
]
