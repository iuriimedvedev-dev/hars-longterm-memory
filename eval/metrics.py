"""Retrieval-quality metrics for tools/memory/eval/ab_bench.py.

Pure functions operating on a ranked list of document identifiers (here:
`file_path` strings — see retrieval_queries.yaml's module docstring for why
the scoring unit is doc-level, not chunk-level) against a gold-relevant set.
No I/O, no LLM, no network — these are the ONLY functions this eval suite
trusts for a "did retrieval get the right document" verdict.

WHY doc-level (file_path), not chunk-level (chunk_id): the two retrieval
paths this harness compares return different granularities of chunk
identity for the same document — the dense/BM25/fusion channel (channels.py)
returns LightRAG's internal chunk `_id`, while LightRAG's own mode-based
context assembly (naive/local/global/hybrid) only exposes `reference_id` ->
`file_path` in its rendered context, not the internal chunk id at all (see
channels.py's `_parse_reference_document_list`). Scoring on file_path is the
only unit both paths can be judged on commensurably, and it also matches
what actually matters to a caller of memory_recall: which SOURCE DOCUMENT
grounded the answer, not which 512-token slice of it.

A bug in this module invalidates every future retrieval decision made from
ab_bench.py's output silently — see
tools/memory/tests/test_eval_metrics.py for hand-computed regression cases
covering every function here.
"""

from __future__ import annotations

import math


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Collapse a ranked list to first-occurrence-only, preserving rank order.

    Needed before scoring: a channel can return multiple chunks from the same
    document at different chunk-level ranks (e.g. dense + BM25 both hit
    chunk-000 and chunk-003 of the same doc). Without deduping, that single
    document would count as satisfying "top-k" at more positions than it
    actually occupies, inflating Recall@k/NDCG@k for large k and gaming
    NDCG's rank-1 bonus if a repeat happens to land first.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def recall_at_k(ranked_docs: list[str], gold_docs: set[str], k: int) -> float:
    """Fraction of `gold_docs` present anywhere in the top-`k` of `ranked_docs`.

    For a single-gold-document query this is the standard binary hit@k (0.0
    or 1.0). For a multi-gold query (e.g. a multihop question needing 2
    documents) it is the fraction found, generalizing hit@k correctly rather
    than silently only checking the first gold doc.
    """
    if not gold_docs:
        raise ValueError("recall_at_k requires a non-empty gold_docs set")
    if k < 1:
        raise ValueError("k must be >= 1")
    top_k = set(ranked_docs[:k])
    return len(top_k & gold_docs) / len(gold_docs)


def ndcg_at_k(ranked_docs: list[str], gold_docs: set[str], k: int) -> float:
    """Binary-relevance NDCG@k: rewards gold documents ranked EARLIER, not
    merely present, unlike recall_at_k.

    IDCG is computed against the ideal ranking of exactly `min(len(gold_docs), k)`
    relevant documents at the top — the correct ideal for a query with more
    than one gold document, not the single-relevant-document formula.
    """
    if not gold_docs:
        raise ValueError("ndcg_at_k requires a non-empty gold_docs set")
    if k < 1:
        raise ValueError("k must be >= 1")
    dcg = 0.0
    for i, doc_id in enumerate(ranked_docs[:k], start=1):
        if doc_id in gold_docs:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold_docs), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def first_rank(ranked_docs: list[str], targets: set[str]) -> int | None:
    """1-indexed rank of the first document in `ranked_docs` that is in
    `targets`, or None if none of `targets` appear at all.
    """
    for i, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in targets:
            return i
    return None


def reciprocal_rank(ranked_docs: list[str], gold_docs: set[str]) -> float:
    """1/rank of the first gold document found anywhere in `ranked_docs`
    (unbounded by any k — MRR is conventionally computed over the full
    returned list), or 0.0 if none of `gold_docs` appear at all.
    """
    if not gold_docs:
        raise ValueError("reciprocal_rank requires a non-empty gold_docs set")
    rank = first_rank(ranked_docs, gold_docs)
    return 0.0 if rank is None else 1.0 / rank


def mean_reciprocal_rank(per_query_reciprocal_ranks: list[float]) -> float:
    if not per_query_reciprocal_ranks:
        return 0.0
    return sum(per_query_reciprocal_ranks) / len(per_query_reciprocal_ranks)


def supersession_violated(
    ranked_docs: list[str], correct_docs: set[str], superseded_docs: set[str]
) -> bool:
    """True iff a superseded document outranks (or is present while every
    correct document is entirely absent from) the ranked list.

    Precise rule, matching the task definition "fraction of supersession
    queries where a superseded doc outranks its replacement":
    - If no superseded document appears anywhere in `ranked_docs` at all, it
      cannot outrank anything -> not violated (superseded docs merely being
      absent is fine; the failure mode is a superseded doc *appearing ahead*
      of the correct one, or a superseded doc appearing while the correct
      doc doesn't show up at all).
    - If a superseded document appears and the correct document does not
      appear anywhere -> violated (the worst case: the caller sees only the
      outdated claim).
    - Otherwise -> violated iff the best (lowest-numbered) rank among
      superseded_docs is strictly better than the best rank among
      correct_docs.
    """
    if not correct_docs:
        raise ValueError("supersession_violated requires a non-empty correct_docs set")
    if not superseded_docs:
        raise ValueError("supersession_violated requires a non-empty superseded_docs set")
    superseded_rank = first_rank(ranked_docs, superseded_docs)
    if superseded_rank is None:
        return False
    correct_rank = first_rank(ranked_docs, correct_docs)
    if correct_rank is None:
        return True
    return superseded_rank < correct_rank


def supersession_error_rate(violations: list[bool]) -> float:
    if not violations:
        return 0.0
    return sum(1 for v in violations if v) / len(violations)


def no_answer_hit_rate(returned_any_hits: list[bool]) -> float:
    """Fraction of no-answer queries where the channel returned >=1 hit at all.

    Retrieval-layer false-confidence proxy: a query with no good answer in
    the corpus should ideally return nothing (or the caller should be able
    to tell there's no real signal), not silently top-k-pad on best-effort
    noise that then gets treated as grounding by the synthesising LLM. A
    HIGH no_answer_hit_rate does not automatically mean retrieval is broken
    (top-k vector search always returns *something* unless a channel applies
    its own hard threshold) — it means the channel offers the caller no
    signal to distinguish "weak but real" from "nothing relevant exists",
    which is exactly the failure mode this metric exists to surface.
    """
    if not returned_any_hits:
        return 0.0
    return sum(1 for r in returned_any_hits if r) / len(returned_any_hits)


__all__ = [
    "dedupe_preserve_order",
    "recall_at_k",
    "ndcg_at_k",
    "first_rank",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "supersession_violated",
    "supersession_error_rate",
    "no_answer_hit_rate",
]
