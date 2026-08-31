#!/usr/bin/env python3
"""MCP server for HARS long-term memory — query/admin surface over the knowledge graph.

Matches hars-control house style:
- try/except ImportError for mcp.server with graceful stub fallback
- json_text() / schema() / tool() helpers
- All tools return list[TextContent]
- sync main() (console-script entrypoint) wraps async _serve() + stdio_server()

Tool catalogue:
  memory_recall       — primary workhorse (local/global/hybrid/naive)
  memory_remember     — save a knowledge note into staging
  memory_entities     — 1-hop neighbourhood lookup (INTROSPECTION, unstable)
  memory_related      — N-hop subgraph for a given entity (INTROSPECTION, unstable)
  memory_status       — index health + staleness (GPU-FREE)
  memory_consolidate  — trigger incremental ingest (admin)
  memory_forget       — purge stale documents by date, with keyword protection (admin)

Query tools work against an existing index using CPU embeddings only.
Only memory_consolidate and memory_forget(apply=True) touch the durable index.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError:
    @dataclass
    class TextContent:  # type: ignore[no-redef]
        type: str
        text: str

    @dataclass
    class Tool:  # type: ignore[no-redef]
        name: str
        description: str
        inputSchema: dict[str, Any]

    class Server:  # type: ignore[no-redef]
        def __init__(self, _name: str) -> None:
            self.name = _name

        def list_tools(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return lambda fn: fn

        def call_tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return lambda fn: fn

        def create_initialization_options(self) -> dict[str, Any]:
            return {}

        async def run(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("The 'mcp' package is required to run this server")

    def stdio_server() -> Any:  # type: ignore[misc]
        raise RuntimeError("The 'mcp' package is required to run this server")


from hars_memory.server.embedder import qdrant_collection_names
from hars_memory.server.legacy_env_guard import refuse_if_legacy_env
from hars_memory.server.logging_setup import log_query_event, log_write_event, setup_logging

setup_logging()
logger = logging.getLogger("hars-longterm-memory-mcp")

# Permanent fail-closed guard: refuse to start against a legacy env prefix
# (e.g. Cortex's pre-2026-07-29 GRAPHRAG_* rename) instead of silently
# falling back to HARS_MEMORY_* defaults. Configured via
# HARS_MEMORY_LEGACY_ENV_PREFIXES (comma-separated); empty/unset = no-op.
# See tools/memory/server/legacy_env_guard.py.
refuse_if_legacy_env()

# ---------------------------------------------------------------------------
# Config (all from env — no hardcoded values)
# ---------------------------------------------------------------------------


def _require_env(name: str) -> str:
    """Read a required env var, or raise a clear, actionable error.

    Used for config that has no meaningful machine-agnostic default (a path
    into this operator's own filesystem) — silently falling back to a
    specific machine's path (e.g. `/tmp/hars_memory_lightrag`,
    `/mnt/datasets/graphrag/staging`) is exactly the "no machine-specific
    defaults" violation this helper exists to prevent. See
    docs/superpowers/specs/2026-08-25-hars-longterm-memory-standalone-extraction-design.md.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required and has no default — this package does not assume any "
            "particular machine's filesystem layout. Set it explicitly (see config/.env.example)."
        )
    return value


HARS_MEMORY_INDEX_DIR = _require_env("HARS_MEMORY_INDEX_DIR")
HARS_MEMORY_VECTOR_STORAGE = os.environ.get("HARS_MEMORY_VECTOR_STORAGE", "NanoVectorDBStorage")
HARS_MEMORY_GRAPH_STORAGE = os.environ.get("HARS_MEMORY_GRAPH_STORAGE", "NetworkXStorage")
HARS_MEMORY_QDRANT_URL = os.environ.get("HARS_MEMORY_QDRANT_URL", "http://localhost:6335")
HARS_MEMORY_QDRANT_COLLECTION = os.environ.get("HARS_MEMORY_QDRANT_COLLECTION", "hars_longterm_memory")
# Required (no meaningful default) when the Qdrant vector backend is in use —
# left as an empty-string sentinel here rather than raised at import time so
# that non-Qdrant test/runtime configurations (the default backend is
# NanoVectorDBStorage) never pay for it. qdrant_collection_names() itself
# raises ValueError on an empty prefix; the memory_status Qdrant branch below
# already wraps that call in a try/except and surfaces it as vector_info.error,
# so an unset prefix fails closed with an actionable message instead of
# silently falling back to the unprefixed (collision-prone) collection names.
HARS_MEMORY_QDRANT_COLLECTION_PREFIX = os.environ.get("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "")
HARS_MEMORY_EMBED_MODEL = os.environ.get("HARS_MEMORY_EMBED_MODEL", "unsloth/embeddinggemma-300m")
HF_HOME = os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home")
HARS_API_BASE_URL = os.environ.get("HARS_API_BASE_URL", "http://localhost:8765")
HARS_MEMORY_EXTRACTOR_BASE_URL = os.environ.get("HARS_MEMORY_EXTRACTOR_BASE_URL", "http://localhost:8080/v1")
HARS_MEMORY_EXTRACTOR_MODEL = os.environ.get("HARS_MEMORY_EXTRACTOR_MODEL", "Qwen3.6-27B-Q4_K_M")
HARS_MEMORY_QUERY_BASE_URL = os.environ.get("HARS_MEMORY_QUERY_BASE_URL", "http://localhost:8081/v1")
HARS_MEMORY_QUERY_MODEL = os.environ.get("HARS_MEMORY_QUERY_MODEL", "Qwen3.5-4B-Q4_K_M")
# Shared with memory_remember (write path) and _staging_backlog_info (memory_status
# read path) — a single source of truth for where staged notes accumulate before
# the next update_kb.sh run merges them into the graph.
HARS_MEMORY_STAGING_DIR = _require_env("HARS_MEMORY_STAGING_DIR")

# --- Hybrid (dense + BM25 sparse) retrieval config — see retrieval/ package ---
# Default derived from HARS_MEMORY_INDEX_DIR (never a separate machine-global
# default, and never inside HARS_MEMORY_INDEX_DIR itself — that path is
# read-only production storage): mirrors HARS_MEMORY_FLAT_DENSE_CACHE_DIR's
# own derivation immediately below (a SIBLING directory of
# HARS_MEMORY_INDEX_DIR). Losing this cache only costs seconds to rebuild
# (unlike the flat-dense cache below), so it does not need its own separate
# required env var — deriving it keeps "no machine-specific defaults" true
# without adding a second knob an operator has to remember to set.
HARS_MEMORY_BM25_CACHE_DIR = os.environ.get(
    "HARS_MEMORY_BM25_CACHE_DIR", str(Path(HARS_MEMORY_INDEX_DIR).parent / "bm25_cache")
)
# Flat dense channel (retrieval/flat_index.py) cache dir — same
# derive-from-HARS_MEMORY_INDEX_DIR pattern as HARS_MEMORY_BM25_CACHE_DIR
# above, for the same "no separate machine-global default" reason. See
# flat_index.py's own module docstring ("WHY cache_dir is NOT defaulted to
# /tmp here"): losing the BM25 cache costs seconds to rebuild; losing this
# one costs up to 68 minutes (a full re-embed of every chunk) — which is why
# this one is a SIBLING directory of HARS_MEMORY_INDEX_DIR itself (never
# inside it — that path is read-only production storage) rather than /tmp,
# so it lives on the same persistent volume as the source
# `kv_store_text_chunks.json` it caches and survives a reboot.
HARS_MEMORY_FLAT_DENSE_CACHE_DIR = os.environ.get(
    "HARS_MEMORY_FLAT_DENSE_CACHE_DIR", str(Path(HARS_MEMORY_INDEX_DIR).parent / "flat_dense_cache")
)
# MEASURED, not a guess: tools/memory/eval/ab_bench.py `alpha-sweep` over
# 0.0-1.0 (step 0.1) on the 46-query tools/memory/eval/retrieval_queries.yaml
# set (hybrid_bm25 channel, top_k=10) found alpha=0.5 IS the empirical ndcg@10
# optimum on this corpus: 0.7216 (alpha=0.5) vs 0.6741 (alpha=0.0, pure BM25)
# vs 0.6163 (alpha=1.0, pure dense); sensitivity spread (max-min across the
# sweep) = 0.1053. Measured 2026-07-29. retrieval/fusion.py's own
# DEFAULT_HYBRID_ALPHA docstring predates this measurement and still reads
# "UNTUNED" — that file is out of scope for this change (owned by a
# concurrent supersession-scoring task); this env default is the one this
# server actually uses, and it is now a measured value, not a placeholder.
HARS_MEMORY_HYBRID_ALPHA = float(os.environ.get("HARS_MEMORY_HYBRID_ALPHA", "0.5"))
# Escape hatch: hybrid fields are additive (see memory_recall docstring) and
# fail soft on their own, but this lets an operator disable the sparse channel
# entirely (e.g. bm25s unavailable in some environment) without patching code.
HARS_MEMORY_HYBRID_ENABLED = os.environ.get("HARS_MEMORY_HYBRID_ENABLED", "1").strip().lower() not in (
    "0", "false", "no", "off",
)

# ---------------------------------------------------------------------------
# Tunable constants (no magic values — see per-item justification in comments)
# ---------------------------------------------------------------------------
GRAPH_FILE_NAME = "graph_chunk_entity_relation.graphml"
# The GraphML files LightRAG writes declare xmlns="http://graphml.graphdrawing.org/xmlns",
# NOT the .../graphml URI. Using the wrong URI silently makes every element lookup
# return None (node_count/edge_count stay None even though the file is well-formed).
GRAPHML_XMLNS = "http://graphml.graphdrawing.org/xmlns"

STALE_INDEX_WARNING_DAYS = 7  # queries against an index older than this get a staleness_warning

# LightRAG's own defaults are top_k=40 (entities) / chunk_top_k=20 (chunks); this MCP
# server feeds one shared `top_k` value into both, so 12 was throttling both budgets.
# `top_k` is the RESULT-COUNT knob (see FETCH-WIDTH KNOB below for the breadth knob):
# it governs LightRAG's own entity/relation QueryParam.top_k, the final truncation
# length of every ranked list this file returns (_merge_context_with_fusion's `limit`,
# _compute_hybrid_block's `fused[:top_k]` slice, citations/entities_used limit), and
# the tool-schema default result count.
# "Refs saturate around 17" CONFIRMED by the fetch-width sweep below (see
# HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER's comment): fetching wider than
# ~17-20 chunks stops helping and starts costing identifier-query precision
# on this corpus — 20 is not an arbitrary round number above that floor,
# it is (empirically) already sitting almost exactly on the saturation
# point, which is why decoupling fetch width from result count measures as
# a no-op at this shipped default (see that comment for the full sweep).
DEFAULT_QUERY_TOP_K = 20

# --- FETCH-WIDTH KNOB (decoupled from DEFAULT_QUERY_TOP_K above, 2026-08-01) ---
#
# CONFIRMED by reading the installed lightrag-hku==1.4.16 source directly
# (tools/memory/.venv/lib/python3.13/site-packages/lightrag/{utils,operate}.py):
# QueryParam.chunk_top_k does THREE jobs at once, all with the SAME number:
#   1. operate.py::_get_vector_context — the initial chunk vector-search fetch
#      size: `search_top_k = query_param.chunk_top_k or query_param.top_k`.
#   2. utils.py::process_chunks_unified step 1 — the reranker's `top_n`:
#      `rerank_top_k = query_param.chunk_top_k or len(unique_chunks)` (moot
#      today: the reranker is OFF, see server/reranker.py — it measurably
#      worsens supersession_error_rate 0.333->0.500).
#   3. utils.py::process_chunks_unified step 3 — the FINAL truncation of that
#      same chunk list: `if len(unique_chunks) > query_param.chunk_top_k:
#      unique_chunks = unique_chunks[:query_param.chunk_top_k]`.
# Before this change, memory_recall passed the SAME `top_k` value as BOTH
# QueryParam.top_k (entity/relation search — untouched by this change, stays
# `top_k`) and QueryParam.chunk_top_k (all three jobs above) — so "how wide
# the initial cosine cut is" and "how many chunks the caller gets back" were
# forced to the same number. That starves _compute_hybrid_block's fusion
# pool (`HYBRID_CANDIDATE_POOL_MULTIPLIER`), the supersession rescoring, and
# the z-score tie-break of any candidate the initial top_k-wide cosine cut
# had already dropped, since _merge_context_with_fusion's round-robin merge
# can only choose among what LightRAG's own chunk_top_k-bounded `context`
# block already contains.
#
# `fetch_top_k` (memory_recall arg) / HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER
# (env) decouple this: QueryParam.chunk_top_k and _compute_hybrid_block's
# pool_size base are driven by the (optionally wider) fetch width; `top_k`
# keeps its existing job as the result-count knob everywhere above.
#
# Multiplier default MEASURED (2026-08-01, env -i controlled, PYTHONHASHSEED
# 0 vs 42 bit-identical) on the 46-query retrieval_queries.yaml set,
# context_priority=merged (shipped default), via a scratch harness built
# ONLY on tools/memory/eval/ab_bench.py's own primitives
# (load_queries/score_query/aggregate_scores) driving THIS file's real
# call_tool("memory_recall", ...) end to end — same pattern the
# context_priority=merged measurement above used. Two regimes, deliberately
# swept separately because they give OPPOSITE answers:
#
#   top_k=10 (the eval harness's own historical convention — NOT the
#   shipped tool default): widening genuinely helps. fetch_top_k=17-20
#   (multiplier ~1.7-2.0) is a robust plateau — recall@1 0.5602->0.5880,
#   ndcg@10 0.7249->0.7356-0.7371, mrr 0.7169->0.7308-0.7324, with ZERO
#   regression on any query type (conceptual improves — one query flips
#   miss->hit and stays flipped from width 15 onward; identifier/multihop/
#   supersession/no_answer are byte-identical to baseline through width 20).
#   Widening further (>=30, i.e. >=3x) starts trading part of that gain
#   away and visibly costs `identifier` queries (recall@1 0.75->0.70,
#   mrr 0.8333->0.7833 by width 30, still down at width 80) for a
#   recall@10-only gain (0.8241->0.8380 at width>=22) nothing else shares.
#
#   top_k=20 (DEFAULT_QUERY_TOP_K — the ACTUAL shipped default): widening
#   does NOT help at all. The unwidened baseline (fetch_top_k=20) already
#   scores 0.5880/0.8241/0.7376/0.7349 (recall@1/recall@10/ndcg@10/mrr) —
#   i.e. numerically the SAME as the top_k=10 sweep's *widened* plateau,
#   because DEFAULT_QUERY_TOP_K=20 already fetches past the corpus's
#   saturation point on its own. Every width tried beyond 20 (30/34/40/
#   60/80) only ever DECREASES recall@1/ndcg@10/mrr (monotonically,
#   driven entirely by `identifier` degrading: recall@1 0.75->0.70,
#   mrr 0.8333->0.775 by width 40) with no compensating gain anywhere —
#   recall@10/multihop/supersession/no_answer_hit_rate stay flat at every
#   width tested. This DIRECTLY CONFIRMS this file's older "Refs saturate
#   around 17" note on DEFAULT_QUERY_TOP_K above (refined: ~17-20, not
#   contradicted) — and explains why decoupling the knobs changes nothing
#   for a default-top_k caller: today's shipped top_k=20 already sits at
#   that saturation point, so the pre-2026-08-01 coupling cost nothing in
#   practice for the common case.
#
# Verdict: 1.0 (no widening) is the correct DEFAULT — it is not merely a
# conservative fallback, it is what the top_k=20 measurement above actually
# recommends. The knob is still real and useful OPT-IN capability for a
# caller that deliberately passes a NARROW top_k (e.g. top_k=10 for a
# smaller response payload) and wants close to top_k=20-equivalent
# retrieval depth without getting 20 results back — set fetch_top_k
# explicitly (~1.7-2.0x top_k) for that case; see the fetch_top_k
# tool-schema description below. Latency cost of widening is real but
# small and roughly linear in pool_size: at top_k=20 (uncached, realistic
# per-query CPU-embed cost included), dense_channel went 50.3ms (width 20)
# -> 52.0ms (width 40) -> 57.7ms (width 80); wall time 164.8ms -> 167.0ms
# -> 175.3ms. Index mtimes were confirmed unchanged before/after this
# entire measurement (read-only, context_only=True path; no LLM, no GPU).
HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER_ENV = "HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER"
HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER = float(
    os.environ.get(HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER_ENV, "1.0")
)

NAIVE_FALLBACK_MODE = "naive"
# max_tokens used to be interpolated into this prose but never enforced (100 and 4096
# produced byte-identical output — see item 9 audit); removed from the schema, this
# fixed instruction is kept since it does shape LLM answer-mode output.
LLM_RESPONSE_TYPE = "Single paragraph."

# 2 hops measured at 160 KB, 3 hops at 2.1 MB (64,841 lines) — unusable for any caller's
# context budget. hops>1 is not supported; the induced subgraph explodes combinatorially.
SUBGRAPH_MAX_HOPS = 1
SUBGRAPH_NODE_BUDGET = 150  # hard cap so even 1 hop on a hub node can't explode the response
SUBGRAPH_EDGE_BUDGET = 300
SUBGRAPH_DESCRIPTION_MAX_CHARS = 300

ENTITY_DESCRIPTION_MAX_CHARS = 400

# Entity/relation context-budget optimum for QueryParam.max_entity_tokens /
# max_relation_tokens — measured via `tools/memory/eval/ab_bench.py
# token-budget-sweep` across a 2-D grid (entity x relation, both axes swept
# 0-8000 then refined 0-1500 x 2000-5500) on the 46-query
# tools/memory/eval/retrieval_queries.yaml set, mode='hybrid' (LightRAG
# 'mix'), top_k=10. This QueryParam kwarg was previously OMITTED entirely, so
# LightRAG's own un-set default (6000 entity / 8000 relation) applied. That
# default actually UNDER-performs a much smaller budget: entity=500/
# relation=4500 is a robust plateau (identical recall@10/ndcg@10 across
# entity in [450,600] and relation in [2500,5000] — not a single lucky grid
# point) that beats the default (ndcg@10 0.608->0.662, recall@10
# 0.676->0.769, mrr 0.622->0.659) AND beats an intermediate 3000/4000 point
# that looked good on ndcg@10 alone (0.639) but regressed
# supersession_error_rate 0.333->0.5; 500/4500 keeps supersession_error_rate
# at the default's 0.333 while still winning ndcg@10/recall@10/mrr outright.
# recall@1 is unaffected by any budget (0.4907 throughout) — the budget only
# governs how many chunks fit under LightRAG's shared max_total_tokens
# ceiling before entity/relation text crowds them out, which is exactly what
# recall@10 tracks.
#
# NOTE ON UNITS: these numbers were tuned on 2026-07-29 while the installed
# tokenizer was `_ByteTokenizer` — i.e. they were UTF-8 BYTE budgets (~1 byte
# per ASCII char in this corpus' English text). Since the switch to the
# tiktoken-backed tokenizer (server/lightrag_init.py::resolve_tokenizer) the
# very same numbers are interpreted by LightRAG as real MODEL TOKENS, so in
# terms of actual text they are now ~4x LARGER (500 tokens ≈ 2000 chars of
# entity text). The *names* keep the _BYTES suffix on purpose: they document
# the unit the tuning was performed in. Values are deliberately left as-is
# and are to be re-tuned against the reindexed corpus — the whole
# entity/relation-vs-chunk trade-off (how much chunk text survives under the
# shared max_total_tokens ceiling) shifts once chunks themselves are token-
# sized rather than byte-sized.
DEFAULT_MAX_ENTITY_CONTEXT_BYTES = 500
DEFAULT_MAX_RELATION_CONTEXT_BYTES = 4500
MAX_CHUNKS_PER_SOURCE: int = 3

# Low-confidence marker threshold on the raw (pre-fusion,
# pre-min-max-normalization) dense score. A low score is only marked when the
# other available retrieval channels provide no corroborating evidence.
NO_ANSWER_DENSE_SCORE_THRESHOLD = 0.28

# Generic question words add noise to live-worktree searches; keep this list
# local because ripgrep is an additive, MCP-specific retrieval channel.
RIPGREP_STOPWORDS = frozenset({
    "what", "how", "when", "where", "which", "does", "procedure", "team",
    "describe", "explain", "show", "list", "about", "standard", "used", "from",
    "with", "for",
})

# memory_recall `context_priority` values — see _merge_context_with_fusion.
#
# Default flipped to CONTEXT_PRIORITY_MERGED on 2026-07-30. Measured on the
# 46-query tools/memory/eval/retrieval_queries.yaml labeled set, current
# index, top_k=10:
#
#   variant                                              recall@1  recall@10  ndcg@10  mrr    supersession_err
#   pre-flip shipped default (lightrag, supersession off)  0.477     0.727      0.637   0.634   0.333
#   merged, supersession off                                0.5324    0.8241     0.7126  0.7003  0.3333
#   merged + supersession on (this flip)                    0.5324    0.8241     0.7151  0.7030  0.1667
#
# Per-query-type breakdown was byte-identical between the last two rows
# except conceptual ndcg@10 (0.6679->0.6711, up) — merged beats the old
# default on every metric and every query type, no regressions. Added
# latency ~0.04ms.
#
# Escape hatch: set HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT=lightrag to revert
# every caller that omits `context_priority` back to the pre-flip behaviour
# without patching call sites — see DEFAULT_CONTEXT_PRIORITY below. A caller
# can also always override per-call by passing context_priority= explicitly,
# regardless of this default (unaffected by the env var either way).
CONTEXT_PRIORITY_LIGHTRAG = "lightrag"  # today's unmodified context field
CONTEXT_PRIORITY_MERGED = "merged"  # round-robin(fusion, lightrag) reorder+inject

_context_priority_default_raw = os.environ.get(
    "HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT", CONTEXT_PRIORITY_MERGED
)
if _context_priority_default_raw not in (CONTEXT_PRIORITY_LIGHTRAG, CONTEXT_PRIORITY_MERGED):
    raise ValueError(
        f"HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT={_context_priority_default_raw!r} must be one of "
        f"{CONTEXT_PRIORITY_LIGHTRAG!r} or {CONTEXT_PRIORITY_MERGED!r}"
    )
DEFAULT_CONTEXT_PRIORITY = _context_priority_default_raw

# Each hybrid channel (dense, sparse) retrieves top_k * this many candidates
# BEFORE fusion — the fused/ranked pool must be wider than the tool's declared
# top_k so fusion (and a downstream reranker, not implemented here) has real
# candidates to choose from instead of already-truncated per-channel top-N.
HYBRID_CANDIDATE_POOL_MULTIPLIER = 3
HYBRID_IDENTIFIER_LOOKUP_LIMIT = 10  # cap on the explicit exact-identifier surfacing list
HYBRID_SNIPPET_MAX_CHARS = 300  # matches ENTITY snippet truncation elsewhere (_extract_citations)

# Context post-processing (memory_recall, context_only path) — see _postprocess_context().
CONTEXT_MIN_CHUNK_CHARS = 200  # headerless orphan fragments below this length are dropped
CONTEXT_DOCUMENT_HEADER_RE = re.compile(r"^\[Document:")
CONTEXT_DESCRIPTION_SEP = "<SEP>"  # matches LightRAG's internal GRAPH_FIELD_SEP

# Known duplicate-entity spellings in the current index (imperfect extraction produced
# separate nodes for the same real-world thing). This only widens search-time recall by
# querying every alias and merging results — it does NOT merge the underlying graph
# nodes. True node merging requires reindexation.
ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "r9700": ("R9700", "R9700 32GB", "AMD R9700 32GB", "AMD Radeon AI Pro R9700 32GB"),
    "zero_vea": ("zero_vea", "Zero_vea", "Zero VEA"),
    "real_vea": ("real_vea", "vea_real"),
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Server("hars-longterm-memory")

# Lazy-initialised LightRAG instance (only for actual queries, not status).
_rag_instance: object | None = None
_rag_lock = asyncio.Lock()

# Serialises the instance-level `rag.llm_model_func` swap in memory_recall's
# synthesis path (LightRAG 1.5.6 has no QueryParam.model_func, so the shared
# instance attribute is the only lever). Without this, two concurrent recalls
# race: the first one's `finally` restores the ORIGINAL extractor func while
# the second is still mid-query, so the second silently synthesises with the
# extractor model — and whichever finishes last can leave the query model
# permanently installed on the shared instance.
_QUERY_MODEL_LOCK = asyncio.Lock()


def json_text(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, indent=2, sort_keys=True, default=str))]


def schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


def tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> Tool:
    return Tool(name=name, description=description, inputSchema=schema(properties, required))


# ---------------------------------------------------------------------------
# Index status helpers (GPU-free)
# ---------------------------------------------------------------------------


def _graph_file_path() -> Path:
    return Path(HARS_MEMORY_INDEX_DIR) / GRAPH_FILE_NAME


def _staleness_info(graph_file: Path | None = None) -> tuple[str | None, int | None]:
    """Return (last_ingest ISO8601 string, age in whole days) for the GraphML index.

    Returns (None, None) when the graph file does not exist yet. Shared by
    _index_status() and the memory_recall staleness fields so age-of-index is
    computed identically everywhere.
    """
    graph_file = graph_file if graph_file is not None else _graph_file_path()
    if not graph_file.exists():
        return None, None
    mtime = datetime.datetime.fromtimestamp(graph_file.stat().st_mtime)
    stale_days = (datetime.datetime.now() - mtime).days
    return mtime.isoformat(), stale_days


def _index_status() -> dict[str, Any]:
    """Return index health without loading LightRAG or calling an LLM."""
    working_dir = Path(HARS_MEMORY_INDEX_DIR)
    index_exists = working_dir.exists() and any(working_dir.iterdir()) if working_dir.exists() else False

    # Try to read node/edge counts from LightRAG's graph file if present.
    node_count: int | None = None
    edge_count: int | None = None

    graph_file = _graph_file_path()
    last_ingest, _stale_days = _staleness_info(graph_file)
    if graph_file.exists():
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(graph_file)
            root = tree.getroot()
            ns = {"g": GRAPHML_XMLNS}
            graph_elem = root.find("g:graph", ns)
            if graph_elem is None:
                graph_elem = root.find("graph")
            if graph_elem is not None:
                node_count = len(graph_elem.findall(f"{{{GRAPHML_XMLNS}}}node"))
                edge_count = len(graph_elem.findall(f"{{{GRAPHML_XMLNS}}}edge"))
        except Exception as exc:
            logger.debug("Could not parse graph file: %s", exc)

    vector_info: dict[str, Any] = {
        "backend": HARS_MEMORY_VECTOR_STORAGE,
    }
    if HARS_MEMORY_VECTOR_STORAGE == "NanoVectorDBStorage":
        vector_files = sorted(working_dir.glob("vdb_*.json")) if working_dir.exists() else []
        vector_info["files"] = [path.name for path in vector_files]
        vector_info["file_count"] = len(vector_files)
    elif "qdrant" in HARS_MEMORY_VECTOR_STORAGE.lower():
        # HARS_MEMORY_QDRANT_COLLECTION is NOT a Qdrant collection name — LightRAG's
        # QdrantVectorDBStorage exports it as QDRANT_WORKSPACE and uses it as the
        # tenant id written into every payload's workspace_id (see
        # lightrag_init.py). The real collection base names are fixed by
        # qdrant_impl.py::__post_init__ (no model_suffix at this call site) to
        # exactly lightrag_vdb_{chunks,entities,relationships}, namespaced per
        # project by HARS_MEMORY_QDRANT_COLLECTION_PREFIX (qdrant_collection_names)
        # — enumerate those three and filter counts by the workspace tenant.
        if not HARS_MEMORY_QDRANT_COLLECTION_PREFIX:
            # Required-config error, NOT a connectivity problem — do not fold
            # this into the generic except below (which sets reachable=False
            # and would mislead an operator into chasing a phantom network
            # issue). Signal it distinctly: reachable=None ("unknown, could
            # not even attempt the check") plus an explicit config_error.
            vector_info.update(
                {
                    "reachable": None,
                    "config_error": "HARS_MEMORY_QDRANT_COLLECTION_PREFIX is not set",
                    "workspace": HARS_MEMORY_QDRANT_COLLECTION,
                }
            )
        else:
            try:
                from qdrant_client import QdrantClient  # type: ignore[import-not-found]
                from qdrant_client import models as _qmodels  # type: ignore[import-not-found]

                qc = QdrantClient(url=HARS_MEMORY_QDRANT_URL, timeout=3)
                collections: dict[str, Any] = {}
                total_workspace_points = 0
                dims: set[int] = set()
                collection_names = qdrant_collection_names(HARS_MEMORY_QDRANT_COLLECTION_PREFIX)
                for ns, name in zip(("chunks", "entities", "relationships"), collection_names):
                    if not qc.collection_exists(name):
                        collections[ns] = {"collection": name, "exists": False}
                        continue
                    info = qc.get_collection(name)
                    dim = int(info.config.params.vectors.size)
                    dims.add(dim)
                    workspace_count = qc.count(
                        collection_name=name,
                        count_filter=_qmodels.Filter(
                            must=[
                                _qmodels.FieldCondition(
                                    key="workspace_id",
                                    match=_qmodels.MatchValue(value=HARS_MEMORY_QDRANT_COLLECTION),
                                )
                            ]
                        ),
                        exact=True,
                    ).count
                    total_workspace_points += workspace_count
                    collections[ns] = {
                        "collection": name,
                        "exists": True,
                        "points_count_workspace": workspace_count,
                        "points_count_total": info.points_count,
                        "dim": dim,
                        "distance": str(info.config.params.vectors.distance),
                    }
                vector_info.update(
                    {
                        "workspace": HARS_MEMORY_QDRANT_COLLECTION,
                        "collections": collections,
                        "points_count": total_workspace_points,
                        "dim": (dims.pop() if len(dims) == 1 else sorted(dims)) if dims else None,
                        "reachable": True,
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Qdrant collection lookup failed for workspace %s @ %s (vector_info.error "
                    "only, not fatal to memory_status): %s",
                    HARS_MEMORY_QDRANT_COLLECTION, HARS_MEMORY_QDRANT_URL, exc,
                )
                vector_info.update(
                    {"error": str(exc), "reachable": False, "workspace": HARS_MEMORY_QDRANT_COLLECTION}
                )

    return {
        "index_exists": index_exists,
        "working_dir": str(working_dir),
        "node_count": node_count,
        "edge_count": edge_count,
        "last_ingest": last_ingest,
        "storage": {
            "vector": vector_info,
            "graph": {"backend": HARS_MEMORY_GRAPH_STORAGE},
        },
        "configured_models": {
            "extractor": f"{HARS_MEMORY_EXTRACTOR_MODEL} @ {HARS_MEMORY_EXTRACTOR_BASE_URL}",
            "query": f"{HARS_MEMORY_QUERY_MODEL} @ {HARS_MEMORY_QUERY_BASE_URL}",
            "embedder": f"{HARS_MEMORY_EMBED_MODEL} (CPU)",
        },
        "staging_backlog": _staging_backlog_info(),
        "message": (
            "Index ready."
            if index_exists
            else "Index empty / not yet built. Run: python -m hars_memory.server.index --paths .plans docs"
        ),
    }


def _staging_backlog_info() -> dict[str, Any]:
    """Report how many notes are waiting in HARS_MEMORY_STAGING_DIR and the
    age of the oldest one, since nobody would otherwise notice a growing
    backlog without listing the directory by hand (memory_remember writes
    here; only the next `update_kb.sh` run merges these into the graph).
    """
    staging = Path(HARS_MEMORY_STAGING_DIR)
    if not staging.exists():
        return {"pending_notes": 0, "oldest_note": None, "oldest_note_age_days": None, "staging_dir": str(staging)}

    notes = sorted(staging.glob("*.md"), key=lambda p: p.stat().st_mtime)
    if not notes:
        return {"pending_notes": 0, "oldest_note": None, "oldest_note_age_days": None, "staging_dir": str(staging)}

    oldest = notes[0]
    oldest_mtime = datetime.datetime.fromtimestamp(oldest.stat().st_mtime)
    age_days = (datetime.datetime.now() - oldest_mtime).days
    return {
        "pending_notes": len(notes),
        "oldest_note": oldest.name,
        "oldest_note_age_days": age_days,
        "staging_dir": str(staging),
    }


async def _get_rag() -> object:
    """Lazily initialise the LightRAG instance (thread-safe)."""
    global _rag_instance
    async with _rag_lock:
        if _rag_instance is None:
            from hars_memory.server.lightrag_init import create_lightrag
            _rag_instance = create_lightrag()
            await _rag_instance.initialize_storages()  # type: ignore[attr-defined]
        return _rag_instance


def _lightrag_mode(mode: str) -> str:
    """Map the MCP-facing API mode to the installed LightRAG mode."""
    if mode == "hybrid":
        return "mix"
    return mode


def _resolve_query_mode(rag_mode: str, ll_keywords: list[str], hl_keywords: list[str]) -> tuple[str, str | None]:
    """Return (effective_lightrag_mode, mode_fallback_reason_or_None).

    local/hybrid/global modes hard-fail with no keywords at all. naive mode
    needs none and empirically has the best signal-to-noise in that situation,
    so fall back to it instead of returning a hard failure.
    """
    if ll_keywords or hl_keywords:
        return rag_mode, None
    return (
        NAIVE_FALLBACK_MODE,
        "no keywords supplied — used naive mode; supply ll_keywords for graph-aware retrieval",
    )


def _resolve_fetch_top_k(top_k: int, fetch_top_k_arg: Any) -> tuple[int | None, str]:
    """Resolve the fetch-width knob (see FETCH-WIDTH KNOB comment above
    DEFAULT_QUERY_TOP_K) for one memory_recall call.

    On success returns (effective_fetch_top_k, source) where source is
    "explicit" (the caller passed fetch_top_k directly) or "multiplier"
    (derived from `round(top_k * HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER)`,
    the env-configurable default — 1.0 out of the box, i.e. IDENTICAL to the
    pre-2026-08-01 `chunk_top_k=top_k` coupling, unless an operator opts in
    via the env var or a caller passes fetch_top_k explicitly).

    On failure (fetch_top_k_arg explicitly narrower than top_k — you cannot
    return more results than you fetched) returns (None, error_message):
    a caller contract violation is surfaced as an error, not silently
    clamped, so a caller bug doesn't masquerade as a quietly-narrowed fetch.
    """
    if fetch_top_k_arg is not None:
        fetch_top_k = int(fetch_top_k_arg)
        if fetch_top_k < top_k:
            return None, (
                f"fetch_top_k ({fetch_top_k}) must be >= top_k ({top_k}) — "
                "you cannot return more results than you fetch."
            )
        return fetch_top_k, "explicit"
    computed = max(top_k, int(round(top_k * HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER)))
    return computed, "multiplier"


# ---------------------------------------------------------------------------
# Cached GraphML graph (search_entities / get_subgraph, GPU-free)
# ---------------------------------------------------------------------------
# The graph file is ~32 MB / 25k nodes; parsing it is synchronous and takes
# ~0.4s. Both tools used to re-parse it on every call. Cache it in a module-
# level singleton keyed on file mtime, and keep parsing off the event loop.
_graph_cache: object | None = None
_graph_cache_mtime: float | None = None
_graph_lock = asyncio.Lock()


async def _get_graph() -> tuple[Any, str]:
    """Return (cached NetworkX graph, cache_status), reloading if the GraphML
    file changed. cache_status is "hit" or "rebuild" — surfaced to callers
    (currently the usage-event log, see logging_setup.log_query_event) so
    cache effectiveness is observable instead of silently discarded.

    Raises FileNotFoundError if the index has not been built yet.
    """
    global _graph_cache, _graph_cache_mtime
    graph_file = _graph_file_path()
    if not graph_file.exists():
        raise FileNotFoundError(f"Index not built yet: {graph_file}")

    mtime = graph_file.stat().st_mtime
    async with _graph_lock:
        cache_status = "hit"
        if _graph_cache is None or _graph_cache_mtime != mtime:
            import networkx as nx  # type: ignore[import-not-found]
            logger.info("Loading GraphML index into cache: %s", graph_file)
            _graph_cache = await asyncio.to_thread(nx.read_graphml, str(graph_file))
            _graph_cache_mtime = mtime
            cache_status = "rebuild"
        return _graph_cache, cache_status


# ---------------------------------------------------------------------------
# Cached BM25 sparse index (retrieval/bm25_index.py) — same mtime-cache shape
# as _get_graph() above, keyed on kv_store_text_chunks.json instead of the
# GraphML file. GPU-free, no LLM, no re-embedding — see retrieval/__init__.py.
# ---------------------------------------------------------------------------
_bm25_cache: Any = None
_bm25_cache_source_mtime: float | None = None
_bm25_lock = asyncio.Lock()


async def _get_bm25_index() -> tuple[Any, str]:
    """Return (cached BM25SparseIndex, cache_status), rebuilding if the source
    chunk store changed. cache_status is "hit" or "rebuild" — surfaced to
    callers (currently the usage-event log, see
    logging_setup.log_query_event) so cache effectiveness is observable
    instead of silently discarded.

    Raises hars_memory.retrieval.bm25_index.BM25IndexUnavailableError if the
    LightRAG index (and therefore kv_store_text_chunks.json) has not been built yet.
    """
    global _bm25_cache, _bm25_cache_source_mtime
    from hars_memory.retrieval.bm25_index import (
        CHUNKS_FILENAME,
        BM25IndexUnavailableError,
        get_or_build_index,
    )

    chunks_file = Path(HARS_MEMORY_INDEX_DIR) / CHUNKS_FILENAME
    if not chunks_file.exists():
        raise BM25IndexUnavailableError(f"No text-chunk store at {chunks_file}")

    mtime = chunks_file.stat().st_mtime
    async with _bm25_lock:
        cache_status = "hit"
        if _bm25_cache is None or _bm25_cache_source_mtime != mtime:
            logger.info("Loading/building BM25 sparse index (source mtime changed or first use)")
            _bm25_cache, _stats = await asyncio.to_thread(
                get_or_build_index, HARS_MEMORY_INDEX_DIR, HARS_MEMORY_BM25_CACHE_DIR
            )
            _bm25_cache_source_mtime = mtime
            cache_status = "rebuild"
        return _bm25_cache, cache_status


# ---------------------------------------------------------------------------
# Cached flat dense index (retrieval/flat_index.py) — same mtime-cache shape
# as _get_bm25_index() above, but ALSO on-disk persisted (see flat_index.py's
# save_index/load_index): the in-memory singleton below only saves repeated
# rebuild/reload work WITHIN one running server process; HARS_MEMORY_FLAT_
# DENSE_CACHE_DIR is what protects the ~68-minute full-embed cost ACROSS
# process restarts. gated behind HARS_MEMORY_FLAT_CHANNEL (default off — see
# retrieval/fusion.py's HARS_MEMORY_FLAT_CHANNEL_ENV comment) — this function
# is never called at all unless that gate is on, so an operator who never
# enables the channel pays zero cost for its existence (no model load, no
# disk read).
# ---------------------------------------------------------------------------
_flat_dense_cache: Any = None
_flat_dense_cache_source_mtime: float | None = None
_flat_dense_lock = asyncio.Lock()


async def _get_flat_dense_index(rag: object) -> tuple[Any, str]:
    """Return (cached FlatDenseIndex, cache_status), incrementally updating
    if the source chunk store changed since the last call (see
    retrieval/flat_index.py::get_or_build_index — incremental, not full
    rebuild, whenever a usable same-embed_model cache already exists).
    cache_status is "hit" or "rebuild" (rebuild covers BOTH the incremental
    and full-rebuild paths — see FlatBuildStats.update_path for the finer
    distinction, surfaced separately in the hybrid.flat_dense report block).

    `rag.embedding_func` (an `EmbeddingFunc` instance whose `__call__` is the
    exact `embed_func(texts, context=...)` shape retrieval/flat_index.py
    expects) is passed straight through — see flat_index.py's own module
    docstring ("WHY the same embedder.py doc/query asymmetric convention")
    for why reusing THIS callable, not a second model, keeps the flat
    channel's vectors in the same embedding space as the production dense
    channel automatically.

    Raises hars_memory.retrieval.flat_index.FlatIndexUnavailableError if
    the LightRAG index (and therefore kv_store_text_chunks.json) has not
    been built yet.
    """
    global _flat_dense_cache, _flat_dense_cache_source_mtime
    from hars_memory.retrieval.flat_index import CHUNKS_FILENAME, get_or_build_index

    chunks_file = Path(HARS_MEMORY_INDEX_DIR) / CHUNKS_FILENAME
    mtime = chunks_file.stat().st_mtime
    async with _flat_dense_lock:
        cache_status = "hit"
        if _flat_dense_cache is None or _flat_dense_cache_source_mtime != mtime:
            logger.info("Loading/updating flat dense index (source mtime changed or first use)")
            _flat_dense_cache, build_stats = await get_or_build_index(
                HARS_MEMORY_INDEX_DIR,
                HARS_MEMORY_FLAT_DENSE_CACHE_DIR,
                rag.embedding_func,  # type: ignore[attr-defined]
                embed_model=HARS_MEMORY_EMBED_MODEL,
            )
            _flat_dense_cache_source_mtime = mtime
            cache_status = "rebuild"
            logger.info(
                "Flat dense index ready: %d chunks (update_path=%s, %.3fs)",
                build_stats.chunk_count, build_stats.update_path, build_stats.build_seconds,
            )
        return _flat_dense_cache, cache_status


def _select_diverse_fused_chunks(
    fused: list[Any], limit: int, *, max_per_source: int = MAX_CHUNKS_PER_SOURCE
) -> list[Any]:
    """Select the highest-scoring chunks while bounding per-file repetition."""
    if limit <= 0:
        return []
    ranked = sorted(fused, key=lambda chunk: (-chunk.fused_score, chunk.chunk_id))
    selected: list[Any] = []
    deferred: list[Any] = []
    source_counts: dict[str, int] = {}
    for chunk in ranked:
        source = chunk.file_path
        if len(selected) < limit and source_counts.get(source, 0) < max_per_source:
            selected.append(chunk)
            source_counts[source] = source_counts.get(source, 0) + 1
        else:
            deferred.append(chunk)
    if len(selected) < limit:
        # Backfill capped sources only after the greedy diverse pass.
        selected.extend(deferred[: limit - len(selected)])
    selected.sort(key=lambda chunk: (-chunk.fused_score, chunk.chunk_id))
    return selected[:limit]


def _ripgrep_term_is_specific(term: str) -> bool:
    """Return whether a ripgrep term matches a high-specificity shape."""
    has_upper = any(char.isupper() for char in term)
    has_lower = any(char.islower() for char in term)
    return (
        (has_upper and has_lower)
        or "_" in term
        or any(char.isdigit() for char in term)
        or "/" in term
        or "." in term
        or (term.isupper() and len(term) >= 2)
        or len(term) > 4
    )


def _ripgrep_keywords(ll_keywords: list[str] | None) -> list[str]:
    """Filter and specificity-order caller-supplied ripgrep terms."""
    terms = [term.strip() for term in (ll_keywords or []) if term.strip()]
    terms = [term for term in terms if term.casefold() not in RIPGREP_STOPWORDS]
    # Python's sort is stable, so terms retain input order within each tier.
    return sorted(terms, key=lambda term: not _ripgrep_term_is_specific(term))


async def _compute_hybrid_block(
    rag: object,
    question: str,
    top_k: int,
    ll_keywords: list[str] | None = None,
    *,
    fetch_top_k: int | None = None,
) -> dict[str, Any]:
    """Return the additive `hybrid` field for memory_recall: BM25 sparse hits,
    dense/sparse fused ranking, and an explicit exact-identifier lookup.

    `fetch_top_k` (see the FETCH-WIDTH KNOB comment above DEFAULT_QUERY_TOP_K)
    is the candidate-pool BASE — `pool_size = max(fetch_top_k *
    HYBRID_CANDIDATE_POOL_MULTIPLIER, fetch_top_k, top_k)` — while `top_k`
    remains the RESULT count this function slices `fused_chunks` down to.
    Defaults to `top_k` when omitted (any future direct caller of this
    function that doesn't pass fetch_top_k gets byte-identical pool_size to
    before this parameter existed).

    `ll_keywords` (caller-supplied, e.g. `['A2S32', 'qdrant_transplant']`) is
    forwarded to the ripgrep channel (`ripgrep_channel.search`) alongside
    `question` — identifier-shaped entries feed `rg` exactly like an
    identifier mentioned in the question text itself would. Fixed
    2026-07-30: this parameter previously existed nowhere in this function's
    signature, so `ripgrep_channel.extract_terms()`'s own `ll_keywords`
    handling (already correct — see `test_ripgrep_channel.py::TestExtractTerms`)
    was unreachable from the MCP server; the memory_recall tool's own
    docstring recommends supplying `ll_keywords` for exactly this identifier
    case, so the documented usage pattern was the one that silently got the
    worst behaviour (0 hits) until this fix. Defaults to `None` (treated as
    empty) so existing question-only callers/tests are unaffected.

    Fails SOFT by design (returns `{"enabled": False, ...}` instead of raising):
    this is an additive capability layered onto the existing context/answer
    response, and its own failure (e.g. bm25s not installed, index not built
    yet) must never break the primary memory_recall contract that predates it.
    """
    if not HARS_MEMORY_HYBRID_ENABLED:
        return {"enabled": False, "reason": "HARS_MEMORY_HYBRID_ENABLED=0"}

    from hars_memory.retrieval.bm25_index import BM25IndexUnavailableError

    try:
        from hars_memory.retrieval.fusion import (
            HARS_MEMORY_FLAT_CHANNEL_ENV,
            HARS_MEMORY_RIPGREP_CHANNEL_ENV,
            ChannelHit,
            apply_flat_dense_gate,
            apply_ripgrep_gate,
            flat_channel_enabled,
            fuse,
            ripgrep_channel_enabled,
        )
        from hars_memory.retrieval.tokenizer import extract_identifier_terms

        effective_fetch_top_k = fetch_top_k if fetch_top_k is not None else top_k
        pool_size = max(
            effective_fetch_top_k * HYBRID_CANDIDATE_POOL_MULTIPLIER, effective_fetch_top_k, top_k
        )

        bm25_index, bm25_cache_status = await _get_bm25_index()
        sparse_start = time.monotonic()
        sparse_search_hits = await bm25_index.asearch(question, pool_size)
        sparse_elapsed_ms = (time.monotonic() - sparse_start) * 1000

        dense_start = time.monotonic()
        dense_raw = await rag.chunks_vdb.query(question, top_k=pool_size)  # type: ignore[attr-defined]
        dense_elapsed_ms = (time.monotonic() - dense_start) * 1000

        sparse_hits = {
            hit.chunk_id: ChannelHit(score=hit.score, content=hit.content, file_path=hit.file_path)
            for hit in sparse_search_hits
        }
        dense_hits = {
            str(entry["id"]): ChannelHit(
                score=float(entry.get("distance", 0.0)),
                content=str(entry.get("content", "")),
                file_path=str(entry.get("file_path", "")),
            )
            for entry in dense_raw
        }

        # Full pool, NOT sliced to top_k yet — apply_ripgrep_gate (if the
        # channel is enabled) needs the whole pool so an off-index file it
        # finds can be APPENDED beyond top_k rather than requiring it to
        # already be in the first top_k dense/sparse slots (which, by
        # definition of "off-index", it never is). Sliced to top_k right
        # before being returned, whichever path runs.
        fused_pool = fuse(dense_hits, sparse_hits, HARS_MEMORY_HYBRID_ALPHA)

        # Ripgrep channel (retrieval/ripgrep_channel.py) — additive gate over
        # the dense+sparse fusion above, NOT a third convex weight; see
        # retrieval/fusion.py's design note above `apply_ripgrep_gate` for
        # the measured reasoning. Gated behind HARS_MEMORY_RIPGREP_CHANNEL
        # (default ON, injection-only — see fusion.py's
        # HARS_MEMORY_RIPGREP_CHANNEL_ENV comment for the measurement behind
        # that default). Fails soft exactly like the
        # rest of this function: any exception here is caught by the
        # surrounding try/except and degrades to hybrid.enabled=False for
        # the WHOLE block, matching the existing all-or-nothing failure
        # contract of this function (ripgrep is not given its own separate
        # failure path since it shares this function's one try/except).
        ripgrep_report: dict[str, Any] = {"enabled": False, "reason": f"{HARS_MEMORY_RIPGREP_CHANNEL_ENV}=0 (default)"}
        if ripgrep_channel_enabled():
            from hars_memory.retrieval import ripgrep_channel

            rg_roots = ripgrep_channel.default_roots()
            rg_result = await ripgrep_channel.asearch(
                question,
                ll_keywords=_ripgrep_keywords(ll_keywords),
                roots=rg_roots,
                top_k=pool_size,
            )
            if rg_result.available and rg_result.hits:
                # Keyed by basename, not chunk_id/file_stable_id — see
                # retrieval/fusion.py's design note for why that is the only
                # join key that actually holds across this corpus's channels.
                ripgrep_hits_by_basename = {
                    Path(hit.file_path).name: ChannelHit(
                        score=hit.score, content=hit.content, file_path=hit.file_path
                    )
                    for hit in rg_result.hits
                }
                fused = apply_ripgrep_gate(fused_pool, ripgrep_hits_by_basename, top_k=top_k)
            else:
                fused = fused_pool[:top_k]
            ripgrep_report = {
                "enabled": True,
                "available": rg_result.available,
                "unavailable_reason": rg_result.unavailable_reason,
                "query_terms": list(rg_result.query_terms),
                "hits_count": len(rg_result.hits),
                "injected_count": sum(1 for c in fused if c.chunk_id.startswith("ripgrep:")),
                "latency_ms": round(rg_result.latency_seconds * 1000, 2),
            }
        else:
            fused = fused_pool[:top_k]

        # Flat dense channel (retrieval/flat_index.py) — additive coverage
        # gate over the SAME dense+sparse fusion pool, run independently of
        # the ripgrep gate above (both consult `fused_pool`, the full
        # pre-truncation pool, so a chunk_id already visible to dense/sparse
        # anywhere in that pool is never re-injected by this channel — see
        # retrieval/fusion.py's design note above `apply_flat_dense_gate`).
        # Composed by APPENDING this channel's own exclusive tail onto
        # whatever `fused` already is (post-ripgrep): the two gates cannot
        # collide on chunk_id — flat's ids are real LightRAG chunk_ids,
        # ripgrep's injected entries are always prefixed `ripgrep:` — so
        # this is a safe union, not a second independent truncation.
        # Gated behind HARS_MEMORY_FLAT_CHANNEL (default OFF — see fusion.py's
        # HARS_MEMORY_FLAT_CHANNEL_ENV comment for the measured reasoning: a
        # real, zero-regression capability with no current live trigger on
        # this index, since kv_store_text_chunks.json and the vector store
        # are presently in exact 1:1 correspondence). Fails soft exactly
        # like the ripgrep block above (shares this function's one
        # try/except).
        flat_report: dict[str, Any] = {
            "enabled": False, "reason": f"{HARS_MEMORY_FLAT_CHANNEL_ENV}=0 (default)"
        }
        if flat_channel_enabled():
            flat_start = time.monotonic()
            flat_index, flat_cache_status = await _get_flat_dense_index(rag)
            flat_search_hits = await flat_index.search(question, rag.embedding_func, pool_size)  # type: ignore[attr-defined]
            flat_elapsed_ms = (time.monotonic() - flat_start) * 1000

            flat_hits_by_chunk_id = {
                hit.chunk_id: ChannelHit(score=hit.score, content=hit.content, file_path=hit.file_path)
                for hit in flat_search_hits
            }
            gated = apply_flat_dense_gate(fused_pool, flat_hits_by_chunk_id, top_k=top_k)
            # Extract just the injected tail by CHUNK_ID membership against
            # `fused_pool`, not a `[top_k:]` positional slice: apply_flat_
            # dense_gate's own "top" portion is `fused_pool[:top_k]`, which is
            # SHORTER than `top_k` whenever the pool itself has fewer than
            # `top_k` candidates (e.g. a small/sparse corpus) — slicing
            # `gated` at a fixed `top_k` offset would then silently drop the
            # injected entries entirely. Every injected chunk is, by
            # apply_flat_dense_gate's own "exclusive" construction,
            # guaranteed to have a chunk_id absent from `fused_pool` — so
            # membership against that set is correct regardless of pool size.
            fused_pool_ids = {c.chunk_id for c in fused_pool}
            existing_ids = {c.chunk_id for c in fused}
            flat_injected = [
                c for c in gated
                if c.chunk_id not in fused_pool_ids and c.chunk_id not in existing_ids
            ]
            fused = fused + flat_injected
            flat_report = {
                "enabled": True,
                "cache_status": flat_cache_status,
                "chunks_indexed": flat_index.chunk_count,
                "cache_dir": HARS_MEMORY_FLAT_DENSE_CACHE_DIR,
                "hits_count": len(flat_search_hits),
                "injected_count": len(flat_injected),
                "latency_ms": round(flat_elapsed_ms, 2),
            }

        # Dedicated exact-identifier path: high-precision surfacing, not buried
        # in fusion weights. A BM25 hit only counts as an "identifier match"
        # here if the identifier string is verbatim present in its content.
        identifier_terms = extract_identifier_terms(question)
        identifier_matches: list[dict[str, Any]] = []
        if identifier_terms:
            terms_casefolded = [term.casefold() for term in identifier_terms]
            for hit in sparse_search_hits:
                content_casefolded = hit.content.casefold()
                if any(term in content_casefolded for term in terms_casefolded):
                    identifier_matches.append({
                        "chunk_id": hit.chunk_id,
                        "sparse_score": round(hit.score, 4),
                        "file_path": hit.file_path,
                        "snippet": hit.content[:HYBRID_SNIPPET_MAX_CHARS],
                    })
                if len(identifier_matches) >= HYBRID_IDENTIFIER_LOOKUP_LIMIT:
                    break

        # No-answer confidence marker (item 2) — uses the RAW dense score
        # (pre-fusion, pre-min-max-normalization), not fused_chunks[].fused_score:
        # fuse()'s per-query min-max normalization stretches the best candidate
        # to ~1.0 for almost any query regardless of true relevance, which
        # measurably destroys the no-answer signal (see
        # NO_ANSWER_DENSE_SCORE_THRESHOLD's docstring above). No dense hits at
        # all is treated as maximally low-confidence, not skipped.
        strong_sparse_evidence = any(hit.score > 0.0 for hit in sparse_search_hits)
        ripgrep_evidence = bool(ripgrep_report.get("available") and rg_result.hits) if ripgrep_channel_enabled() else False
        if dense_hits:
            top_dense_score = max(hit.score for hit in dense_hits.values())
            low_confidence = (
                top_dense_score < NO_ANSWER_DENSE_SCORE_THRESHOLD
                and not strong_sparse_evidence
                and not ripgrep_evidence
            )
        else:
            top_dense_score = None
            low_confidence = not strong_sparse_evidence and not ripgrep_evidence

        # Apply diversity after both additive gates so the public result is the
        # actual bounded, source-diverse fused selection.
        fused = _select_diverse_fused_chunks(fused, len(fused))

        # Cross-encoder Reranking if configured (e.g. HARS_MEMORY_RERANK_MODEL)
        rerank_func = getattr(rag, "rerank_model_func", None)
        rerank_elapsed_ms = None
        if rerank_func and fused:
            rerank_start = time.monotonic()
            try:
                pool_to_rerank = fused[:pool_size]
                docs = [c.content for c in pool_to_rerank]
                scored = await rerank_func(question, docs, top_n=top_k)
                rerank_elapsed_ms = (time.monotonic() - rerank_start) * 1000.0
                if scored:
                    reranked_pool: list[Any] = []
                    seen_cids: set[str] = set()
                    for item in scored:
                        idx = int(item["index"])
                        if 0 <= idx < len(pool_to_rerank):
                            candidate = pool_to_rerank[idx]
                            reranked_pool.append(candidate)
                            seen_cids.add(candidate.chunk_id)
                    for candidate in fused:
                        if candidate.chunk_id not in seen_cids:
                            reranked_pool.append(candidate)
                    fused = reranked_pool
            except Exception as exc:
                logger.warning("Reranking candidate pool failed: %s", exc)

        return {
            "enabled": True,
            "alpha": HARS_MEMORY_HYBRID_ALPHA,
            "alpha_status": "MEASURED 2026-07-29 empirical optimum (ndcg@10=0.7216 on the "
                             "46-query retrieval_queries.yaml set) — see HARS_MEMORY_HYBRID_ALPHA "
                             "comment in this file",
            "candidate_pool_size": pool_size,
            "fetch_top_k": effective_fetch_top_k,
            "bm25_index": {
                "chunks_indexed": bm25_index.chunk_count,
                "cache_dir": HARS_MEMORY_BM25_CACHE_DIR,
                "cache_status": bm25_cache_status,
            },
            "latency_ms": {
                "sparse_channel": round(sparse_elapsed_ms, 2),
                "dense_channel": round(dense_elapsed_ms, 2),
                "ripgrep_channel": ripgrep_report.get("latency_ms"),
                "flat_dense_channel": flat_report.get("latency_ms"),
                "rerank_channel": round(rerank_elapsed_ms, 2) if rerank_elapsed_ms is not None else None,
            },
            "ripgrep": ripgrep_report,
            "flat_dense": flat_report,
            "confidence": {
                "top_dense_score": round(top_dense_score, 4) if top_dense_score is not None else None,
                "low_confidence": low_confidence,
                "threshold": NO_ANSWER_DENSE_SCORE_THRESHOLD,
                "note": (
                    "top_dense_score is the best RAW dense cosine similarity across the "
                    "retrieved candidate pool for this question — NOT fused_chunks[].fused_score "
                    "(that value is per-query min-max normalized and is near-1.0 for almost any "
                    "top hit, uninformative for this purpose). low_confidence=true means this "
                    "score falls in the range measured for verified-absent (no_answer) queries "
                    "on this corpus. This is a MARKER, not a filter: results below threshold are "
                    "still returned in full here and in `context`/`answer` — treat them as "
                    "unverified, not as evidence, before citing them to the user."
                ),
            },
            "identifier_query_detected": bool(identifier_terms),
            "identifiers": identifier_terms,
            "identifier_matches": identifier_matches or None,
            "fused_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "fused_score": round(chunk.fused_score, 4),
                    "dense_score": chunk.dense_score,
                    "sparse_score": chunk.sparse_score,
                    "file_path": chunk.file_path,
                    "content": chunk.content,
                    "snippet": chunk.content[:HYBRID_SNIPPET_MAX_CHARS],
                }
                for chunk in fused
            ],
        }
    except BM25IndexUnavailableError as exc:
        return {"enabled": False, "reason": str(exc)}
    except Exception as exc:
        logger.warning("Hybrid retrieval failed (non-fatal, additive field only): %s", exc)
        return {"enabled": False, "error_type": type(exc).__name__, "error": str(exc)}


# ---------------------------------------------------------------------------
# Entity search matching (normalization, token-AND, tiered ranking, aliases)
# ---------------------------------------------------------------------------
_ENTITY_SEPARATOR_RE = re.compile(r"[_\-\s]+")

MATCH_TIER_EXACT_ID = "exact_id"
MATCH_TIER_ID_SUBSTRING = "id_substring"
MATCH_TIER_ID_ALL_TOKENS = "id_all_tokens"
MATCH_TIER_DESCRIPTION = "description"
_MATCH_TIER_RANK = {
    MATCH_TIER_EXACT_ID: 0,
    MATCH_TIER_ID_SUBSTRING: 1,
    MATCH_TIER_ID_ALL_TOKENS: 2,
    MATCH_TIER_DESCRIPTION: 3,
}


def _normalize_entity_text(text: str) -> str:
    """Case-fold and collapse `_`, `-`, and whitespace to a single space.

    e.g. 'Phase_C', 'Phase-C', 'phase c' and 'PHASE  C' all normalize to 'phase c',
    so config-style snake_case queries reach Title Case canonical node ids.
    """
    return _ENTITY_SEPARATOR_RE.sub(" ", text).strip().casefold()


def _entity_match_tier(query_norm: str, query_tokens: list[str], node_id: str, description: str) -> str | None:
    """Classify how a node matches a normalized query, or None if it doesn't.

    Rank order (best to worst): exact normalized id match -> id substring ->
    all query tokens present in id -> all query tokens present in description.
    """
    id_norm = _normalize_entity_text(str(node_id))
    if id_norm == query_norm:
        return MATCH_TIER_EXACT_ID
    if query_norm in id_norm:
        return MATCH_TIER_ID_SUBSTRING
    if query_tokens and all(token in id_norm for token in query_tokens):
        return MATCH_TIER_ID_ALL_TOKENS
    desc_norm = _normalize_entity_text(str(description))
    if query_tokens and all(token in desc_norm for token in query_tokens):
        return MATCH_TIER_DESCRIPTION
    return None


def _expand_query_aliases(entity_name: str) -> list[str]:
    """Expand a search query to every known alias in its duplicate-entity family.

    Search-time only — see ENTITY_ALIASES for why this doesn't merge graph nodes.
    """
    query_norm = _normalize_entity_text(entity_name)
    expanded = {entity_name}
    for aliases in ENTITY_ALIASES.values():
        if query_norm in {_normalize_entity_text(alias) for alias in aliases}:
            expanded.update(aliases)
    return sorted(expanded)


def _edge_relation_label(edge_data: dict[str, Any]) -> str:
    """Relation label for an edge — prefer relation_type, fall back to keywords.

    Shared by memory_entities and memory_related so the same
    edge reports the same label regardless of which tool asked (previously the
    two tools used inverted fallback order and disagreed).
    """
    return str(edge_data.get("relation_type", edge_data.get("keywords", "")))


def _extract_answer(result: dict[str, Any]) -> str:
    llm_response = result.get("llm_response") or {}
    return str(llm_response.get("content") or "")


def _extract_citations(result: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    refs_by_id = {
        str(ref.get("reference_id", "")): ref
        for ref in data.get("references", [])
        if isinstance(ref, dict)
    }
    citations: list[dict[str, Any]] = []
    for chunk in data.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        ref_id = str(chunk.get("reference_id", ""))
        ref = refs_by_id.get(ref_id, {})
        citations.append(
            {
                "node_id": str(chunk.get("chunk_id") or ref_id),
                "source_path": str(chunk.get("file_path") or ref.get("file_path") or ""),
                "snippet": str(chunk.get("content", ""))[:300],
                "score": chunk.get("score") or chunk.get("distance"),
            }
        )
        if len(citations) >= limit:
            break
    return citations


def _extract_entities_used(result: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    entities: list[dict[str, Any]] = []
    for entity in data.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entities.append(
            {
                "id": str(entity.get("entity_name") or entity.get("id") or ""),
                "entity_type": str(entity.get("entity_type", "")),
                "source_id": str(entity.get("source_id", "")),
            }
        )
        if len(entities) >= limit:
            break
    return entities


# ---------------------------------------------------------------------------
# Context post-processing (memory_recall, context_only path)
# ---------------------------------------------------------------------------
# `rag.aquery(..., only_need_context=True)` returns a plain str in the
# installed lightrag-hku==1.4.16 (confirmed by reading
# lightrag.operate._build_context_str / PROMPTS["kg_query_context"]): it is a
# markdown document with fixed section headers, each followed by a fenced
# ```json block containing one JSON object per line (NOT a JSON array). That
# structure is parsed below. If a future LightRAG version changes this format,
# _postprocess_context() fails closed — it returns the original, unmodified
# context rather than raising, since this is best-effort cleanup, not a
# correctness-critical path.
_CONTEXT_ENTITY_SECTION_HEADER = "Knowledge Graph Data (Entity):"
_CONTEXT_CHUNK_SECTION_HEADER = "Document Chunks ("
_CONTEXT_REFERENCE_SECTION_HEADER = "Reference Document List ("
_CONTEXT_REFERENCE_LINE_RE = re.compile(r"^\[(?P<ref_id>[^\]]+)\]")


def _extract_fenced_block(context: str, header: str) -> tuple[int, int, str] | None:
    """Return (content_start, content_end, content) for the fenced block after `header`.

    The offsets delimit the block's content within `context` so a caller can
    splice a replacement back in without disturbing the surrounding template.
    """
    header_idx = context.find(header)
    if header_idx == -1:
        return None
    fence_start = context.find("```", header_idx)
    if fence_start == -1:
        return None
    content_start = context.find("\n", fence_start)
    if content_start == -1:
        return None
    content_start += 1
    fence_end = context.find("```", content_start)
    if fence_end == -1:
        return None
    content_end = context.rfind("\n", content_start, fence_end)
    if content_end == -1 or content_end < content_start:
        content_end = fence_end
    return content_start, content_end, context[content_start:content_end]


def _truncate_on_word_boundary(text: str, max_chars: int) -> str:
    """Truncate to max_chars without cutting a word in half."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def _sanitize_for_json(obj: Any, max_depth: int = 5, _depth: int = 0) -> Any:
    """Recursively sanitize a LightRAG result dict for JSON serialization.
    
    Converts non-serializable types (bytes, sets, numpy types, etc.) to strings,
    truncates oversized strings, and limits recursion depth.
    """
    if _depth >= max_depth:
        return str(obj)[:500]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v, max_depth, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v, max_depth, _depth + 1) for v in obj[:200]]
    if isinstance(obj, (str, int, float, bool)):
        return str(obj)[:5000] if isinstance(obj, str) else obj
    if obj is None:
        return None
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")[:5000]
    if isinstance(obj, set):
        return sorted(str(v) for v in obj)
    try:
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
    except NameError:
        pass
    return str(obj)[:5000]


def _dedupe_description(description: str) -> str:
    """Split a <SEP>-joined entity description into unique parts, then re-truncate cleanly."""
    if CONTEXT_DESCRIPTION_SEP not in description:
        return _truncate_on_word_boundary(description, ENTITY_DESCRIPTION_MAX_CHARS)
    parts = [part.strip() for part in description.split(CONTEXT_DESCRIPTION_SEP) if part.strip()]
    unique_parts = list(dict.fromkeys(parts))  # preserve order, drop exact duplicates
    return _truncate_on_word_boundary(" | ".join(unique_parts), ENTITY_DESCRIPTION_MAX_CHARS)


def _clean_entities_block(raw: str) -> str:
    """Drop UNKNOWN-typed / stray-'|' extraction garbage; dedupe <SEP>-joined descriptions."""
    kept: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        entity = json.loads(line)
        name = str(entity.get("entity", ""))
        entity_type = str(entity.get("type", ""))
        if entity_type == "UNKNOWN" or "|" in name:
            continue
        entity["description"] = _dedupe_description(str(entity.get("description", "")))
        kept.append(json.dumps(entity, ensure_ascii=False))
    return "\n".join(kept)


def _clean_chunks_block(raw: str) -> tuple[str, set[str]]:
    """Dedupe chunks by reference_id/content; drop short headerless orphan fragments."""
    seen: set[str] = set()
    kept: list[str] = []
    kept_reference_ids: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        chunk = json.loads(line)
        reference_id = str(chunk.get("reference_id", ""))
        content = str(chunk.get("content", ""))
        dedupe_key = reference_id or content
        if dedupe_key in seen:
            continue
        has_header = bool(CONTEXT_DOCUMENT_HEADER_RE.match(content.strip()))
        if len(content) < CONTEXT_MIN_CHUNK_CHARS and not has_header:
            continue
        seen.add(dedupe_key)
        kept.append(json.dumps(chunk, ensure_ascii=False))
        if reference_id:
            kept_reference_ids.add(reference_id)
    return "\n".join(kept), kept_reference_ids


def _clean_reference_block(raw: str, kept_reference_ids: set[str]) -> str:
    """Drop reference-list lines whose chunk was deduped/dropped above."""
    kept: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _CONTEXT_REFERENCE_LINE_RE.match(line.strip())
        if match and match.group("ref_id") not in kept_reference_ids:
            continue
        kept.append(line)
    return "\n".join(kept)


def _postprocess_context(context: str) -> str:
    """Clean up known LightRAG context garbage for the context_only query path.

    Applies (when the corresponding section is present): entity dedup of
    UNKNOWN/stray-'|' garbage nodes and <SEP>-joined description dedup with
    word-boundary truncation; chunk dedup by reference_id/content and removal
    of short headerless orphan fragments; and reference-list pruning to match
    the chunks that survived. Relations are left untouched (not implicated by
    the measured garbage). Fails closed to the original context — see the
    module comment above this section.
    """
    try:
        result = context

        entities_block = _extract_fenced_block(result, _CONTEXT_ENTITY_SECTION_HEADER)
        if entities_block is not None:
            start, end, raw = entities_block
            result = result[:start] + _clean_entities_block(raw) + result[end:]

        kept_reference_ids: set[str] | None = None
        chunks_block = _extract_fenced_block(result, _CONTEXT_CHUNK_SECTION_HEADER)
        if chunks_block is not None:
            start, end, raw = chunks_block
            cleaned, kept_reference_ids = _clean_chunks_block(raw)
            result = result[:start] + cleaned + result[end:]

        if kept_reference_ids is not None:
            refs_block = _extract_fenced_block(result, _CONTEXT_REFERENCE_SECTION_HEADER)
            if refs_block is not None:
                start, end, raw = refs_block
                result = result[:start] + _clean_reference_block(raw, kept_reference_ids) + result[end:]

        return result
    except Exception as exc:
        logger.debug("Context post-processing skipped (unexpected format): %s", exc)
        return context


def _has_graph_entity_context(context: object) -> bool:
    """Return whether LightRAG supplied at least one graph entity record."""
    if not isinstance(context, str):
        return False
    block = _extract_fenced_block(context, _CONTEXT_ENTITY_SECTION_HEADER)
    if block is None:
        return False
    for line in block[2].splitlines():
        try:
            entity = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(entity, dict) and entity.get("entity"):
            return True
    return False


# ---------------------------------------------------------------------------
# Opt-in primary-context merge (item 3, memory_recall `context_priority`).
#
# Measured via a scratch script built ONLY on tools/memory/eval/ab_bench.py's
# own scoring primitives (run_config/score_query/aggregate_scores — the
# harness itself has no "merged channel" config, per task constraints not
# editing eval/*), on the 46-query retrieval_queries.yaml set, top_k=10:
#
#   channel                                   recall@1  recall@10  ndcg@10  mrr
#   hybrid_bm25 (fusion) alone                  0.574     0.810     0.722   0.720
#   hybrid (lightrag mix, tuned budget) alone    0.491     0.769     0.662   0.659
#   round-robin(fusion first, lightrag) MERGE    0.574     0.880     0.750   0.732
#
# Merge beats BOTH pure channels on every aggregate metric, including
# multihop specifically (ndcg@10 0.759 fusion-alone / 0.802 lightrag-alone /
# 0.830 merged) — the two channels are measurably complementary, not
# redundant: fusion wins identifier/conceptual/latency, lightrag's own
# entity/relation graph traversal recovers multihop docs fusion misses, and
# the union recovers gains from both. round-robin with FUSION first was
# measured to matter (recall@1 0.574 fusion-first vs 0.491 lightrag-first
# on the identical merge) since fusion's BM25 channel wins exact-identifier
# rank-1 placement.
#
# This is opt-in (CONTEXT_PRIORITY_MERGED, default CONTEXT_PRIORITY_LIGHTRAG
# = today's unmodified `context` field) and only touches the Document
# Chunks / Reference Document List sections — entity/relation graph sections
# are left as LightRAG produced them, preserving the structure a pure-fusion
# swap would have discarded. Fusion-only documents (not already present in
# LightRAG's own chunk block) are injected using their hybrid-channel
# snippet (HYBRID_SNIPPET_MAX_CHARS), not full chunk content — a deliberate
# simplicity tradeoff so this reuses the already-computed, already-tested
# `hybrid.fused_chunks` data with zero extra embedding/BM25 calls.
# ---------------------------------------------------------------------------
_CONTEXT_REFERENCE_LINE_WITH_PATH_RE = re.compile(r"^\[(?P<ref_id>[^\]]+)\]\s*(?P<path>.+)$")


def _round_robin_merge(primary: list[str], secondary: list[str], limit: int) -> list[str]:
    """Interleave two ranked lists (primary[0], secondary[0], primary[1], ...),
    deduping and stopping at `limit`. `primary` is emitted first each round —
    see the module comment above for why that ordering was measured to matter.
    """
    out: list[str] = []
    seen: set[str] = set()
    i = j = 0
    while (i < len(primary) or j < len(secondary)) and len(out) < limit:
        if i < len(primary):
            item = primary[i]
            i += 1
            if item not in seen:
                seen.add(item)
                out.append(item)
        if len(out) >= limit:
            break
        if j < len(secondary):
            item = secondary[j]
            j += 1
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _parse_chunk_block(raw: str) -> list[dict[str, Any]]:
    """Parse the Document Chunks fenced JSON-lines block into an ordered list
    of {"reference_id": ..., "content": ...} dicts, in LightRAG's own emitted
    (rank) order. Malformed lines are skipped (fail-soft, matches
    _clean_chunks_block's tolerance)."""
    chunks: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return chunks


def _parse_reference_block(raw: str) -> dict[str, str]:
    """Parse the Reference Document List fenced block into {reference_id: file_path}."""
    mapping: dict[str, str] = {}
    for line in raw.splitlines():
        match = _CONTEXT_REFERENCE_LINE_WITH_PATH_RE.match(line.strip())
        if match:
            mapping[match.group("ref_id")] = match.group("path").strip()
    return mapping


def _chunk_dedup_key(chunk: dict[str, Any]) -> str:
    """Stable per-chunk identity WITHIN one document, used as the second half
    of the (file_path, chunk_id) dedup key. Prefers an explicit chunk_id
    (hybrid channel always carries one); LightRAG's own context block has no
    chunk_id field, so its chunks fall back to their content."""
    chunk_id = chunk.get("chunk_id") or chunk.get("id")
    if chunk_id:
        return str(chunk_id)
    return str(chunk.get("content") or chunk.get("snippet") or "")


def _content_covered(candidate: str, existing: str) -> bool:
    """True when `candidate` adds nothing over `existing` — the hybrid
    channel's snippet is a truncated prefix of the same chunk LightRAG
    already emitted, so it must not be injected a second time."""
    candidate_norm = " ".join(candidate.split())
    existing_norm = " ".join(existing.split())
    if not candidate_norm:
        return True
    return candidate_norm in existing_norm


def _representative_content(
    path: str,
    chunks_by_path: dict[str, list[dict[str, Any]]],
    fusion_by_path: dict[str, list[dict[str, Any]]],
) -> str:
    """Content used to score a DOCUMENT (supersession rescoring operates on
    documents, not chunks). Prefers LightRAG's own (fuller) chunks, joining
    all of them so a marker in any chunk of the document is seen; falls back
    to the fusion channel's content/snippet for fusion-exclusive documents."""
    lightrag = chunks_by_path.get(path) or []
    if lightrag:
        return "\n".join(str(chunk.get("content", "")) for chunk in lightrag)
    for fused_chunk in fusion_by_path.get(path) or []:
        content = str(fused_chunk.get("content") or fused_chunk.get("snippet", ""))
        if content:
            return content
    return ""


def _merge_context_with_fusion(
    context: str, fused_chunks: list[dict[str, Any]], limit: int
) -> tuple[str, bool]:
    """Reorder/augment `context`'s Document Chunks + Reference Document List
    sections into round-robin(fusion, lightrag) document order.

    Returns (new_context, applied). applied=False (context returned
    unchanged) if either fenced section is missing or the merge would be a
    no-op — fails closed, matching _postprocess_context's contract.
    """
    chunks_block = _extract_fenced_block(context, _CONTEXT_CHUNK_SECTION_HEADER)
    refs_block = _extract_fenced_block(context, _CONTEXT_REFERENCE_SECTION_HEADER)
    if chunks_block is None or refs_block is None:
        return context, False

    _, _, chunks_raw = chunks_block
    _, _, refs_raw = refs_block
    lightrag_chunks = _parse_chunk_block(chunks_raw)
    ref_id_to_path = _parse_reference_block(refs_raw)
    if not lightrag_chunks or not ref_id_to_path:
        return context, False

    # Multi-chunk per document: a single file legitimately contributes SEVERAL
    # ranked chunks (different sections of the same markdown document). Keying
    # by path alone dropped every chunk after the first, silently losing
    # relevant content. Chunks are deduped by (file_path, chunk_id) instead,
    # while DOCUMENT order (and therefore `limit`, which counts documents)
    # stays exactly as the round-robin merge produced it.
    lightrag_order: list[str] = []
    chunks_by_path: dict[str, list[dict[str, Any]]] = {}
    seen_lightrag_keys: set[tuple[str, str]] = set()
    for chunk in lightrag_chunks:
        ref_id = str(chunk.get("reference_id", ""))
        path = ref_id_to_path.get(ref_id, "")
        if not path:
            continue
        key = (path, _chunk_dedup_key(chunk))
        if key in seen_lightrag_keys:
            continue
        seen_lightrag_keys.add(key)
        if path not in chunks_by_path:
            lightrag_order.append(path)
            chunks_by_path[path] = []
        chunks_by_path[path].append(chunk)

    fusion_order: list[str] = []
    fusion_by_path: dict[str, list[dict[str, Any]]] = {}
    seen_fusion_keys: set[tuple[str, str]] = set()
    for fused_chunk in fused_chunks:
        path = str(fused_chunk.get("file_path", ""))
        if not path:
            continue
        key = (path, _chunk_dedup_key(fused_chunk))
        if key in seen_fusion_keys:
            continue
        seen_fusion_keys.add(key)
        if path not in fusion_by_path:
            fusion_order.append(path)
            fusion_by_path[path] = []
        fusion_by_path[path].append(fused_chunk)

    merged_order = _round_robin_merge(fusion_order, lightrag_order, limit)
    if not merged_order:
        return context, False

    # Supersession-aware rescoring of the FINAL MERGED list — companion to
    # HARS_MEMORY_SUPERSESSION_SCORING, gated by the SAME flag `fuse()` uses
    # (see retrieval/fusion.py's `marker_penalty_enabled`). Without this,
    # HARS_MEMORY_SUPERSESSION_SCORING is a no-op on context_priority=merged:
    # fuse() only ever reorders the FUSION channel's OWN candidates, so a
    # superseded doc reachable only via LightRAG's own graph/vector ranking
    # (never scored, never touched by fuse()) can still outrank the correct
    # doc in the interleaved output even when the fusion channel alone would
    # get the ordering right. Measured on the 46-query retrieval_queries.yaml
    # set (tools/memory/eval): merged+supersession was 0.333 supersession_
    # error_rate (identical to supersession OFF) before this rescoring pass;
    # 0.167 after — matching the fusion-channel-alone number, i.e. the gain
    # is no longer washed out by the merge. See
    # retrieval/supersession.py's `apply_marker_penalty_to_ranked_list`
    # docstring for why this is a stable partition (not the multiplicative
    # attenuation `fuse()` itself uses) and why that composes safely with
    # fuse()'s own already-applied penalty on the fusion-side share of this
    # same list without double-penalizing it (idempotent by construction).
    # Content lookup prefers LightRAG's own (fuller) chunk content when a
    # path is in both channels; falls back to the fusion channel's truncated
    # snippet (HYBRID_SNIPPET_MAX_CHARS) for fusion-exclusive documents —
    # same preference order the chunk-emission loop below already uses.
    from hars_memory.retrieval.fusion import marker_penalty_enabled

    if marker_penalty_enabled():
        from hars_memory.retrieval.supersession import apply_marker_penalty_to_ranked_list

        ranked_with_content = [
            (path, _representative_content(path, chunks_by_path, fusion_by_path))
            for path in merged_order
        ]
        merged_order = apply_marker_penalty_to_ranked_list(ranked_with_content)

    existing_ref_ids = {int(rid) for rid in ref_id_to_path if rid.isdigit()}
    next_ref_id = (max(existing_ref_ids) + 1) if existing_ref_ids else 1

    used_ref_ids: set[str] = set(ref_id_to_path)
    # Every EMITTED chunk gets its own reference_id: LightRAG's original id is
    # kept for the first chunk of a document (stable citations), any further
    # chunk of the same document (and every fusion-exclusive injection) gets a
    # freshly allocated one, so no two emitted chunks ever share an id.
    emitted_ref_ids: set[str] = set()

    def _allocate_ref_id() -> str:
        nonlocal next_ref_id
        while str(next_ref_id) in used_ref_ids:
            next_ref_id += 1
        ref_id = str(next_ref_id)
        next_ref_id += 1
        used_ref_ids.add(ref_id)
        return ref_id

    new_chunk_lines: list[str] = []
    new_ref_lines: list[str] = []
    for path in merged_order:
        emitted_contents: list[str] = []
        for chunk in chunks_by_path.get(path, []):
            own_ref_id = str(chunk.get("reference_id", ""))
            if own_ref_id and own_ref_id not in emitted_ref_ids:
                ref_id = own_ref_id
            else:
                ref_id = _allocate_ref_id()
            emitted_ref_ids.add(ref_id)
            emitted = dict(chunk)
            emitted["reference_id"] = ref_id
            emitted_contents.append(str(emitted.get("content", "")))
            new_chunk_lines.append(json.dumps(emitted, ensure_ascii=False))
            new_ref_lines.append(f"[{ref_id}] {path}")
        # Fusion-exclusive document: LightRAG's own context never surfaced it,
        # so every one of its fusion chunks is injected — full chunk content
        # when the hybrid channel provides it, snippet otherwise. Documents
        # LightRAG already emitted keep LightRAG's own (fuller) chunks only,
        # exactly as before, to avoid duplicating the same content twice.
        if chunks_by_path.get(path):
            continue
        for fused_chunk in fusion_by_path.get(path, []):
            content = str(fused_chunk.get("content") or fused_chunk.get("snippet", ""))
            if not content.strip():
                continue
            if any(_content_covered(content, existing) for existing in emitted_contents):
                continue
            ref_id = _allocate_ref_id()
            emitted_ref_ids.add(ref_id)
            emitted_contents.append(content)
            new_chunk_lines.append(json.dumps({"reference_id": ref_id, "content": content}, ensure_ascii=False))
            new_ref_lines.append(f"[{ref_id}] {path}")

    if not new_chunk_lines:
        return context, False

    result = context
    # Splice the chunks block, then re-locate the reference block on the
    # now-mutated `result` before splicing it — same sequential-splice
    # pattern as _postprocess_context, required because splicing the first
    # block shifts every downstream string offset.
    chunks_block_now = _extract_fenced_block(result, _CONTEXT_CHUNK_SECTION_HEADER)
    assert chunks_block_now is not None  # just matched above; header text is unchanged
    start, end, _ = chunks_block_now
    result = result[:start] + "\n".join(new_chunk_lines) + result[end:]

    refs_block_now = _extract_fenced_block(result, _CONTEXT_REFERENCE_SECTION_HEADER)
    assert refs_block_now is not None
    start, end, _ = refs_block_now
    result = result[:start] + "\n".join(new_ref_lines) + result[end:]

    return result, True


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        tool(
            "memory_recall",
            (
                "Query the HARS knowledge graph. "
                "IMPORTANT: supply ll_keywords (specific entities/codes, e.g. ['A2S32','Phase C','DINOv3']) "
                "and hl_keywords (themes/concepts, e.g. ['hypothesis falsification','depth encoding']) — "
                "you are the LLM, extract them from your question yourself; with keywords provided the "
                "query runs with NO local LLM (CPU embeddings + graph only). "
                "If BOTH ll_keywords and hl_keywords are omitted, the query silently runs in naive mode "
                "instead of the requested mode (naive needs no keywords and empirically has the best "
                "signal-to-noise without them) — the response's mode_fallback field explains when this fired. "
                "Default context_only=true returns raw graph context for you to synthesise. "
                "mode: local (entity-neighborhood) | global (community themes) | "
                "hybrid (default, both) | naive (pure vector, also the automatic no-keyword fallback). "
                "The response ALSO always includes a top-level `hybrid` field — NOT the same thing as "
                "mode='hybrid' above. It is additive dense+BM25-sparse fusion: exact identifiers "
                "(e.g. 'A2S32', 'phase_c1_lora_safe', 'hyp:7f62cfb4') are matched verbatim by the BM25 "
                "channel, which dense embeddings alone can miss. Check `hybrid.identifier_matches` first "
                "when the question names a specific code/id; `hybrid.fused_chunks` is the general "
                "alpha-weighted dense+sparse ranking (alpha via HARS_MEMORY_HYBRID_ALPHA, measured optimum "
                "default 0.5). Some `fused_chunks` entries have a `chunk_id` prefixed `ripgrep:` — these "
                "come from the ripgrep channel (HARS_MEMORY_RIPGREP_CHANNEL, default on), which searches "
                "the LIVE worktree (not the index) for exact identifiers drawn from BOTH the question text "
                "AND ll_keywords, so it can surface files added or edited since the last consolidation "
                "that the index has never seen at all — supplying ll_keywords (as instructed above) "
                "directly improves this channel's recall, not just the graph-mode retrieval; see "
                "`hybrid.ripgrep` for its own status (enabled/available/hits_count/query_terms). These "
                "entries are always appended AFTER the real top_k dense/sparse results, never displacing "
                "one. A separate flat-dense channel (HARS_MEMORY_FLAT_CHANNEL, default OFF — see "
                "`hybrid.flat_dense` for status) can additively cover chunks present in the index's raw "
                "chunk store but missing/stale in the primary vector store; off by default because that "
                "gap is currently empty on this index and the channel pays its own CPU embed cost per "
                "query — enable it if you have a specific reason to suspect index/vector-store drift. "
                "`hybrid.enabled=false` with a `reason`/`error` means the sparse channel "
                "wasn't available for this call — the rest of the response (context/answer) is "
                "unaffected. ALWAYS check `hybrid.confidence.low_confidence` before treating "
                "`context`/`answer` as reliable grounding: this corpus has no query that reliably "
                "returns empty on a no-answer question (measured no_answer_hit_rate stays 0.9-1.0 across "
                "every retrieval channel here), so a low_confidence=true marker is the only signal that "
                "distinguishes a real answer from confident-sounding noise."
            ),
            {
                "question": {"type": "string", "description": "Natural language question."},
                "mode": {
                    "type": "string",
                    "enum": ["local", "global", "hybrid", "naive"],
                    "default": "hybrid",
                    "description": "Retrieval mode. Overridden to 'naive' automatically when both "
                                   "ll_keywords and hl_keywords are empty (see mode_fallback in the response).",
                },
                "top_k": {
                    "type": "integer", "default": DEFAULT_QUERY_TOP_K, "minimum": 1, "maximum": 50,
                    "description": "The RESULT-COUNT knob — how many entities/relations/chunks you get "
                                   "back: LightRAG's own entity/relation search width, and the final "
                                   "truncation length of every ranked list this tool returns (merged "
                                   "context chunks, hybrid.fused_chunks, citations/entities_used). Does "
                                   "NOT control how wide the initial chunk fetch is — see fetch_top_k.",
                },
                "fetch_top_k": {
                    "type": "integer", "minimum": 1, "maximum": 200,
                    "description": "The FETCH-WIDTH knob — how many chunk candidates the vector search "
                                   "(LightRAG's QueryParam.chunk_top_k) and the hybrid dense+BM25 fusion "
                                   "pool retrieve BEFORE truncating down to top_k results. Must be >= "
                                   "top_k (refused otherwise). Widening this gives the fusion channel, "
                                   "the supersession rescoring, and the z-score tie-break more candidates "
                                   "to choose from without changing how many results you get back — at "
                                   "the cost of dense-channel latency (roughly linear in fetch width; "
                                   "measured +8ms wall time at 4x width on this corpus, BM25 stays ~0.5ms "
                                   "regardless). MEASURED 2026-08-01 (see "
                                   "HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER's comment for the full sweep): "
                                   "at the shipped top_k=20 default, widening does NOT help — every width "
                                   "tried beyond 20 only ever loses recall@1/ndcg@10/mrr, driven by "
                                   "identifier-query precision loss, with no compensating gain anywhere. "
                                   "It DOES help for a caller that deliberately passes a NARROWER top_k "
                                   "(e.g. top_k=10 for a smaller payload): set fetch_top_k to ~1.7-2.0x "
                                   "that top_k (e.g. top_k=10, fetch_top_k=18) to recover close to "
                                   "top_k=20-equivalent retrieval quality without getting 20 results back "
                                   "— going wider than ~2x starts giving some of that back. Omit to use "
                                   "top_k * HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER (env-configurable, "
                                   "default 1.0 = identical to top_k, i.e. no widening, today's "
                                   "pre-2026-08-01 behaviour — this is the measured-correct default at "
                                   "top_k=20, not merely a conservative fallback). See "
                                   "HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER's comment in "
                                   "this file for the measured sweep behind that default.",
                },
                "ll_keywords": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Low-level keywords: specific entities, codes, names from the question. "
                                   "If both ll_keywords and hl_keywords are left empty, the query silently "
                                   "falls back to naive mode — supply ll_keywords for graph-aware retrieval.",
                },
                "hl_keywords": {
                    "type": "array", "items": {"type": "string"},
                    "description": "High-level keywords: themes/concepts behind the question.",
                },
                "context_only": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return the retrieved graph context (entities/relations/chunks) "
                                   "WITHOUT running the local answer LLM. Default True: the calling "
                                   "agent synthesises the answer itself, which beats the small local LLM.",
                },
                "context_priority": {
                    "type": "string",
                    "enum": [CONTEXT_PRIORITY_LIGHTRAG, CONTEXT_PRIORITY_MERGED],
                    "default": DEFAULT_CONTEXT_PRIORITY,
                    "description": (
                        f"Default (as of 2026-07-30) is '{CONTEXT_PRIORITY_MERGED}' — see "
                        "DEFAULT_CONTEXT_PRIORITY / HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT to change the "
                        f"server-wide default; pass '{CONTEXT_PRIORITY_LIGHTRAG}' explicitly for the "
                        "pre-2026-07-30 behaviour (today's unmodified `context` field) on any single call. "
                        f"'{CONTEXT_PRIORITY_MERGED}' reorders/augments the context_only `context` "
                        "field's Document Chunks section into round-robin(fusion, lightrag) document "
                        "order, with supersession-aware rescoring of the merged list when "
                        "HARS_MEMORY_SUPERSESSION_SCORING is on (default on). Measured on this corpus "
                        "(46-query retrieval_queries.yaml, top_k=10) vs the pre-flip lightrag-only "
                        "default: recall@1 0.477->0.5324, recall@10 0.727->0.8241, ndcg@10 0.637->0.7151, "
                        "mrr 0.634->0.7030, supersession_error 0.333->0.1667 — the two channels are "
                        "complementary, not redundant, and supersession scoring further fixes stale-doc "
                        "outranking with no measured regression on any metric or query type. Fusion-only "
                        "documents are injected using their `hybrid.fused_chunks[].snippet` (truncated), "
                        "not full chunk content. Only affects context_only=True; ignored (no-op, see "
                        "`context_priority_applied` in the response) for the answer-generation path."
                    ),
                },
                "debug": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return debug data: fused_chunks at top level, latency_breakdown "
                                   "(total_ms, graph_channel_ms, hybrid_channels), llm_usage (token "
                                   "counts), and raw_result (raw LightRAG output). Use for benchmarking "
                                   "and comparison with other retrieval systems (other systems).",
                },
            },
            ["question"],
        ),
        tool(
            "memory_remember",
            (
                "Save a knowledge note into the long-term memory staging area. Notes accumulate as "
                "markdown files and are merged into the knowledge graph by the next KB update "
                "run (tools/memory/scripts/update_kb.sh). No LLM, no GPU — instant file write. "
                "Use for durable findings: experiment outcomes, decisions, falsifications, "
                "infra facts. Write self-contained prose with full entity names/codes."
            ),
            {
                "title": {"type": "string", "description": "Short kebab-case slug for the note filename."},
                "content": {"type": "string", "description": "The knowledge itself — self-contained markdown prose."},
                "importance": {"type": "string", "enum": ["critical", "normal", "low"], "default": "normal"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Topic tags, e.g. ['vea','b1','rocm']."},
            },
            ["title", "content"],
        ),
        tool(
            "memory_entities",
            "Search graph entities by name or alias. Returns matching nodes + 1-hop neighbourhood.",
            {
                "name": {"type": "string", "description": "Entity name or partial alias."},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            },
            ["name"],
        ),
        tool(
            "memory_related",
            (
                "Return a compact node/edge list for targeted graph traversal (1 hop from an entity). "
                "hops>1 is not supported: the induced subgraph explodes combinatorially "
                "(measured 2 hops = 160 KB, 3 hops = 2.1 MB) — unusable in any caller's context budget. "
                f"Even 1 hop on a hub node is capped at ~{SUBGRAPH_NODE_BUDGET} nodes / "
                f"~{SUBGRAPH_EDGE_BUDGET} edges; see the response's truncated/dropped_* fields."
            ),
            {
                "entity_id": {"type": "string", "description": "Stable entity ID (e.g. hyp:abc, exp:42)."},
                "hops": {"type": "integer", "default": SUBGRAPH_MAX_HOPS, "minimum": 1, "maximum": SUBGRAPH_MAX_HOPS},
            },
            ["entity_id"],
        ),
        tool(
            "memory_status",
            (
                "Return index freshness, node/edge counts, last ingest time, source breakdown, "
                "and configured models. Call this first to confirm the index exists before querying. "
                "This tool is GPU-free and always available."
            ),
            {},
        ),
        tool(
            "memory_consolidate",
            (
                "Trigger incremental ingest (admin). "
                "If HARS_MEMORY_GPU_GUARD_SCRIPT_PATH is configured, blocked while a "
                "GPU-exclusive workflow the consuming project defines is running; if unset, "
                "this generic package has no opinion on GPU concurrency and proceeds. "
                "paths: optional list of dirs to (re)index. "
                "since: ISO timestamp — only reindex files newer than this."
            ),
            {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to ingest. Defaults to .plans, docs.",
                },
                "since": {
                    "type": "string",
                    "description": "ISO8601 timestamp — reindex only files modified after this.",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Walk + count only, no LLM calls. Default true for safety.",
                },
            },
        ),
        tool(
            "memory_forget",
            (
                "Purge stale documents from the long-term memory index by document date, with "
                "keyword protection for knowledge that must survive. This is the memory system's "
                "forget/expire operation — without it the index only ever accumulates. "
                "GPU-free, no LLM calls: pure KV-store scan + LightRAG adelete_by_doc_id. "
                "Docs with an unrecognised/'unknown' header date are NEVER deleted, unconditionally "
                "(not affected by any argument here). "
                "Guardrails: apply defaults to false (dry-run report only, nothing deleted); "
                "apply=true is REFUSED unless at least one `protect` pattern is supplied OR "
                "confirm_unprotected=true is passed explicitly — this prevents an unqualified "
                "date cutoff from silently wiping an entire section. "
                "Supply exactly one of `before` / `older_than_days`."
            ),
            {
                "before": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD). Docs with a header Date strictly before "
                                   "this are candidates. Mutually exclusive with older_than_days.",
                },
                "older_than_days": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Docs older than this many days (from today) are candidates. "
                                   "Mutually exclusive with before.",
                },
                "protect": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Regex patterns (case-insensitive); a candidate whose filename OR "
                                   "content matches ANY pattern is never deleted. Required for "
                                   "apply=true unless confirm_unprotected=true is also set.",
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only consider these Section header values (e.g. ['session','db']). "
                                   "Default: all sections.",
                },
                "apply": {
                    "type": "boolean",
                    "default": False,
                    "description": "false (default) = dry-run report only. true = actually delete — "
                                   "refused without `protect` or confirm_unprotected=true.",
                },
                "confirm_unprotected": {
                    "type": "boolean",
                    "default": False,
                    "description": "Explicit opt-out of the `protect`-pattern requirement for "
                                   "apply=true. Only set this when you intend to delete every "
                                   "matching document in the date range with no keyword exceptions.",
                },
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    args: dict[str, Any] = arguments if isinstance(arguments, dict) else {}

    try:
        # ------------------------------------------------------------------ #
        # memory_status — always GPU-free                                    #
        # ------------------------------------------------------------------ #
        if name == "memory_status":
            return json_text({"ok": True, **_index_status()})

        # ------------------------------------------------------------------ #
        # memory_recall                                                        #
        # ------------------------------------------------------------------ #
        if name == "memory_recall":
            request_start = time.monotonic()
            question = str(args.get("question", ""))
            if not question:
                return json_text({"ok": False, "error": "question is required"})

            mode = str(args.get("mode", "hybrid"))
            rag_mode = _lightrag_mode(mode)
            top_k = int(args.get("top_k", DEFAULT_QUERY_TOP_K))
            fetch_top_k, fetch_top_k_source = _resolve_fetch_top_k(top_k, args.get("fetch_top_k"))
            if fetch_top_k is None:
                return json_text({"ok": False, "error": fetch_top_k_source})
            context_only = bool(args.get("context_only", True))
            debug = bool(args.get("debug", False))
            context_priority = str(args.get("context_priority", DEFAULT_CONTEXT_PRIORITY))
            ll_keywords = [str(k) for k in (args.get("ll_keywords") or [])]
            hl_keywords = [str(k) for k in (args.get("hl_keywords") or [])]
            # Pre-supplied keywords skip LightRAG's keyword-extraction LLM call
            # entirely — no local LLM needed on the query path.
            kw_args = {"ll_keywords": ll_keywords, "hl_keywords": hl_keywords} if (ll_keywords or hl_keywords) else {}
            rag_mode, mode_fallback = _resolve_query_mode(rag_mode, ll_keywords, hl_keywords)

            last_ingest, stale_days = _staleness_info()
            staleness_fields: dict[str, Any] = {"last_ingest": last_ingest, "stale_days": stale_days}
            if stale_days is not None and stale_days > STALE_INDEX_WARNING_DAYS:
                staleness_fields["staleness_warning"] = (
                    f"Knowledge graph last ingested {last_ingest} ({stale_days} days ago); "
                    "anything added or changed after that date is absent from this response."
                )

            rag = await _get_rag()

            # Additive hybrid (dense + BM25 sparse) fields — computed once, attached
            # to both response shapes below. Never raises; see _compute_hybrid_block.
            hybrid_block = await _compute_hybrid_block(
                rag, question, top_k, ll_keywords, fetch_top_k=fetch_top_k
            )

            # Usage-tracking envelope shared by every return path below (success or
            # failure) — see logging_setup.log_query_event. graph_channel_ms is
            # filled in around the LightRAG aquery/aquery_llm call itself (it does
            # entity/relation graph traversal for every mode except naive).
            recall_start = time.monotonic()

            def _emit_recall_event(
                *, ok: bool, results: list[dict[str, Any]], graph_channel_ms: float | None,
                context_priority_applied: str | None,
            ) -> None:
                hybrid_latency = hybrid_block.get("latency_ms") or {}
                log_query_event(
                    question=question,
                    ll_keywords=ll_keywords,
                    hl_keywords=hl_keywords,
                    mode_requested=mode,
                    mode_resolved=rag_mode,
                    mode_fallback=mode_fallback,
                    top_k=top_k,
                    context_only=context_only,
                    context_priority_requested=context_priority,
                    context_priority_applied=context_priority_applied,
                    ok=ok,
                    latency_ms={
                        "dense_channel": hybrid_latency.get("dense_channel"),
                        "sparse_channel": hybrid_latency.get("sparse_channel"),
                        "graph_channel": round(graph_channel_ms, 2) if graph_channel_ms is not None else None,
                        "total": round((time.monotonic() - recall_start) * 1000, 2),
                    },
                    candidate_pool_size=hybrid_block.get("candidate_pool_size"),
                    hybrid_enabled=bool(hybrid_block.get("enabled", False)),
                    hybrid_fail_reason=hybrid_block.get("reason") or hybrid_block.get("error"),
                    cache={
                        "bm25": (hybrid_block.get("bm25_index") or {}).get("cache_status"),
                        "graphml": None,  # not touched on this path — see log_query_event docstring
                    },
                    staleness_warning="staleness_warning" in staleness_fields,
                    low_confidence=(hybrid_block.get("confidence") or {}).get("low_confidence"),
                    results=results,
                )

            try:
                from lightrag import QueryParam  # type: ignore[import-not-found]
                from hars_memory.server.lightrag_init import create_query_model_func

                query_timeout = float(os.environ.get("HARS_MEMORY_QUERY_TIMEOUT_SECONDS", "60"))
                if context_only:
                    # No local answer LLM: return the retrieved graph context and let
                    # the calling agent synthesise the answer itself.
                    graph_start = time.monotonic()
                    context = await asyncio.wait_for(
                        rag.aquery(  # type: ignore[attr-defined]
                            question,
                            param=QueryParam(
                                mode=rag_mode,
                                top_k=top_k,
                                chunk_top_k=fetch_top_k,
                                only_need_context=True,
                                max_entity_tokens=DEFAULT_MAX_ENTITY_CONTEXT_BYTES,
                                max_relation_tokens=DEFAULT_MAX_RELATION_CONTEXT_BYTES,
                                **kw_args,
                            ),
                        ),
                        timeout=query_timeout,
                    )
                    graph_channel_ms = (time.monotonic() - graph_start) * 1000
                    has_context = bool(context) and str(context).strip() not in ("", "[no-context]")
                    if _has_graph_entity_context(context):
                        hybrid_block["confidence"]["low_confidence"] = False
                    context_priority_applied = CONTEXT_PRIORITY_LIGHTRAG
                    if has_context and isinstance(context, str):
                        context_for_response = _postprocess_context(context)
                        if context_priority == CONTEXT_PRIORITY_MERGED:
                            merged_context, applied = _merge_context_with_fusion(
                                context_for_response, hybrid_block.get("fused_chunks") or [], top_k
                            )
                            if applied:
                                context_for_response = merged_context
                                context_priority_applied = CONTEXT_PRIORITY_MERGED
                    elif has_context:
                        # Defensive: a future LightRAG version could return a structured
                        # object instead of a str. Post-processing only knows how to
                        # parse the confirmed str format, so it is skipped here rather
                        # than risking a bad transform on an unrecognised shape. Promoted
                        # to usable output: with HARS_MEMORY_LOG_LEVEL=DEBUG this now
                        # actually reaches the log file (previously dead at the
                        # hardcoded INFO level basicConfig() shipped).
                        context_for_response = context
                        logger.debug(
                            "memory_recall context is %s, not str — skipping post-processing.",
                            type(context).__name__,
                        )
                    else:
                        context_for_response = None
                    _emit_recall_event(
                        ok=has_context,
                        results=[
                            {"id": c.get("chunk_id"), "score": c.get("fused_score"), "source": "hybrid_fused"}
                            for c in (hybrid_block.get("fused_chunks") or [])
                        ],
                        graph_channel_ms=graph_channel_ms,
                        context_priority_applied=context_priority_applied,
                    )
                    response = {
                        "ok": has_context,
                        "context": context_for_response,
                        "error": None if has_context else "no context retrieved for this question",
                        "answer": None,
                        "mode": mode,
                        "lightrag_mode": rag_mode,
                        "mode_fallback": mode_fallback,
                        "top_k": top_k,
                        "fetch_top_k": fetch_top_k,
                        "fetch_top_k_source": fetch_top_k_source,
                        "question": question,
                        "note": "context_only=True — synthesise the answer from `context`.",
                        "context_priority_applied": context_priority_applied,
                        "hybrid": hybrid_block,
                        **staleness_fields,
                    }
                    if debug:
                        fused = hybrid_block.get("fused_chunks") or []
                        response["debug"] = {
                            "fused_chunks_count": len(fused),
                            "fused_chunks": fused,
                            "latency_breakdown": {
                                "total_ms": round((time.monotonic() - request_start) * 1000, 2),
                                "graph_channel_ms": graph_channel_ms,
                                "hybrid_channels": {
                                    "bm25_ms": hybrid_block.get("bm25_cache", {}).get("build_ms"),
                                    "ripgrep_ms": hybrid_block.get("ripgrep", {}).get("latency_ms"),
                                    "flat_dense_ms": hybrid_block.get("flat_dense", {}).get("latency_ms"),
                                },
                            },
                        }
                    return json_text(response)
                graph_start = time.monotonic()
                # LightRAG 1.5.6 does not support QueryParam.model_func.
                # Temporarily swap the instance-level llm_model_func so the
                # query LLM (rather than the extractor) is used for synthesis.
                async with _QUERY_MODEL_LOCK:
                    _orig_llm_func = getattr(rag, "llm_model_func", None)
                    try:
                        rag.llm_model_func = create_query_model_func()
                        result = await asyncio.wait_for(
                            rag.aquery_llm(  # type: ignore[attr-defined]
                                question,
                                param=QueryParam(
                                    mode=rag_mode,
                                    top_k=top_k,
                                    chunk_top_k=fetch_top_k,
                                    response_type=LLM_RESPONSE_TYPE,
                                    include_references=True,
                                    max_entity_tokens=DEFAULT_MAX_ENTITY_CONTEXT_BYTES,
                                    max_relation_tokens=DEFAULT_MAX_RELATION_CONTEXT_BYTES,
                                    **kw_args,
                                ),
                            ),
                            timeout=query_timeout,
                        )
                    finally:
                        rag.llm_model_func = _orig_llm_func
                graph_channel_ms = (time.monotonic() - graph_start) * 1000
                citations = _extract_citations(result, top_k)
                if _extract_entities_used(result, 1):
                    hybrid_block["confidence"]["low_confidence"] = False
                _emit_recall_event(
                    ok=result.get("status") != "failure",
                    results=[
                        {"id": c.get("node_id"), "score": c.get("score"), "source": "citation"}
                        for c in citations
                    ],
                    graph_channel_ms=graph_channel_ms,
                    context_priority_applied=None,  # context_priority only applies to the context_only path
                )
                response = {
                    "ok": result.get("status") != "failure",
                    "answer": _extract_answer(result),
                    "citations": citations,
                    "entities_used": _extract_entities_used(result, top_k),
                    "mode": mode,
                    "lightrag_mode": rag_mode,
                    "mode_fallback": mode_fallback,
                    "top_k": top_k,
                    "fetch_top_k": fetch_top_k,
                    "fetch_top_k_source": fetch_top_k_source,
                    "question": question,
                    "message": result.get("message"),
                    "hybrid": hybrid_block,
                    **staleness_fields,
                }
                if debug:
                    fused = hybrid_block.get("fused_chunks") or []
                    response["debug"] = {
                        "fused_chunks_count": len(fused),
                        "fused_chunks": fused,
                        "latency_breakdown": {
                            "total_ms": round((time.monotonic() - request_start) * 1000, 2),
                            "graph_channel_ms": graph_channel_ms,
                            "hybrid_channels": {
                                "bm25_ms": hybrid_block.get("bm25_cache", {}).get("build_ms"),
                                "ripgrep_ms": hybrid_block.get("ripgrep", {}).get("latency_ms"),
                                "flat_dense_ms": hybrid_block.get("flat_dense", {}).get("latency_ms"),
                            },
                        },
                        "llm_usage": {
                            "input_tokens": result.get("input_tokens") or result.get("usage", {}).get("input_tokens"),
                            "output_tokens": result.get("output_tokens") or result.get("usage", {}).get("output_tokens"),
                        },
                        "raw_lightrag_result": _sanitize_for_json(result),
                    }
                return json_text(response)
            except Exception as exc:
                logger.warning(
                    "memory_recall failed (mode=%s, context_only=%s, question=%r): %s",
                    rag_mode, context_only, question, exc,
                )
                _emit_recall_event(
                    ok=False, results=[], graph_channel_ms=None, context_priority_applied=None,
                )
                return json_text({
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "hint": (
                        "Ensure the index is built and the embedding backend is reachable "
                        "(context_only path uses no LLM)."
                        if context_only
                        else "Ensure the query LLM is running and the index is built."
                    ),
                })

        # ------------------------------------------------------------------ #
        # memory_remember                                                     #
        # ------------------------------------------------------------------ #
        if name == "memory_remember":
            import datetime
            import re as _re

            title = _re.sub(r"[^A-Za-z0-9._-]", "-", str(args.get("title", "note")))[:80]
            content = str(args.get("content", "")).strip()
            if not content:
                return json_text({"ok": False, "error": "content is required"})
            importance = str(args.get("importance", "normal"))
            tags = [str(x) for x in (args.get("tags") or [])]
            staging = Path(HARS_MEMORY_STAGING_DIR)
            staging.mkdir(parents=True, exist_ok=True)
            now = datetime.datetime.now()
            fname = f"{now:%Y-%m-%d}_{title}_{now:%H%M%S}.md"
            header = (f"[Document: {fname} | Section: staging | Date: {now:%Y-%m-%d} "
                      f"| Importance: {importance}]\n\n")
            meta = f"Tags: {', '.join(tags)}\n\n" if tags else ""
            (staging / fname).write_text(header + meta + content + "\n")
            pending = len(list(staging.glob("*.md")))
            log_write_event(
                tool="memory_remember",
                ok=True,
                detail={"saved": str(staging / fname), "pending_notes": pending,
                        "importance": importance, "tags": tags},
            )
            return json_text({
                "ok": True,
                "saved": str(staging / fname),
                "pending_notes": pending,
                "note": "Will be merged into the graph by the next update_kb.sh run.",
            })

        # ------------------------------------------------------------------ #
        # memory_entities                                              #
        # ------------------------------------------------------------------ #
        if name == "memory_entities":
            entity_name = str(args.get("name", ""))
            limit = int(args.get("limit", 10))
            if not entity_name:
                return json_text({"ok": False, "error": "name is required"})

            graph_file = _graph_file_path()
            if not graph_file.exists():
                return json_text({
                    "ok": False,
                    "error": "Index not built yet.",
                    "hint": "Run memory_consolidate or python -m hars_memory.server.index",
                })

            try:
                G, _graph_cache_status = await _get_graph()

                query_variants = _expand_query_aliases(entity_name)
                variant_infos = [
                    (norm, [t for t in norm.split(" ") if t])
                    for norm in (_normalize_entity_text(v) for v in query_variants)
                ]

                ranked: list[tuple[int, str, str, dict[str, Any]]] = []
                for node_id, node_data in G.nodes(data=True):
                    description = str(node_data.get("description", ""))
                    best_tier: str | None = None
                    for query_norm, query_tokens in variant_infos:
                        tier = _entity_match_tier(query_norm, query_tokens, node_id, description)
                        if tier is not None and (
                            best_tier is None or _MATCH_TIER_RANK[tier] < _MATCH_TIER_RANK[best_tier]
                        ):
                            best_tier = tier
                    if best_tier is not None:
                        ranked.append((_MATCH_TIER_RANK[best_tier], best_tier, str(node_id), node_data))

                # Rank by match tier, then id, for deterministic ordering. No padding
                # to `limit` — a relevance floor means genuine matches only, even if
                # that's fewer than `limit`.
                ranked.sort(key=lambda m: (m[0], m[2]))
                ranked = ranked[:limit]

                results = []
                for _rank, tier, node_id_str, node_data in ranked:
                    neighbors = [
                        {
                            "id": str(nb),
                            "relation": _edge_relation_label(G[node_id_str][nb]) if G.has_edge(node_id_str, nb) else "",
                        }
                        for nb in list(G.neighbors(node_id_str))[:10]
                    ]
                    results.append({
                        "id": node_id_str,
                        "description": str(node_data.get("description", ""))[:ENTITY_DESCRIPTION_MAX_CHARS],
                        "entity_type": str(node_data.get("entity_type", "")),
                        "match_tier": tier,
                        "neighbors": neighbors,
                    })

                return json_text({"ok": True, "query": entity_name, "results": results, "count": len(results)})
            except Exception as exc:
                logger.warning("memory_entities failed for query=%r: %s", entity_name, exc)
                return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

        # ------------------------------------------------------------------ #
        # memory_related                                                 #
        # ------------------------------------------------------------------ #
        if name == "memory_related":
            entity_id = str(args.get("entity_id", ""))
            hops = min(int(args.get("hops", SUBGRAPH_MAX_HOPS)), SUBGRAPH_MAX_HOPS)
            if not entity_id:
                return json_text({"ok": False, "error": "entity_id is required"})

            graph_file = _graph_file_path()
            if not graph_file.exists():
                return json_text({"ok": False, "error": "Index not built yet."})

            try:
                G, _graph_cache_status = await _get_graph()

                # Find node by exact id or case-insensitive match
                root_node = entity_id
                if root_node not in G.nodes:
                    matches = [n for n in G.nodes if entity_id.lower() in str(n).lower()]
                    if not matches:
                        return json_text({"ok": False, "error": f"Entity '{entity_id}' not found in graph."})
                    root_node = matches[0]

                # BFS up to hops
                subgraph_nodes: set[str] = {root_node}
                frontier = {root_node}
                for _ in range(hops):
                    next_frontier: set[str] = set()
                    for node in frontier:
                        next_frontier.update(G.neighbors(node))
                        if hasattr(G, "predecessors"):
                            next_frontier.update(G.predecessors(node))
                    frontier = next_frontier - subgraph_nodes
                    subgraph_nodes |= next_frontier
                    if not frontier:
                        break

                sub = G.subgraph(subgraph_nodes)
                nodes_out = [
                    {
                        "id": str(n),
                        "description": str(sub.nodes[n].get("description", ""))[:SUBGRAPH_DESCRIPTION_MAX_CHARS],
                        "entity_type": str(sub.nodes[n].get("entity_type", "")),
                    }
                    for n in sub.nodes
                ]
                edges_out = [
                    {
                        "source": str(u),
                        "target": str(v),
                        "relation": _edge_relation_label(sub[u][v]),
                    }
                    for u, v in sub.edges
                ]

                # Budget guard: even 1 hop on a hub node can produce thousands of
                # nodes/edges (measured 3 hops = 2.1 MB / 64,841 lines). Truncate
                # deterministically, always keeping the root node.
                truncated = False
                original_node_count = len(nodes_out)
                original_edge_count = len(edges_out)
                if len(nodes_out) > SUBGRAPH_NODE_BUDGET:
                    root_id = str(root_node)
                    non_root = sorted((n for n in nodes_out if n["id"] != root_id), key=lambda n: n["id"])
                    root_entry = [n for n in nodes_out if n["id"] == root_id]
                    nodes_out = (root_entry + non_root)[:SUBGRAPH_NODE_BUDGET]
                    truncated = True

                kept_node_ids = {n["id"] for n in nodes_out}
                edges_out = [e for e in edges_out if e["source"] in kept_node_ids and e["target"] in kept_node_ids]
                if len(edges_out) > SUBGRAPH_EDGE_BUDGET:
                    edges_out = edges_out[:SUBGRAPH_EDGE_BUDGET]
                    truncated = True

                dropped_nodes = original_node_count - len(nodes_out)
                dropped_edges = original_edge_count - len(edges_out)

                return json_text({
                    "ok": True,
                    "root": root_node,
                    "hops": hops,
                    "nodes": nodes_out,
                    "edges": edges_out,
                    "node_count": len(nodes_out),
                    "edge_count": len(edges_out),
                    "truncated": truncated,
                    "dropped_nodes": dropped_nodes,
                    "dropped_edges": dropped_edges,
                })
            except Exception as exc:
                logger.warning("memory_related failed for entity_id=%r hops=%d: %s", entity_id, hops, exc)
                return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

        # ------------------------------------------------------------------ #
        # memory_consolidate                                                      #
        # ------------------------------------------------------------------ #
        if name == "memory_consolidate":
            dry_run = bool(args.get("dry_run", True))
            paths = args.get("paths") or [".plans", "docs"]

            # GPU guard — reindex requires extractor LLM, and on a shared GPU
            # machine that LLM must not compete with a training run. Whether
            # such a concurrency check even applies is entirely the consuming
            # project's concern, not this generic package's: the guard script
            # path is fully configurable via HARS_MEMORY_GPU_GUARD_SCRIPT_PATH
            # (a Python file exposing `assert_gpu_free(api_base_url)`, loaded
            # dynamically via its file path since it lives in a separate,
            # consumer-owned project with its own venv — no normal package
            # import is possible across that boundary). Unset (the default for
            # a fresh install of this package) means "no GPU guard configured"
            # — this is logged, not treated as an error, and memory_consolidate
            # proceeds: GPU-guarding is Cortex's own operational concern (see
            # tools/memory-config/scripts/gpu_guard.py in the consuming repo),
            # not something this generic package should hardcode a path for.
            if not dry_run:
                gpu_guard_script_path = os.environ.get("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", "").strip()
                if not gpu_guard_script_path:
                    logger.info(
                        "HARS_MEMORY_GPU_GUARD_SCRIPT_PATH not set — no GPU guard configured, "
                        "proceeding without a GPU-concurrency check. Set this env var to a "
                        "Python file exposing assert_gpu_free(api_base_url) to enable one."
                    )
                else:
                    try:
                        import importlib.util as _importlib_util

                        _gpu_guard_path = Path(gpu_guard_script_path)
                        _gpu_guard_spec = _importlib_util.spec_from_file_location(
                            "hars_memory_gpu_guard", _gpu_guard_path
                        )
                        if _gpu_guard_spec is None or _gpu_guard_spec.loader is None:
                            raise ImportError(f"could not load gpu_guard module from {_gpu_guard_path}")
                        _gpu_guard_module = _importlib_util.module_from_spec(_gpu_guard_spec)
                        _gpu_guard_spec.loader.exec_module(_gpu_guard_module)
                        assert_gpu_free = _gpu_guard_module.assert_gpu_free

                        assert_gpu_free(HARS_API_BASE_URL)
                    except Exception as exc:
                        logger.warning("memory_consolidate blocked by GPU guard: %s", exc)
                        log_write_event(
                            tool="memory_consolidate",
                            ok=False,
                            detail={"dry_run": dry_run, "paths": [str(p) for p in paths],
                                    "stage": "gpu_guard", "error_type": type(exc).__name__, "error": str(exc)},
                        )
                        return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

            # Build CLI args and launch in subprocess so MCP server stays responsive.
            # Invoked as an installed module (`-m hars_memory.server.index`), NOT a
            # constructed file path — this is correct whether the package is a real
            # pip/uv install (site-packages) or a source checkout, with no
            # "project root" path-guessing either way. Inherits this process's own
            # environment (including sys.executable's venv), so HARS_MEMORY_* config
            # already validated by this server applies identically to the subprocess.
            import subprocess
            cmd = [
                sys.executable,
                "-m", "hars_memory.server.index",
                "--paths", *[str(p) for p in paths],
            ]
            if dry_run:
                cmd.append("--dry-run")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200 if not dry_run else 60,
                )
                log_write_event(
                    tool="memory_consolidate",
                    ok=result.returncode == 0,
                    detail={"dry_run": dry_run, "paths": [str(p) for p in paths],
                            "returncode": result.returncode},
                )
                return json_text({
                    "ok": result.returncode == 0,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-3000:],
                    "stderr": result.stderr[-1000:],
                    "dry_run": dry_run,
                })
            except subprocess.TimeoutExpired:
                logger.error("memory_consolidate subprocess timed out after 7200s (paths=%s)", paths)
                log_write_event(
                    tool="memory_consolidate",
                    ok=False,
                    detail={"dry_run": dry_run, "paths": [str(p) for p in paths],
                            "stage": "subprocess", "error": "timeout (7200s)"},
                )
                return json_text({"ok": False, "error": "Indexing timed out (7200s)."})
            except Exception as exc:
                logger.error("memory_consolidate subprocess failed (paths=%s): %s", paths, exc)
                log_write_event(
                    tool="memory_consolidate",
                    ok=False,
                    detail={"dry_run": dry_run, "paths": [str(p) for p in paths],
                            "stage": "subprocess", "error_type": type(exc).__name__, "error": str(exc)},
                )
                return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

        # ------------------------------------------------------------------ #
        # memory_forget                                                        #
        # ------------------------------------------------------------------ #
        if name == "memory_forget":
            import datetime as _dt

            from hars_memory.scripts.cleanup_kb import find_candidates, purge_documents

            before = args.get("before")
            older_than_days = args.get("older_than_days")
            protect_patterns = [str(p) for p in (args.get("protect") or [])]
            sections = {str(s).strip() for s in (args.get("sections") or []) if str(s).strip()}
            apply_ = bool(args.get("apply", False))
            confirm_unprotected = bool(args.get("confirm_unprotected", False))

            if bool(before) == (older_than_days is not None):
                return json_text({
                    "ok": False,
                    "error": "supply exactly one of 'before' (ISO date) or 'older_than_days'",
                })
            if apply_ and not protect_patterns and not confirm_unprotected:
                return json_text({
                    "ok": False,
                    "error": (
                        "apply=true refused: no 'protect' patterns supplied and "
                        "confirm_unprotected is not set. Add at least one protect regex, or pass "
                        "confirm_unprotected=true if you intend to delete every candidate in the "
                        "date range with no keyword exceptions."
                    ),
                })

            try:
                cutoff = (
                    _dt.date.fromisoformat(str(before))
                    if before
                    else _dt.date.today() - _dt.timedelta(days=int(older_than_days))
                )
            except ValueError as exc:
                return json_text({"ok": False, "error_type": "ValueError", "error": str(exc)})

            keep_res = [re.compile(p, re.IGNORECASE) for p in protect_patterns]
            wdir = Path(HARS_MEMORY_INDEX_DIR)

            try:
                report = find_candidates(wdir, cutoff, keep_res, sections)
            except FileNotFoundError as exc:
                return json_text({
                    "ok": False,
                    "error": f"Index not built yet at {wdir}: {exc}",
                })
            except Exception as exc:
                return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

            candidates_out = [
                {"doc_id": v.doc_id, "file": v.fname, "section": v.section, "date": v.date}
                for v in sorted(report.victims, key=lambda v: v.date)
            ]
            result: dict[str, Any] = {
                "ok": True,
                "dry_run": not apply_,
                "working_dir": str(wdir),
                "cutoff": cutoff.isoformat(),
                "docs_total": report.docs_total,
                "candidates": candidates_out,
                "candidate_count": len(report.victims),
                "protected": report.protected_count,
                "undated_never_touched": report.undated_count,
                "sections_scanned": sorted(sections) if sections else "all",
                "deleted": 0,
            }
            if not apply_:
                return json_text(result)
            if not report.victims:
                result["note"] = "nothing to delete"
                return json_text(result)

            try:
                deleted = await purge_documents(wdir, report.victims)
            except Exception as exc:
                logger.error(
                    "memory_forget purge failed (cutoff=%s, candidates=%d): %s",
                    cutoff.isoformat(), len(report.victims), exc,
                )
                result["ok"] = False
                result["error_type"] = type(exc).__name__
                result["error"] = str(exc)
                log_write_event(
                    tool="memory_forget",
                    ok=False,
                    detail={
                        "cutoff": cutoff.isoformat(), "apply": apply_, "candidate_doc_ids": [v.doc_id for v in report.victims],
                        "candidate_count": len(report.victims), "protected": report.protected_count,
                        "sections_scanned": sorted(sections) if sections else "all",
                        "error_type": type(exc).__name__, "error": str(exc),
                    },
                )
                return json_text(result)

            result["deleted"] = deleted
            log_write_event(
                tool="memory_forget",
                ok=True,
                detail={
                    "cutoff": cutoff.isoformat(), "apply": apply_,
                    "deleted_doc_ids": [v.doc_id for v in report.victims], "deleted_count": deleted,
                    "protected": report.protected_count, "sections_scanned": sorted(sections) if sections else "all",
                },
            )
            return json_text(result)

        return json_text({"ok": False, "error": f"Unknown tool: {name}"})

    except Exception as exc:
        logger.exception("Unhandled error in tool %s", name)
        error_type = type(exc).__name__
        error_message = str(exc)
        # Kill-switch drill (2026-07-30 Qdrant migration, plan §8/§9): a
        # low-level transport exception ("[Errno 111] Connection refused",
        # qdrant_client.http.exceptions.ResponseHandlingException /
        # httpx.ConnectError) is technically caught here already (no crash),
        # but is not by itself an ACTIONABLE error — it never names Qdrant or
        # suggests a fix. Rewrite it when the configured backend is Qdrant and
        # the failure looks connection-shaped; every other exception type/
        # message is returned unmodified.
        if "qdrant" in HARS_MEMORY_VECTOR_STORAGE.lower() and (
            "Connection refused" in error_message
            or "ConnectError" in error_type
            or "ResponseHandlingException" in error_type
            or "ConnectTimeout" in error_type
            or "TimeoutError" in error_type
        ):
            error_message = (
                f"Qdrant vector backend unreachable at {HARS_MEMORY_QDRANT_URL} "
                f"(underlying error: {error_type}: {error_message}). Check the "
                "hars-memory-qdrant container: `docker ps --filter name=hars-memory-qdrant` "
                "/ restart with `docker compose -f docker-compose.dev.yml up -d hars-memory-qdrant`."
            )
        return json_text({"ok": False, "error_type": error_type, "error": error_message})


# ---------------------------------------------------------------------------
# Server entrypoint
# ---------------------------------------------------------------------------


async def _serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    """Synchronous entrypoint — this is what `[project.scripts]` points the
    `hars-longterm-memory-mcp` console script at. Console-script entrypoints
    must be plain callables (setuptools/hatchling invoke them with no event
    loop running), so this wraps the actual async server loop (`_serve()`)
    in `asyncio.run()` rather than exposing that coroutine function directly.
    """
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
