# Retrieval

`memory_recall` combines several channels and merges them into one ranked
context.

## Channels

| Channel | Default | Purpose |
|---|---|---|
| LightRAG graph + vectors | on | entity/relation traversal (`local`, `global`, `hybrid`) or plain vector search (`naive`) |
| BM25 sparse | on | exact identifiers and rare tokens that dense embeddings blur |
| Ripgrep live files | on when roots are configured | finds content added since the last indexing; appended after the top results, never displacing them |
| Flat dense | off (`HARS_MEMORY_FLAT_CHANNEL=1`) | covers chunks missing or stale in the vector store; costs CPU embedding per query |

Dense and sparse scores are combined as `alpha*dense + (1-alpha)*sparse`
(`HARS_MEMORY_HYBRID_ALPHA`, default 0.5). Ranking is deterministic: ties are
broken by a z-score signal, and documents found by several channels receive a
small agreement bonus (`HARS_MEMORY_FUSION_AGREEMENT_BONUS`, 0.05).

Supplying `ll_keywords` (specific names/codes) and `hl_keywords` (themes) skips
the keyword-extraction LLM call and improves both the graph and ripgrep channels.

## Context assembly

With `context_only=true` (default) and `context_priority=merged` (default):
LightRAG's context is parsed into chunks, fusion results are interleaved with
them round-robin, supersession rescoring re-ranks the list, and fusion-only
documents are injected as truncated snippets. `context_priority=lightrag` keeps
LightRAG's original order. Chunks carry a `section` breadcrumb and, for indexes
built with location metadata, `heading_path`, `start_line`, `end_line` and
`source_path`.

## Supersession-aware ranking

Documents that declare themselves deprecated or superseded are demoted
(`HARS_MEMORY_SUPERSESSION_SCORING=1`, marker penalty on). A recency-discount
signal exists but measured harmful and is off by default.

## Reranking

`HARS_MEMORY_RERANK_BACKEND`:

| Value | Behaviour |
|---|---|
| unset | legacy: use the local cross-encoder only if `HARS_MEMORY_RERANK_MODEL` is set |
| `off` | no reranking anywhere (also disables LightRAG's native rerank pass) |
| `local` | force the local cross-encoder (`HARS_MEMORY_RERANK_MODEL`, `_DEVICE`, `_BATCH_SIZE`) |
| `http` | POST the top `HARS_MEMORY_RERANK_POOL` (20) fused candidates to `HARS_MEMORY_RERANK_HTTP_URL` (a `/v1/rerank`-style endpoint, e.g. a llama.cpp server with a reranker model); the rest keep their order |

On timeout (`HARS_MEMORY_RERANK_TIMEOUT_S`, 8) or error the fused order is kept and
`hybrid.rerank.fallback_reason` says why. A reranker adds seconds per query on
CPU; measure it on your own data before enabling.

## Response size

| Mode | Effect |
|---|---|
| default (`view=full`) | everything, including the context blob |
| `compact=true` | removes duplicated text (`snippet`s, chunk content already in `context`) |
| `view=lean` | no context blob; chunks with ids, scores and location metadata; `omitted` lists dropped fields |

## Tuning knobs

| Goal | Knob |
|---|---|
| More candidates before truncation | `fetch_top_k`, `HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER` |
| Favour exact tokens over semantics | lower `HARS_MEMORY_HYBRID_ALPHA` |
| Smaller chunks / better table handling | `HARS_MEMORY_CHUNKER=markdown`, `HARS_MEMORY_CHUNK_TOKEN_SIZE` |
| Fewer stale hits | keep supersession scoring on |
| Precision at rank 1 | enable a reranker |
| Freshness without re-indexing | configure ripgrep roots / sources manifest `grep: true` |

Change one knob at a time and compare with the evaluation harness
([evaluation.md](evaluation.md)).
