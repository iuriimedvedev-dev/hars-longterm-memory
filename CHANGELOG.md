# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.7.0] - 2026-10-17

### Added

- Live recall telemetry over SSE (`memory_trace`): a read-only streaming
  endpoint that emits one event per `memory_recall` pipeline stage as it runs
  (`hars_memory/trace.py`, `ContextVar` emitter, dependency-free HTTP/1.1 SSE
  listener started by `memory mcp` on a sibling port —
  `HARS_MEMORY_TRACE_HTTP_PORT`, default 8098, `0` disables). The default
  MCP/CLI path emits nothing and is behaviourally unchanged; `/mcp` is
  untouched. Notably enables live animation in `memory-travel`. See
  [docs/trace-api.md](docs/trace-api.md).
- Relational multi-hop evaluation harness for the graph branch
  (`hars_memory/eval/relational_eval.py`): read-only `validate` / `run` /
  `mine` subcommands that score the graph branch (graph_expand entities +
  relations + assembled context docs) and the fusion-only branch side by side,
  reusing `hars_memory.eval.metrics` conventions.
- Optional server-side stop sequences via `HARS_MEMORY_LLM_STOP`
  (comma-split, default off): `make_llm_func` now sends a Chat Completions
  `stop` list that stops generation at the `<|COMPLETE|>` delimiter instead of
  discarding ~24% of generated tokens afterwards (RCA
  `eval_agentic/rca/index_perf_analysis.md` §3b). Routed through the existing
  named-param override semantics, not `extra_body`. Cache-safe (prompt-keyed).

### Changed

- Documented the true two-branch recall DAG: fusion/rerank (`_compute_hybrid_block`,
  dense `chunks_vdb.query` + BM25 → fuse → rerank) completes BEFORE the LightRAG
  entity/relation pass (`graph_expand`); fusion has no graph channel. A dependency
  guard pins the recorded call order, a graph-free fusion payload and
  `graph_expand.derived_from == context_string` so the ordering cannot regress to
  match a UI label. See [docs/trace-api.md](docs/trace-api.md).
- Operations handoff for memory indexing setup, and a note on the live trace SSE
  endpoint (`:8098`).

### Fixed

- `md_struct` chunking used the wrong units: `md_struct_chunking_func` passed the
  token budget (`chunk_token_size`, default 512) as a *character* target to a
  tokenizer-based measure, producing ~4× oversized (~2048-token / ~8 KB) section
  chunks. The parameter is renamed to `chunk_token_budget` (tokens), documented,
  and converted to the chunker's char budget via the canonical `CHARS_PER_TOKEN=4`
  ratio while measuring in chars; the tokenizer still records the real token
  count. Also hardened `_split_block_lines`: a table whose repeated header rows
  alone exceed the budget no longer explodes into one chunk per character
  (a 26 kB table had produced 22,003 overlapping chunks) — it falls back to a
  plain hard split.
- `relational_eval` mining and aggregation: `mine` now requires A/C chains to be
  document-local with B as the cross-document hub (the bridge), and aggregate
  scores add graph/fusion `doc_solved@10` at a matched top-10 budget (the served
  context is ~top-10 while `fused_chunks` is a wide candidate pool), so the
  "would fusion alone solve it" comparison is fair.

## [0.6.0] - 2026-10-14

### Added

- Remote/shared embedder mode: `HARS_MEMORY_EMBED_PROVIDER=http` sends texts to
  an OpenAI-compatible `/v1/embeddings` endpoint instead of loading
  sentence-transformers in-process (~300–600 MB RAM saved by sharing one
  embedder, e.g. a llama.cpp `--embeddings` server or a hosted API). Query
  prompts / doc prefixes are applied client-side for vector parity;
  batching (`HARS_MEMORY_EMBED_BATCH_SIZE`,
  `HARS_MEMORY_EMBED_HTTP_MAX_INPUTS_PER_REQUEST` with automatic halving on
  size-limit 400s), 429 `Retry-After` + 5xx retries, optional pacing
  (`HARS_MEMORY_EMBED_HTTP_MIN_INTERVAL_MS`), Bearer auth
  (`HARS_MEMORY_EMBED_API_KEY`), dimension validation against the index, and
  token-usage cost logging. Default (no env vars) behaviour is unchanged.
  See README "Shared & hosted remote embedders".
- Remote/shared reranker mode: `HARS_MEMORY_RERANK_PROVIDER=http` sends rerank
  calls to a Cohere-style `POST {BASE_URL}/rerank` endpoint (verified against
  llama.cpp `/v1/rerank`; same code-diver endpoint the MCP fusion rerank uses)
  instead of loading a cross-encoder in-process. Chunked requests, client-side
  `top_n` after merge, 429 `Retry-After` + 5xx retries, optional pacing, Bearer
  auth, run-summary logs. `auto` (default) selects http when
  `HARS_MEMORY_RERANK_BASE_URL` is set, local when `HARS_MEMORY_RERANK_MODEL`
  is set, disabled otherwise. See README "Shared & hosted remote rerankers".
- Structure-aware Markdown chunking: `HARS_MEMORY_CHUNKER_MODE=md_struct`
  (or `index --chunker-mode md_struct`, new `ingest/md_structure.py`) chunks
  notes on section boundaries with breadcrumb prefixes, frontmatter parsing,
  fence-aware headings, table row-blocks and per-note manifests for future
  section routing. Default `legacy` is byte-identical to before. Adopting
  `md_struct` changes chunk texts/ids and needs a full re-embed, not an
  incremental update.
- Extractor benchmark tool: `memory extractor-bench run/score/report` over
  fixed fixtures (`eval/fixtures/extractor_bench_v1/`) with a thinking-leak
  guard, for comparing extractor LLMs.
- `memory_recall` accepts an optional `max_chars` argument sizing the returned
  context; context assembly caps each document block
  (`HARS_MEMORY_CONTEXT_MAX_DOC_CHARS`, default 6000, truncated at line
  boundaries) and admits whole documents within the total budget
  (`HARS_MEMORY_CONTEXT_MAX_CHARS`) — no more mid-buffer amputations.
- Rerank quality guards in the MCP fusion path: per-request document truncation
  (`HARS_MEMORY_RERANK_HTTP_DOC_CHARS`, default 2000), margin-guarded blending
  (`HARS_MEMORY_RERANK_FUSION_MARGIN`, default 0.15 — a saturated reranker can
  no longer promote a worst-in-pool doc past a confident fusion top-1), and
  rerank-skip when fusion is already confident
  (`HARS_MEMORY_RERANK_SKIP_FUSION_SCORE`, default 0.9).

### Fixed

- `memory recall` no longer crashes with `TypeError` when
  `hybrid.identifier_matches` is `None` (treat `None` as empty).
- Bridge servability: supergateway now runs `--stateful` (one stdio child per
  bridge instead of one per request — removes the 13–15 s per-call LightRAG
  re-init; see `cortex/scripts/hars-longterm-memory-mcp-http.sh`).

## [0.5.0] - 2026-10-13

### Added

- Incremental refresh of a LightRAG index: `python -m hars_memory.scripts.incremental_refresh`
  with `plan`, `apply`, `audit` and `repair-orphans` subcommands, see
  [docs/incremental-refresh.md](docs/incremental-refresh.md).
  - `plan` is read-only and classifies every walked document against the fingerprint
    sidecar and the content LightRAG actually stored (new, changed, retry, adopt,
    unchanged, gone, kept).
  - `apply` deletes many documents in one batched pass (entity/relation rebuild and
    storage flushes run once for the union) and inserts in verified batches with a
    fingerprint checkpoint after each batch. An unchanged index is a strict no-op.
  - `audit` cross-checks the KV stores, the graph and Qdrant; `repair-orphans` removes
    Qdrant points left behind by a killed run.
  - `HARS_MEMORY_ID_PATH_MAP=ACTUAL=LOGICAL` lets a refresh run from another checkout
    location reproduce the document ids stored in an existing index.
- Config-driven extractor provider: `HARS_MEMORY_EXTRACTOR_PROVIDER=local|openrouter`
  with `HARS_MEMORY_OPENROUTER_MODEL`, `HARS_MEMORY_OPENROUTER_BASE_URL`,
  `HARS_MEMORY_EXTRACTOR_API_KEY_ENV`, `HARS_MEMORY_EXTRACTOR_REASONING_EFFORT` and
  `HARS_MEMORY_EXTRACTOR_EXTRA_BODY_JSON`. The default `local` provider is unchanged.
  Credit: Angkira.
- Optional dotenv-style key file `HARS_MEMORY_EXTRACTOR_API_KEY_FILE`, consulted only
  when the key variable is unset. Credit: Angkira.
- CHANGELOG.md and a configuration reference section for the extractor provider.

### Changed

- `memory_consolidate` and `cortex-scripts/update_kb.sh` skip the GPU guard (and do not
  start a local extractor) when the extractor provider is hosted. An empty
  `HARS_MEMORY_EXTRACTOR_PROVIDER` is treated as `local`, so the guard still runs.
  Credit: Angkira.
- `cortex-scripts/update_kb.sh` archives a staged note only after its LightRAG
  `doc_status` is `processed` (`hars_memory.scripts.archive_ingested`). Failed notes stay
  in staging and the script exits non-zero instead of silently losing them.
  Credit: Angkira.

### Fixed

- `audit` tolerates LightRAG's `md5(src + tgt)` relationship point-id clashes: edges
  that share a point id are reported under `relationship_id_collisions` instead of
  failing the audit, and self-loop edges no longer crash the clash lookup.

## [0.4.0] - 2026-10-03

- Release commit `a69ae4b`; session, plan and experiment files are no longer tracked
  and are excluded from the sdist. No detailed notes were kept for earlier versions,
  see the git history.
