# Live LLM E2E preparation and strategy benchmark abstractions

Date: 2026-08-26

## Implemented

- `IndexStrategy` and `SearchStrategy`: immutable, validated, serializable,
  stable-hash contracts.
- Per-job indexing strategy transport through SDK/API/worker into corpus and
  LightRAG engines. Published manifests contain the strategy snapshot/hash.
- Strict YAML `StrategyMatrix` with relative-path resolution, compatible
  index/search expansion, warmups, repetitions, k-values, and an explicit
  primary metric/direction.
- `memory strategy-bench` and `memory-strategy-bench`: corpus runs execute
  directly; LightRAG indexing and retrieval-only `ab_bench` run in isolated
  subprocesses.
- Reports contain corpus/query/index/strategy hashes, build time, index size,
  document/chunk throughput, Recall/NDCG/MRR, latency/variance, raw repeated
  measurements, and a leaderboard.
- Strategy benchmarks opt into portable gold-path normalization, so an index
  storing absolute or `upload://` paths scores against basename/suffix gold
  labels without changing the historical A/B evaluator default.
- Tiny deterministic live corpus, extension, gold queries, sample strategy
  matrix, and opt-in real SDK/API/LightRAG/MCP E2E.

## Safety and live gate

Normal tests never contact an LLM. The live journey requires
`HARS_RUN_LIVE_LLM_E2E=1`, preflights both `/v1/models` endpoints, uses local
CPU embeddings plus NanoVectorDB/NetworkX for portable artifacts, limits
indexing to one concurrent extraction slot, and has a bounded job timeout.

The live journey validates create v1, extend v2, immutable v1 redownload,
strategy provenance, MCP context differences between v1/v2, and query-LLM
synthesis grounded on the rollback extension. It remains unexecuted until the
operator starts the real model servers.

## Offline evidence

- Final full suite: 581 passed, 6 skipped (the additional skip is the opt-in
  live test).
- Offline strategy matrix: two chunking strategies, warmup plus two measured
  repetitions, sparse eval, aggregate metrics, leaderboard.
- CLI entrypoints, repository-wide Ruff, compilation, and the updated service
  container build passed; the container exposes `memory-strategy-bench`.
- The eight pre-existing Ruff findings in legacy modules and tests were fixed
  before the final commit.
- Container verification exposed and fixed the installed-layout `cli.py`
  `parents[3]` crash. The root bootstrap now discovers a source checkout and
  safely falls back to the installed package parent; `battle.py` remains
  outside this change.
