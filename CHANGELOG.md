# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

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
