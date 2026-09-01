# Configuration reference

All configuration is via environment variables. No hardcoded defaults for
machine-specific paths — this package does not assume any particular filesystem
layout. The canonical source of truth is `config/.env.example` (176 lines with
inline rationale).

## Required variables

| Variable | Description |
|---|---|
| `HARS_MEMORY_INDEX_DIR` | Path to LightRAG working directory (NetworkX graph + KV store + Qdrant workspace config). No default — must be set explicitly |
| `HARS_MEMORY_STAGING_DIR` | Directory for `memory_remember` note files. Must exist and be writable |

## MCP server

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_EXTRACTOR_BASE_URL` | — | OpenAI-compatible endpoint for entity extraction (GPU). Required for indexing |
| `HARS_MEMORY_EXTRACTOR_MODEL` | — | Model name for extraction, e.g. `Qwen3.6-27B-Q4_K_M` |
| `HARS_MEMORY_QUERY_BASE_URL` | — | OpenAI-compatible endpoint for query LLM. Optional if `context_only=true` (default) |
| `HARS_MEMORY_QUERY_MODEL` | — | Model name for query LLM, e.g. `Qwen3.5-4B` |
| `HARS_MEMORY_EMBED_MODEL` | `embeddinggemma-300m` | Embedding model. Changing requires `--full` reindex |
| `HARS_MEMORY_QUERY_TIMEOUT_SECONDS` | `60` | Timeout for LightRAG `aquery` call |
| `HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER` | `1.0` | Multiplier for `fetch_top_k` when not explicitly passed. Measured optimum 1.0 at `top_k=20` |
| `HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT` | `merged` | Server-wide default for `context_priority` parameter |
| `HARS_MEMORY_SUPERSESSION_SCORING` | `1` | Enable supersession-aware rescoring (`0` to disable) |
| `HARS_MEMORY_HYBRID_ALPHA` | `0.5` | Alpha weight for dense+sparse fusion. Measured optimum 0.5 |
| `HARS_MEMORY_EXTRACTOR_EXTRA_HEADERS` | — | JSON object of extra HTTP headers for the extractor endpoint |
| `HARS_MEMORY_QUERY_EXTRA_HEADERS` | — | JSON object of extra HTTP headers for the query endpoint |

## Vector storage

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_VECTOR_STORAGE` | `NanoVectorDBStorage` | Vector backend: `NanoVectorDBStorage` (file-based, default), `QdrantVectorDBStorage` (recommended for production) |
| `HARS_MEMORY_QDRANT_URL` | `http://localhost:6333` | Qdrant gRPC/HTTP endpoint |
| `HARS_MEMORY_QDRANT_COLLECTION` | — | NOT a Qdrant collection name — LightRAG uses it as the `workspace_id` tenant id written into every payload. Actual collections are always `lightrag_vdb_{chunks,entities,relationships}` |
| `HARS_MEMORY_QDRANT_COLLECTION_PREFIX` | — | Optional prefix for Qdrant collection names. Shared Qdrant instances need distinct prefixes per tenant |
| `HARS_MEMORY_QDRANT_API_KEY` | — | Qdrant API key for authentication |

## Retrieval channels

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_RIPGREP_CHANNEL` | `1` | Enable the ripgrep live-worktree channel (`0` to disable). Searches the live filesystem for exact identifiers |
| `HARS_MEMORY_RIPGREP_ROOTS` | — | Colon-separated paths for ripgrep search roots. Defaults to paths registered in `knowledge-sources.yaml` |
| `HARS_MEMORY_FLAT_CHANNEL` | `0` | Enable the flat-dense channel. Off by default — pays CPU embed cost per query |
| `HARS_MEMORY_BM25_INDEX_PATH` | — | Path to BM25 sparse index. Auto-detected under `HARS_MEMORY_INDEX_DIR` if unset |
| `HARS_MEMORY_BM25_INDEX_ID` | — | BM25 index identifier for multi-tenant BM25. Defaults to `default` |

## Indexing

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_CHUNK_TOKEN_SIZE` | `512` | Token budget per chunk (tiktoken `cl100k_base`). Overridden by `CHUNKER_CHUNK_SIZE` for legacy compat |
| `HARS_MEMORY_CHUNK_OVERLAP_TOKENS` | `128` | Token overlap between adjacent chunks |
| `HARS_MEMORY_GPU_GUARD_SCRIPT_PATH` | — | Path to a Python file exposing `assert_gpu_free(api_base_url)`. Loaded dynamically; if unset, no GPU-concurrency check is performed |
| `HARS_MEMORY_ENTITY_SCHEMA_PATH` | — | Path to custom entity/relation schema YAML. Default uses `hars_memory/schema/default_schema.yaml` |
| `HARS_MEMORY_LEGACY_ENV_PREFIXES` | — | Comma-separated forbidden env prefixes (e.g. `GRAPHRAG_`). Fails closed at startup if any env var with that prefix is set |

## Index service (HTTP server)

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_API_KEYS_JSON` | — | JSON object mapping API keys to tenant IDs. Required for the HTTP service. Example: `{"key1":"tenant-a","key2":"tenant-b"}` |
| `HARS_MEMORY_SERVICE_HOST` | `0.0.0.0` | HTTP service bind address |
| `HARS_MEMORY_SERVICE_PORT` | `8787` | HTTP service port |
| `HARS_MEMORY_SERVICE_DATABASE_URL` | `sqlite:///./hars-memory-service.sqlite3` | Job database URL. Postgres: `postgresql+psycopg://user:pass@host/db` |
| `HARS_MEMORY_ARTIFACT_STORE_URL` | `file:///./hars-memory-artifacts` | Artifact store URL. S3: `s3://bucket/prefix` |
| `HARS_MEMORY_S3_ENDPOINT_URL` | — | Custom S3 endpoint (for MinIO, etc.) |
| `HARS_MEMORY_S3_REGION` | — | S3 region for artifact store |
| `HARS_MEMORY_WORKER_SCRATCH_DIR` | `/tmp/hars-memory-worker-scratch` | Scratch directory for index workers |
| `HARS_MEMORY_WORKER_POLL_SECONDS` | `5` | Worker poll interval for new jobs |
| `HARS_MEMORY_WORKER_LEASE_SECONDS` | `30` | Job lease duration before another worker can claim it |

## Logging

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `HARS_MEMORY_LOG_FILE` | — | Optional log file path. If unset, logs to stderr |
| `HARS_MEMORY_LOG_QUERY_EVENTS` | `1` | Enable structured JSON query event logging |
| `HARS_MEMORY_LOG_QUERY_EVENTS_DIR` | `/tmp/hars-memory-query-events` | Directory for query event log files |

## Deprecated / legacy

| Variable | Status | Description |
|---|---|---|
| `CHUNKER_CHUNK_SIZE` | Deprecated | Use `HARS_MEMORY_CHUNK_TOKEN_SIZE` |
| `CHUNKER_OVERLAP_SIZE` | Deprecated | Use `HARS_MEMORY_CHUNK_OVERLAP_TOKENS` |
| `HARS_EMBED_MODEL` | Deprecated | Use `HARS_MEMORY_EMBED_MODEL` |
| `GRAPHRAG_*` | Forbidden | Legacy prefix — server fails closed if detected. Set `HARS_MEMORY_LEGACY_ENV_PREFIXES=GRAPHRAG_` to enforce |

## Variable naming convention

- `HARS_MEMORY_*` — current prefix for all env vars
- `HARS_*` (without `MEMORY`) — deprecated fallback, read if `HARS_MEMORY_*` is unset
- `CHUNKER_*` — legacy chunker config, deprecated in favour of `HARS_MEMORY_CHUNK_*`
- `GRAPHRAG_*` — legacy prefix, forbidden — the server refuses to start if any
  env var with this prefix is set (see `HARS_MEMORY_LEGACY_ENV_PREFIXES`)