# Configuration

All settings are environment variables. `config/.env.example` is an annotated
template; copy it to `.env` (git-ignored). Defaults below are the code defaults.
"Required" means the server refuses to start without it.

## Paths and storage

| Variable | Default | Notes |
|---|---|---|
| `HARS_MEMORY_INDEX_DIR` | required | LightRAG working directory |
| `HARS_MEMORY_STAGING_DIR` | required (MCP) | where `memory_remember` writes notes |
| `HARS_MEMORY_BM25_CACHE_DIR` | `<index>/../bm25_cache` | BM25 index location |
| `HARS_MEMORY_VECTOR_STORAGE` | `NanoVectorDBStorage` | or `QdrantVectorDBStorage` |
| `HARS_MEMORY_GRAPH_STORAGE` | `NetworkXStorage` | |
| `HARS_MEMORY_QDRANT_URL` | `http://localhost:6335` | |
| `HARS_MEMORY_QDRANT_COLLECTION` | `hars_longterm_memory` | used as the workspace/tenant id, not the collection name |
| `HARS_MEMORY_FINGERPRINT_STORE` | `<index>/doc_fingerprints.json` | change-detection sidecar |
| `HARS_MEMORY_CATALOG_PATH` | package default | document catalog |
| `HARS_MEMORY_SOURCES_MANIFEST` | unset | knowledge-sources manifest, see [indexing.md](indexing.md) |
| `HARS_MEMORY_SOURCES_MANIFEST_LOCAL` | `meta/knowledge-sources.local.yaml` next to manifest | machine-local additions |
| `HARS_MEMORY_PROJECTS_CONFIG`, `HARS_MEMORY_PROJECTS_DIR` | `~/.local/share/hars-longterm-memory/projects` | multi-project registry |

## LLMs

| Variable | Default | Notes |
|---|---|---|
| `HARS_MEMORY_EXTRACTOR_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible endpoint used during indexing |
| `HARS_MEMORY_EXTRACTOR_MODEL` | none | served model alias |
| `HARS_MEMORY_EXTRACTOR_TEMPERATURE` | `0.1` | |
| `HARS_MEMORY_EXTRACTOR_MAX_TOKENS` | see `.env.example` | max output tokens per chunk |
| `HARS_MEMORY_QUERY_BASE_URL` | `http://localhost:8081/v1` | endpoint for keyword extraction / answers |
| `HARS_MEMORY_QUERY_MODEL` | none | |
| `HARS_MEMORY_QUERY_TEMPERATURE` | `0.1` | |
| `HARS_MEMORY_QUERY_MAX_TOKENS` | `2048` | |
| `HARS_MEMORY_QUERY_TIMEOUT_SECONDS` | see `.env.example` | |
| `HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER` | `1.0` | default `fetch_top_k = top_k * multiplier` |
| `HARS_MEMORY_LLM_API_KEY` | `not-needed` | API key for hosted providers; never commit it |
| `HARS_MEMORY_LLM_TIMEOUT_SECONDS` | `8` in the template | per-request timeout |
| `HARS_MEMORY_LLM_RETRY_ATTEMPTS` | `3` | |
| `HARS_MEMORY_LLM_RETRY_BACKOFF_INITIAL_SECONDS` / `_MAX_SECONDS` | `1.0` / `8.0` | |
| `HARS_MEMORY_LLM_MAX_TOKEN_SIZE` | `30720` | cap on the extraction prompt input; keep below the server's per-slot context |
| `HARS_MEMORY_LLM_MAX_ASYNC` | `4` | concurrent extraction calls |
| `HARS_MEMORY_EXTRACTION_LANGUAGE` | `English` | |
| `HARS_MEMORY_MAX_GLEANING` | `1` | extra extraction pass (LightRAG does at most one) |

## Embedder

| Variable | Default | Notes |
|---|---|---|
| `HARS_MEMORY_EMBED_MODEL` | `unsloth/embeddinggemma-300m` | any sentence-transformers model; changing it changes vector dimensions, so reindex fully |
| `HARS_MEMORY_EMBED_DEVICE` | `cpu` | |
| `HARS_MEMORY_EMBED_BATCH_SIZE` | `32` | |
| `HARS_MEMORY_EMBED_MAX_TOKENS` | chunk token size | |
| `HARS_MEMORY_EMBED_DOC_PREFIX` | empty | e.g. `passage: ` for e5 models |
| `HARS_MEMORY_EMBED_PROMPT_NAME` / `_QUERY_PROMPT_NAME` | empty / `query` | sentence-transformers prompt names |
| `HARS_MEMORY_EMBED_LOCAL_FILES_ONLY` | `1` in the template | do not download models at runtime |
| `HF_HOME` | Hugging Face default | model cache |

## Chunking and indexing

| Variable | Default | Notes |
|---|---|---|
| `HARS_MEMORY_CHUNKER` | `token` | `token` or `markdown` |
| `HARS_MEMORY_CHUNK_TOKEN_SIZE` | `512` | max tokens per chunk |
| `HARS_MEMORY_CHUNK_OVERLAP_TOKENS` | `64` | not applied in markdown mode |
| `HARS_MEMORY_CHUNK_MIN_TOKENS` | `200` | markdown mode: merge smaller chunks into a neighbour; `0` disables |
| `HARS_MEMORY_INSERT_BATCH_SIZE` | `10` | |
| `HARS_MEMORY_MAX_PARALLEL_INSERT` | `2` | |
| `HARS_MEMORY_BATCH_STALL_MINUTES` | `20` | `index-batch run` stall detection; `0` disables |
| `HARS_MEMORY_BATCH_CANCEL_WAIT_MINUTES` | `15` | |
| `HARS_MEMORY_CLAUDE_MEMORY_DIR` | unset | optional extra ingest root |
| `HARS_MEMORY_GPU_GUARD_SCRIPT_PATH` | unset | optional Python file exposing `assert_gpu_free(api_base_url)`, called before non-dry-run `memory_consolidate` |

## Retrieval

| Variable | Default | Notes |
|---|---|---|
| `HARS_MEMORY_HYBRID_ENABLED` | `1` | additive dense+BM25 `hybrid` block |
| `HARS_MEMORY_HYBRID_ALPHA` | `0.5` | `alpha*dense + (1-alpha)*sparse` |
| `HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT` | `merged` | or `lightrag` |
| `HARS_MEMORY_RECALL_VIEW` | `full` | or `lean` |
| `HARS_MEMORY_QUERY_DEFAULT_TOP_K` | `6` | |
| `HARS_MEMORY_RIPGREP_CHANNEL` | `1` | live-file channel on/off |
| `HARS_MEMORY_RIPGREP_ROOTS` | unset | comma-separated absolute roots; unset disables the channel unless a manifest supplies roots |
| `HARS_MEMORY_FLAT_CHANNEL` | `0` | flat dense channel |
| `HARS_MEMORY_FLAT_DENSE_CACHE_DIR` | `<index parent>/flat_dense_cache` | |
| `HARS_MEMORY_FUSION_AGREEMENT_BONUS` | `0.05` | bonus for documents found by several channels |
| `HARS_MEMORY_FUSION_TIE_EPSILON` | see `fusion.py` | tie-break threshold |
| `HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL` | see `fusion.py` | single-channel tie-break signal |
| `HARS_MEMORY_SUPERSESSION_SCORING` | `1` | demote stale/deprecated documents |
| `HARS_MEMORY_SUPERSESSION_MARKER_PENALTY` | `1` | penalise self-declared "DEPRECATED" documents |
| `HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT` | `0` | measured harmful; leave off |

### Reranking

| Variable | Default | Notes |
|---|---|---|
| `HARS_MEMORY_RERANK_BACKEND` | unset | `off`, `local`, `http`; unset keeps legacy local behaviour |
| `HARS_MEMORY_RERANK_HTTP_URL` | `http://127.0.0.1:8081/v1/rerank` | |
| `HARS_MEMORY_RERANK_HTTP_MODEL` | `x` | model id sent in the request |
| `HARS_MEMORY_RERANK_POOL` | `20` | candidates reranked |
| `HARS_MEMORY_RERANK_TIMEOUT_S` | `8` | falls back to fused order on failure |
| `HARS_MEMORY_RERANK_MODEL` | empty | local cross-encoder model |
| `HARS_MEMORY_RERANK_DEVICE` / `_BATCH_SIZE` / `_LOCAL_FILES_ONLY` | `cpu` / `1` / `1` | local backend |
| `HARS_MEMORY_MIN_RERANK_SCORE` | `0.0` | |

## Auth

See [auth.md](auth.md): `HARS_MEMORY_AUTH_ENABLED` (`0`), `HARS_MEMORY_MASTER_KEY`,
`HARS_MEMORY_AUTH_PASSPHRASE`, `HARS_MEMORY_KEYSTORE_PATH`,
`HARS_MEMORY_TOKENS_CONFIG`, `HARS_MEMORY_ACCESS_TOKEN`.

## Services

| Variable | Default | Notes |
|---|---|---|
| `HARS_MEMORY_API_KEYS_JSON` | required for the HTTP service | JSON map of API key to tenant id |
| `HARS_MEMORY_SERVICE_DATABASE_URL` | `sqlite:///./hars-memory-service.sqlite3` | or PostgreSQL |
| `HARS_MEMORY_ARTIFACT_STORE_URL` | `file:///tmp/hars-memory-artifacts` | or `s3://bucket/prefix` |
| `HARS_MEMORY_S3_ENDPOINT_URL`, `HARS_MEMORY_S3_REGION` | unset | S3-compatible stores |
| `HARS_MEMORY_SERVICE_HOST` / `_PORT` | `127.0.0.1` / `8787` | |
| `HARS_MEMORY_WORKER_POLL_SECONDS`, `HARS_MEMORY_WORKER_SCRATCH_DIR` | `1`, unset | |
| `HARS_MEMORY_MAX_UPLOAD_FILES` / `_FILE_BYTES` / `_TOTAL_BYTES` | `100` / `2097152` / `20971520` | |
| `HARS_MEMORY_GRPC_HOST` / `_PORT` | `0.0.0.0` / `8788` | |
| `HARS_MEMORY_GRPC_MAX_WORKERS` / `_MAX_MESSAGE_SIZE` | `10` / `4194304` | |

## Logging

`HARS_MEMORY_LOG_LEVEL` (`INFO`), `HARS_MEMORY_LOG_FILE`, `HARS_MEMORY_LOG_MAX_BYTES`,
`HARS_MEMORY_LOG_BACKUP_COUNT`, `HARS_MEMORY_EVENT_LOG_FILE` (structured
per-query events), `HARS_MEMORY_EVENT_LOG_MAX_BYTES`, `HARS_MEMORY_EVENT_LOG_BACKUP_COUNT`.

## Example: minimal local setup

```bash
export HARS_MEMORY_INDEX_DIR=$HOME/.hars-memory/index
export HARS_MEMORY_STAGING_DIR=$HOME/.hars-memory/staging
export HARS_MEMORY_EXTRACTOR_BASE_URL=http://127.0.0.1:8080/v1
export HARS_MEMORY_EXTRACTOR_MODEL=my-extractor
export HARS_MEMORY_QUERY_BASE_URL=http://127.0.0.1:8081/v1
export HARS_MEMORY_QUERY_MODEL=my-query-model
export HARS_MEMORY_CHUNKER=markdown
```
