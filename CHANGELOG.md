# Changelog

Derived from the git history. Versions before 0.2.0 are the tagged releases;
`0.2.0` is the version in `pyproject.toml` and collects everything since `v0.1.3`.
Format loosely follows Keep a Changelog.

## [0.2.0] - unreleased (2026-08 to 2026-10)

### Added
- Remote indexing: HTTP index-job service with SQLite/PostgreSQL and
  filesystem/S3 artifact stores, Python SDK, validated index strategies and a
  declarative strategy benchmark (`memory strategy-bench`).
- gRPC server, client and proto (`memory grpc`).
- Unified `memory` CLI: `recall`, `mcp`, `server`, `grpc`, `consolidate`,
  `export`, `import`, `estimate-cost`.
- Multi-project isolation, JWT tokens (EdDSA/HS256), AES-256-GCM keystore,
  revocation and RBAC (`memory auth ...`).
- MCP knowledge tools: `memory_inspect_entity`, `memory_upsert_document`,
  `memory_delete_document`, `memory_sync_status`, `memory_list_projects`.
- Section-level citation accuracy and file/chunk evaluation metrics.
- Structure-aware Markdown chunker (`HARS_MEMORY_CHUNKER=markdown`), chunk
  location metadata, `memory migrate-index`.
- `memory index-batch`: Batch API extraction with resumable phases, stall
  detection, cancel and synchronous fallback.
- Table rows split at cell boundaries; front matter folded into the first chunk.
- `memory_recall` `view=lean` (`HARS_MEMORY_RECALL_VIEW`) and `compact`.
- Opt-in HTTP reranker stage (`HARS_MEMORY_RERANK_BACKEND=off|local|http`).
- Fail-safe guard for deleted-document garbage collection.
- CI quality gates and container image builds.

### Fixed
- Auth: hot-reload of revoked tokens, health-check bypass, passphrase from env.
- gRPC: `context_only` and `debug` marked `optional` for `HasField` presence.

## [0.1.3] - 2026-08-26
- Fix `_PROJECT_ROOT` path bug in `server/index.py` when installed via pip/uv.

## [0.1.2] - 2026-08-26
- Final-review fix wave: path bugs, dead config, removal of machine-specific defaults.

## [0.1.1] - 2026-08-26
- Upper-bound `mcp` dependency to `<2.0.0`.

## [0.1.0] - 2026-08-25
- First standalone release, extracted from a larger research repository:
  package namespace `hars_memory`, MCP server as a console script, public
  Document ingest API, configurable entity schema (YAML), per-project Qdrant
  collection prefixes, `memory regress` AB-bench config flags.

## Pre-0.1 (2026-06 to 2026-08)
- LightRAG graph + vector memory with MCP server (originally `hars-graphrag`).
- Qdrant vector storage, flat dense channel, ripgrep freshness channel.
- BM25 fusion with deterministic ranking, supersession-aware scoring.
- LLM-free corpus index and retrieval-quality evaluation harness.
- Structured logging and consolidation runs.
