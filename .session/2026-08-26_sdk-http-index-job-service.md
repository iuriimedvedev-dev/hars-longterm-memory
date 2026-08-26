# SDK + HTTP index-job service implementation

Date: 2026-08-26

## Outcome

Implemented a tenant-scoped Python SDK and FastAPI service for submitting text
files to create or extend immutable index versions. The service runs the
CPU-only `corpus` engine without an LLM/GPU, or delegates graph construction to
the existing LightRAG indexer on a GPU-capable server.

## Durable surfaces

- `hars_memory/sdk.py`: synchronous SDK for submit, poll, cancel, inspect, and
  checksum-verified artifact download.
- `hars_memory/service/`: API/auth, bounded upload validation, SQLite/PostgreSQL
  repository, local/S3 artifact stores, corpus/LightRAG engines, and a
  lease-based worker.
- `hars-longterm-memory-server`: environment-wired service entrypoint.
- `Dockerfile.service`: non-root local/cloud service image.

Successful publication inserts an immutable version and switches the active
version in one database transaction. Candidate artifacts use unique IDs;
failed/cancelled builds do not replace the last good version. Worker leases
cover input materialization, engine execution, artifact upload, and commit.

## Offline verification

- Installed CLI subprocess: build, status, sparse query, changed/deleted/add
  rebuild.
- Real MCP stdio: initialize, list all seven tools, status, remember, and
  consolidate dry-run.
- Real SDK and HTTP subprocess: authenticated create, terminal idempotent retry,
  wait, download, extend, tenant isolation, process restart, and persisted v2.
- Each journey disabled CUDA and model downloads, pointed LLM URLs at a dead
  endpoint, loaded a `sitecustomize` network tripwire, and asserted no outbound
  connection attempt.

The real LightRAG/LLM path was deliberately not executed while the GPU was
busy; its subprocess command, timeout/cancellation teardown, and Qdrant
per-candidate workspace isolation are covered by focused tests.

Final gates: `569 passed, 5 skipped`; all three offline subprocess E2Es passed;
Ruff and `git diff --check` passed. `Dockerfile.service` built as
`hars-memory-service:local-test`, started successfully, and its in-container
health request returned `{"status":"ok"}`.
