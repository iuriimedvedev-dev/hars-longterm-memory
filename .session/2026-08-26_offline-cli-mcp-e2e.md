# 2026-08-26 — Offline CLI and MCP E2E

## Scope

Verified the standalone package's installed CLI and MCP stdio transport while
the GPU was busy. No LightRAG extraction/query LLM was started and no durable
graph index was mutated.

## Results

- `uv sync --extra dev` installed `torch==2.12.0+cpu`.
- Full suite: **532 passed, 5 skipped**, one Qdrant compatibility warning.
- Real CLI subprocess journey passed with extractor/query URLs pointed at
  `127.0.0.1:9`: build two text documents, status, exact sparse query, then an
  unchanged incremental rebuild (`added=0 changed=0 unchanged=2 deleted=0`).
- Real MCP JSON-RPC-over-stdio journey passed through the official MCP client:
  initialize, exact seven-tool discovery, `memory_status` on an absent temporary
  index, `memory_remember` into temporary staging, and
  `memory_consolidate(dry_run=true)` over one Markdown fixture.
- The MCP journey ran with a `sitecustomize.py` network guard that raises on
  `socket.connect`, `connect_ex`, and `create_connection`; no network-attempt
  sentinel was written. The nested consolidation process inherited the guard
  and reported `DRY RUN complete — no LLM extraction performed`.

## Architecture gap found

Standalone `main` at `777d948719da209973b3b7adabcebd771a8d62bb` has no HTTP
server, SDK, upload endpoint, index-job model, artifact export/download API, or
pluggable local/cloud job database. The only remote branch is `main`, and a
cross-workspace/history search found no omitted implementation. The existing
`hars_memory.ingest.api.ingest_documents()` is synchronous and in-process; it
is not a network SDK.

Before cloud E2E can run, recover a missing branch/commit if one exists outside
the inspected GitLab repository, or implement the SDK/API/worker/artifact layer
as a new capability with an explicit contract.
