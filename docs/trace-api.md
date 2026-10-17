# `memory_trace` — live recall telemetry over SSE (implemented)

Status: **implemented** (2026-10-17). This is the implementation-side contract;
the consumer-side draft lives in `memory-travel/docs/trace-api.md` and is kept in
sync (see "Deviations from the draft" below). Everything here was verified
against the running service (see "Verification").

Memory Travel v1 animates **saved** recall fixtures. v2 animates a recall
**while it executes** by subscribing to this endpoint instead of replaying
fixtures.

## Endpoint

```
GET /trace?question=...&mode=hybrid&top_k=10&fetch_top_k=...&context_only=true&ll_keywords=a,b&hl_keywords=x&project=default
Accept: text/event-stream          (add ?format=json for a buffered JSON body)
```

Behaves exactly like `tools/call memory_recall` with `context_only=true`
(default), except that it emits one SSE frame per pipeline stage **as that stage
finishes**, then a final `result` frame with the full, unchanged `memory_recall`
JSON, then a tiny terminal `done` frame. `format=json` buffers the same events
into `{"ok": true, "trace_id": ..., "events": [{"event": ..., "data": ...}]}`
(handy for non-SSE clients and debugging).

### Transport — why a sibling port, not `/mcp`

The service is a **stdio MCP server** wrapped by **supergateway**
(`--stateful`, `10.66.66.99:8099/mcp`). supergateway is a Node process and
exposes no hook to add an extra HTTP route, so the draft's `GET /mcp/trace`
cannot be hosted there. Instead the MCP process itself starts a small
**read-only HTTP/1.1 + SSE listener** on a sibling port, on the **same event
loop**, sharing `_rag_instance` and the existing caches with the MCP path.

The listener is implemented with `asyncio.start_server` — **no new runtime
dependency**. (`fastapi`/`uvicorn` are an optional extra here; depending on them
in the stdio server would be a silent failure mode after `uv run` prunes extras.)

### Configuration (all optional)

| Env var | Default | Meaning |
| --- | --- | --- |
| `HARS_MEMORY_TRACE_HTTP_PORT` | `8098` | Listener port. `0` or empty disables the listener entirely. |
| `HARS_MEMORY_TRACE_HTTP_HOST` | `HARS_MEMORY_MCP_BIND_HOST`, else `127.0.0.1` | Bind address. In the systemd service `HARS_MEMORY_MCP_BIND_HOST=10.66.66.99`, so the trace endpoint is WireGuard-only like the MCP bridge. |
| `HARS_MEMORY_TRACE_KEEPALIVE_S` | `10` | `: ping` comment interval (rerank can take 7+ s). |
| `HARS_MEMORY_TRACE_HITS_CAP` | `30` | Rows per `*_hits` event. Payload-size only; the full pool still travels in `result`. |

Bind failure (port taken) is logged and non-fatal: the MCP server always starts.

Other routes: `GET /healthz` → `{"ok": true, "service": ..., "trace": true}`;
`OPTIONS` → CORS preflight (`Access-Control-Allow-Origin: *`, so the Vite dev
UI at `:5174` can `EventSource` the endpoint); anything else → 404; a missing
`question` → 400.

## Events (wire order = real execution order)

| # | event | when | payload |
| --- | --- | --- | --- |
| 0 | `trace_start` | recall accepted | `{trace_id, seq, t_ms, question, mode, lightrag_mode, mode_fallback, top_k, fetch_top_k, fetch_top_k_source, alpha, index_id, index_dir, stale_days, last_ingest, context_only, context_priority, ll_keywords, hl_keywords}` |
| 1 | `bm25_hits` | sparse channel done | `{trace_id, seq, t_ms, latency_ms, chunks_indexed, pool_size, hits:[{chunk_id, sparse_score, doc, section}]}` |
| 2 | `vector_hits` | dense channel done | `{trace_id, seq, t_ms, latency_ms, embedder, pool_size, hits:[{chunk_id, dense_score, doc, section}]}` |
| 3 | `fusion` | RRF/alpha blend + diversity done | `{trace_id, seq, t_ms, alpha, candidate_pool_size, pool_size, order:[chunk_id…], scores:{chunk_id: fused}}` |
| 4 | `rerank` | cross-encoder done / skipped | `{trace_id, seq, t_ms, applied, backend, latency_ms, pool, fallback_reason, max_fusion_score, skip_threshold, blocked_inversions, fusion_margin, before:[ids], after:[ids]}` |
| 5 | `graph_expand` | LightRAG entity/relation pass done | `{trace_id, seq, t_ms, mode, latency_ms, entity_count, relation_count, traversed_edge_count, entities:[{name, type}], relations:[{entity1, entity2}], derived_from}` |
| 6 | `context` | prompt context assembled | `{trace_id, seq, t_ms, has_context, chars, context_priority_applied, source_count, entity_count, relation_count, sources:[{reference_id, doc, section}], caps:{max_chars, max_entity_tokens, max_relation_tokens}}` |
| 7 | `result` | final | **today's full `memory_recall` JSON, unchanged** (no trace fields added) |
| 8 | `done` | terminal (success) | `{trace_id, ok, total_ms, has_context}` |
| 8 | `error` | terminal (failure) | `{trace_id, ok:false, total_ms, error_type, error, hint?}` |

Notes:

- `order`/`scores`/`before`/`after` are the **true pre-rerank order**, measured
  where `_compute_diverse` runs, not reconstructed from dense scores — this
  removes the only reconstructed pixel in the v1 demo.
- A skipped rerank still emits event 4 (`applied=false`) with its recorded
  `fallback_reason` (`fusion_confident`, `empty_pool`, …); `latency_ms` is
  `null` when the backend was never called, never a fabricated `0`.
- `result` carries no `trace_id`/`seq`/`t_ms`: it is byte-for-byte the normal
  response (the client can correlate it via the terminal `done.trace_id`).
- A recall whose implementation returns a bare `{ok:false, error…}` (no
  `hybrid`) is surfaced as a terminal `error`, never a fake `result`. A genuine
  "no context" answer still carries `hybrid` and is reported as a normal result.

## Measured vs derived (honesty)

**Measured** (read straight off the pipeline):

- every `latency_ms` (`bm25_hits`, `vector_hits`, `rerank`, `graph_expand`) and
  the terminal `total_ms`;
- hit ids/scores/doc/section/file for both channels; `chunks_indexed`,
  `candidate_pool_size`, `pool_size`; `alpha`; the whole rerank status
  (`applied`, `backend`, `pool`, `fallback_reason`, `max_fusion_score`,
  `skip_threshold`, `blocked_inversions`, `fusion_margin`);
- `order`/`before`/`after` (pre-rerank is captured before rerank mutates the
  pool); `chars`, `has_context`, `context_priority_applied`, the caps;
  `stale_days`, `last_ingest`, `index_id`; `embedder`.

**Derived** (documented, never faked):

- `graph_expand.{entity_count, relation_count, traversed_edge_count, entities,
  relations}` and `context.{source_count, entity_count, relation_count,
  sources}` are **parsed from the rendered LightRAG context string** (the same
  parse `memory-travel/scripts/compile_trace.py::parse_context` uses). On the
  `context_only` path that string is the only place these names exist. The
  `derived_from` field states the source (`"context_string"` /
  `"answer_result"`); the answer path has no rendered context, so it reports
  `relation_count: null` rather than guessing.
- `seq`, `t_ms`, `trace_id` are synthetic (ordering/correlation only).

**Omitted** (cannot be measured at that point):

- a separate `embed` event: embedding latency is not separable from the dense
  `chunks_vdb.query` call and is included in `vector_hits.latency_ms`. The
  draft's `embed`/`query_xy_hint` are dropped (the latter needs a persisted UMAP
  reducer the service does not have).

## Read-only and isolation

- Zero index writes; no new auth surface (same project/`access_token` handling
  as `memory_recall` through the same code path).
- The MCP/CLI path is untouched: an unset `ContextVar` makes every `emit()` a
  no-op. Verified by a test that runs the identical recall with and without an
  active trace and compares the responses.

## Verification

```bash
# after restarting hars-longterm-memory-mcp.service
curl -N 'http://10.66.66.99:8098/trace?question=A2S32%20sidecar%20gate&ll_keywords=A2S32&top_k=3'
curl -s 'http://10.66.66.99:8098/healthz'
HARS_MEMORY_TRACE_URL=http://10.66.66.99:8098/trace uv run pytest -q tests/test_trace_endpoint.py::TestLiveTraceEndpoint
```

`memory-travel` v2 client: append events by name into the existing `TravelTrace`
shape — stage panels consume `vector_hits`/`bm25_hits`/`fusion`/`rerank`/
`graph_expand`/`context`, the answer/grounding panel consumes `result`. No
component change is needed for the fixture schema itself.
