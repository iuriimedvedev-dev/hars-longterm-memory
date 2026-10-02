# MCP tools

The server (`memory-mcp` / `hars-longterm-memory-mcp`, stdio) exposes 12 tools.
Every tool also accepts two optional common arguments:

| Arg | Default | Description |
|---|---|---|
| `project` | `default` | project id for multi-project isolation |
| `access_token` | `HARS_MEMORY_ACCESS_TOKEN` | JWT or static token (only checked when `HARS_MEMORY_AUTH_ENABLED=1`) |

Responses are JSON with an `ok` flag; failures carry an `error` string.

| Tool | Purpose |
|---|---|
| `memory_recall` | Retrieve context for a question (main tool) |
| `memory_remember` | Save a note into the staging directory |
| `memory_entities` | Find entities by name plus 1-hop neighbours |
| `memory_related` | N-hop subgraph around an entity |
| `memory_inspect_entity` | Details of one entity |
| `memory_status` | Index health and freshness |
| `memory_consolidate` | Incremental re-index (admin) |
| `memory_forget` | Purge documents by date (admin) |
| `memory_upsert_document` | Index or update one document |
| `memory_delete_document` | Remove one document |
| `memory_sync_status` | Compare disk against the index |
| `memory_list_projects` | Projects visible to the caller |

## memory_recall

| Arg | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | natural-language question |
| `mode` | `local` / `global` / `hybrid` / `naive` | `hybrid` | graph mode; becomes `naive` automatically when no keywords are given |
| `top_k` | int 1-50 | `6` (`HARS_MEMORY_QUERY_DEFAULT_TOP_K`) | result count |
| `fetch_top_k` | int | `top_k * multiplier` | candidate width before truncation (>= `top_k`) |
| `ll_keywords` | string[] | `[]` | specific entities/codes; skips the keyword-extraction LLM call when keywords are supplied |
| `hl_keywords` | string[] | `[]` | themes/concepts |
| `context_only` | bool | `true` | return retrieved context without a generated answer; the calling agent synthesises |
| `context_priority` | `merged` / `lightrag` | `merged` | context ordering (see [retrieval.md](retrieval.md)) |
| `compact` | bool | `false` | omit `hybrid` text that repeats other fields (every `snippet`, and chunk `content` already inside `context`); ids and location fields are kept |
| `view` | `full` / `lean` | `full` (`HARS_MEMORY_RECALL_VIEW`) | `lean` drops the large `context` blob and duplicate/debug fields, keeps chunks with location metadata. `debug=true` always returns full detail |
| `debug` | bool | `false` | add `fused_chunks`, `latency_breakdown`, `llm_usage`, `raw_result` |

Abridged `view=lean` response (field set varies with the query):

```json
{
  "ok": true,
  "view": "lean",
  "hybrid": {
    "enabled": true,
    "confidence": "high",
    "rerank": {"backend": "http", "applied": true, "latency_ms": 420, "pool": 20},
    "fused_chunks": [
      {
        "chunk_id": "chunk-ab12...",
        "content": "| Parameter | Value | ...",
        "score": 0.83,
        "source_path": "kb/billing/overview.md",
        "heading_path": ["Billing", "Ownership"],
        "start_line": 12,
        "end_line": 31
      }
    ]
  },
  "omitted": ["context", "hybrid.fused_chunks[].snippet"]
}
```

`omitted` lists what was dropped; re-request with `view="full"` to get it.
`hybrid.rerank` reports `backend`, `applied`, `latency_ms`, `fallback_reason`
and `pool` when a reranker is configured. Ripgrep matches appear under
`hybrid.ripgrep_hits` when that channel is on.

## memory_remember

| Arg | Type | Default | Description |
|---|---|---|---|
| `title` | string | required | kebab-case slug for the file name |
| `content` | string | required | self-contained markdown |
| `importance` | `critical` / `normal` / `low` | `normal` | |
| `tags` | string[] | | written as front matter |

Writes a dated `.md` file into `HARS_MEMORY_STAGING_DIR`; it becomes searchable
after the next consolidation. Response: `{"ok": true, "path": "...", "note": "..."}`.

## memory_entities / memory_related / memory_inspect_entity

| Tool | Args |
|---|---|
| `memory_entities` | `name` (required), `limit` (1-50, default 10) |
| `memory_related` | `entity_id` (required), `hops` (default and max: server `SUBGRAPH_MAX_HOPS`) |
| `memory_inspect_entity` | `name` (name or id, required) |

## memory_status

No arguments. Returns `index_exists`, node/edge/chunk counts, `last_ingest`,
staleness warning (older than 30 days), per-source document counts and the
configured models.

## memory_consolidate (admin)

| Arg | Default | Description |
|---|---|---|
| `paths` | configured ingest roots | paths to ingest |
| `since` | | ISO8601; only files modified after this |
| `dry_run` | `true` | walk and count only, no LLM calls |

## memory_forget (admin)

Dry-run by default.

| Arg | Default | Description |
|---|---|---|
| `before` | | ISO date; documents dated strictly before it |
| `older_than_days` | | alternative age filter |
| `protect` | `[]` | case-insensitive regexes; matching file names are never deleted |
| `sections` | | only consider these section header values |
| `apply` | `false` | actually delete; refused without `protect` unless `confirm_unprotected` |
| `confirm_unprotected` | `false` | explicit opt-out of the `protect` requirement |

## Document tools

| Tool | Args |
|---|---|
| `memory_upsert_document` | `file_path` (required), `content` (optional override of disk content) |
| `memory_delete_document` | `file_path` or `doc_id` |
| `memory_sync_status` | `paths` (optional; defaults to ingest roots of the sources manifest) |
| `memory_list_projects` | none |

See also [mcp-reference.md](mcp-reference.md) for longer response examples
(some details there predate the newest tools).
