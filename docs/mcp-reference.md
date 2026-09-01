# MCP tool reference

The `hars-longterm-memory` package exposes 7 MCP tools. All tools are registered
in `.mcp.json` and launched via `uv run --project <project> memory-mcp` (or the
standalone `hars-longterm-memory-mcp` entrypoint).

All tools use `stdio` transport. The server is CPU-only for queries — no GPU
required after indexing.

## Tool overview

| Tool | Purpose | GPU-free | Admin |
|---|---|---|---|
| `memory_recall` | Query the knowledge graph (primary workhorse) | Yes | No |
| `memory_remember` | Save a knowledge note into staging | Yes | No |
| `memory_entities` | Search entities by name + 1-hop neighbourhood | Yes | No |
| `memory_related` | N-hop subgraph traversal from an entity | Yes | No |
| `memory_status` | Index health, freshness, source breakdown | Yes | No |
| `memory_consolidate` | Trigger incremental reindex | No (if GPU guard configured) | Yes |
| `memory_forget` | Purge stale documents by date | Yes | Yes |

---

## memory_recall

Primary retrieval tool. Returns relevant context from the knowledge graph for a
natural language question. Supports four retrieval modes with an additive
dense+BM25 sparse fusion channel layered on top.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | `string` | **required** | Natural language question |
| `mode` | `enum` | `hybrid` | Retrieval mode: `local` (entity-neighbourhood), `global` (community themes), `hybrid` (both), `naive` (pure vector, automatic fallback when no keywords) |
| `top_k` | `integer` | `20` | Result count: entities/relations/chunks returned. Range 1–50 |
| `fetch_top_k` | `integer` | — | Fetch width before truncation. Must be >= top_k. Defaults to `top_k * HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER` (env-configurable, default 1.0) |
| `ll_keywords` | `string[]` | `[]` | Low-level keywords: specific entities, codes, names. When supplied, skips LightRAG's keyword-extraction LLM call entirely |
| `hl_keywords` | `string[]` | `[]` | High-level keywords: themes/concepts. Same as ll_keywords — both must be empty for keyword extraction to run |
| `context_only` | `boolean` | `true` | Return raw graph context without running the local answer LLM. Default `true` — the calling agent synthesises the answer |
| `context_priority` | `enum` | `merged` | Context ordering: `merged` (round-robin fusion+LightRAG with supersession rescoring), `lightrag` (original LightRAG order). Only affects `context_only=true` |
| `debug` | `boolean` | `false` | Include debug data: `fused_chunks`, `latency_breakdown`, `llm_usage`, `raw_result` |

### Response shape

```json
{
  "ok": true,
  "context": "Document Chunks section with [ref:1] references...",
  "hybrid": {
    "enabled": true,
    "fused_chunks": [
      {
        "file_path": "k8s-networking-gke-ingress.md",
        "content": "## GKE Ingress\n\nBody text...",
        "snippet": "Truncated preview...",
        "score": 0.85,
        "section": "## GKE Ingress"
      }
    ],
    "identifier_matches": ["A2S32", "hyp:abc"],
    "ripgrep": {
      "enabled": true,
      "available": true,
      "hits_count": 3,
      "query_terms": ["A2S32"]
    },
    "confidence": {
      "low_confidence": false
    },
    "latency_ms": {
      "dense_channel": 450,
      "sparse_channel": 0.5
    }
  },
  "citations": [
    {
      "node_id": "chunk:abc123",
      "source_path": "k8s-networking-gke-ingress.md",
      "snippet": "Truncated preview...",
      "score": 0.85,
      "section": "## GKE Ingress"
    }
  ],
  "entities_used": ["GKE Ingress", "Kubernetes Networking"],
  "mode_fallback": null,
  "mode": "hybrid",
  "context_priority_applied": "merged",
  "last_ingest": "2026-08-25",
  "stale_days": 7
}
```

### Key behaviours

- **Keyword fallback**: when both `ll_keywords` and `hl_keywords` are omitted,
  the query silently runs in `naive` mode — the `mode_fallback` field explains
  when this fired
- **Hybrid field**: the `hybrid` field is always present, even without
  `debug=true`. It contains fused dense+BM25 results, identifier matches from
  the BM25 channel, and the ripgrep live-worktree channel status
- **Ripgrep channel** (default on): searches the LIVE worktree for exact
  identifiers from the question and `ll_keywords`. Catches files added/edited
  since the last consolidation that the index has never seen
- **Flat-dense channel** (default off): additive channel covering chunks in the
  index's raw chunk store but missing/stale in the primary vector store
- **Supersession rescoring** (default on): re-ranks merged results so that
  superseded/outdated documents rank lower
- **Section field**: each `fused_chunks` and `citations` entry includes a
  `section` field with the breadcrumb heading chain (e.g. `# K8s > ## Networking`)
  for section-level citation accuracy

### Latency profile

| Metric | Value |
|---|---|
| p50 | ~0.7 s |
| p90 | ~0.8 s |
| p95 | ~0.8 s |
| p99 | ~0.8 s |
| max | ~15 s (cold-start, first query after index load) |

Measured on 100 hard queries with 8.1 unique files per query on average.

---

## memory_remember

Save a knowledge note into the long-term memory staging area. Notes accumulate
as markdown files and are merged into the knowledge graph by the next KB update
run. No LLM, no GPU — instant file write.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `title` | `string` | **required** | Short kebab-case slug for the note filename |
| `content` | `string` | **required** | The knowledge itself — self-contained markdown prose |
| `importance` | `enum` | `normal` | `critical`, `normal`, or `low` |
| `tags` | `string[]` | `[]` | Topic tags, e.g. `['vea', 'b1', 'rocm']` |

### Response shape

```json
{
  "ok": true,
  "path": "/path/to/staging/2026-08-25_my-note-slug.md",
  "note": "Note saved to path"
}
```

### Key behaviours

- Writes to `HARS_MEMORY_STAGING_DIR` (required env var)
- Filename is prefixed with the current date
- Tags are written as YAML frontmatter in the markdown file
- No LLM call, no GPU — purely a filesystem write

---

## memory_entities

Search graph entities by name or alias. Returns matching nodes + their 1-hop
neighbourhood.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `string` | **required** | Entity name or partial alias |
| `limit` | `integer` | `10` | Max results. Range 1–50 |

### Response shape

```json
{
  "ok": true,
  "entities": [
    {
      "entity_id": "hyp:abc123",
      "entity_name": "Phase C hypothesis",
      "type": "hypothesis",
      "description": "Detailed description...",
      "neighbours": [
        {"entity_id": "exp:42", "entity_name": "Experiment 42", "relation": "produces"}
      ]
    }
  ]
}
```

### Key behaviours

- **INTROSPECTION, unstable**: this tool's schema may change in future versions
- Partial name matching — searches by name prefix and alias
- 1-hop neighbourhood is included for each matched entity

---

## memory_related

Return a compact node/edge list for targeted graph traversal, starting from a
single entity.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `entity_id` | `string` | **required** | Stable entity ID (e.g. `hyp:abc`, `exp:42`) |
| `hops` | `integer` | `1` | Traversal depth. Range 1–1 (see note below) |

### Response shape

```json
{
  "ok": true,
  "nodes": [
    {"id": "hyp:abc", "label": "Phase C hypothesis", "type": "hypothesis"}
  ],
  "edges": [
    {"source": "hyp:abc", "target": "exp:42", "label": "produces"}
  ],
  "truncated": false,
  "dropped_nodes": 0,
  "dropped_edges": 0
}
```

### Key behaviours

- **1 hop only**: `hops>1` is not supported — the induced subgraph explodes
  combinatorially (measured 2 hops = 160 KB, 3 hops = 2.1 MB). Even 1 hop on a
  hub node is capped at ~500 nodes / ~2000 edges
- **INTROSPECTION, unstable**: this tool's schema may change in future versions
- The response includes `truncated`, `dropped_nodes`, and `dropped_edges` fields
  when the result exceeds the budget

---

## memory_status

Return index freshness, node/edge counts, last ingest time, source breakdown,
and configured models. Call this first to confirm the index exists before
querying. GPU-free and always available.

### Parameters

None.

### Response shape

```json
{
  "ok": true,
  "index_exists": true,
  "node_count": 1500,
  "edge_count": 4200,
  "chunk_count": 8500,
  "last_ingest": "2026-08-25T14:30:00",
  "stale_days": 7,
  "staleness_warning": "Knowledge graph last ingested 2026-08-25 (7 days ago)...",
  "sources": {
    "kb": 120,
    "repos/sre-docs": 45,
    ".plans": 30
  },
  "models": {
    "extractor": "Qwen3.6-27B",
    "query_llm": "Qwen3.5-4B",
    "embedder": "embeddinggemma-300m"
  }
}
```

### Key behaviours

- Always available, even mid-training — CPU embeddings only
- Returns `staleness_warning` if the index is older than 30 days
- `index_exists: false` if LightRAG workspace is not found
- Source breakdown shows how many documents per source path

---

## memory_consolidate

Trigger incremental ingest (admin). Walks the configured paths, detects changed
files (via content fingerprint sidecar), and reindexes them. GPU-guarded when
`HARS_MEMORY_GPU_GUARD_SCRIPT_PATH` is configured.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `paths` | `string[]` | `[".plans", "docs"]` | Paths to (re)index |
| `since` | `string` | — | ISO8601 timestamp — only reindex files modified after this |
| `dry_run` | `boolean` | `true` | Walk + count only, no LLM calls. Default `true` for safety |

### Response shape

```json
{
  "ok": true,
  "dry_run": true,
  "added": 5,
  "changed": 3,
  "deleted": 1,
  "skipped": 120,
  "total": 129,
  "paths": [".plans", "docs"],
  "since": null,
  "details": {
    "added": ["file1.md", "file2.md"],
    "changed": ["file3.md"],
    "deleted": ["file4.md"]
  }
}
```

### Key behaviours

- **GPU guard**: if `HARS_MEMORY_GPU_GUARD_SCRIPT_PATH` is set, blocked while a
  GPU-exclusive workflow is running. If unset, proceeds unconditionally
- **Fingerprint sidecar**: content fingerprints are stored alongside the index.
  Only files whose content hash changed are reindexed
- **Deleted document GC**: documents removed from the filesystem are deleted
  from the index, BM25 index, and chunk store
- **`dry_run=true`** (default): walks and counts, no LLM calls, no index changes

---

## memory_forget

Purge stale documents from the long-term memory index by document date, with
keyword protection for knowledge that must survive. GPU-free, no LLM calls —
pure KV-store scan + LightRAG `adelete_by_doc_id`.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `before` | `string` | — | ISO date (YYYY-MM-DD). Docs with `Date:` header strictly before this are candidates. Mutually exclusive with `older_than_days` |
| `older_than_days` | `integer` | — | Docs older than this many days (from today) are candidates. Mutually exclusive with `before` |
| `protect` | `string[]` | `[]` | Regex patterns (case-insensitive); a candidate matching ANY pattern is never deleted. Required for `apply=true` |
| `sections` | `string[]` | all | Only consider these Section header values (e.g. `['session', 'db']`) |
| `apply` | `boolean` | `false` | `false` = dry-run report only. `true` = actually delete — refused without `protect` or `confirm_unprotected=true` |
| `confirm_unprotected` | `boolean` | `false` | Explicit opt-out of the `protect`-pattern requirement |

### Response shape

```json
{
  "ok": true,
  "applied": false,
  "dry_run": true,
  "candidates": [
    {
      "doc_id": "old-session-2025-01-01.md",
      "date": "2025-01-01",
      "section": "session",
      "protected": false,
      "deleted": false
    }
  ],
  "total_candidates": 5,
  "total_protected": 2,
  "total_deleted": 0
}
```

### Key behaviours

- **Docs without a `Date:` header**: documents with an unrecognised/`unknown`
  header date are NEVER deleted, unconditionally — not affected by any argument
- **Guardrails**: `apply=true` is REFUSED unless at least one `protect` pattern
  is supplied OR `confirm_unprotected=true` is passed. This prevents an
  unqualified date cutoff from silently wiping an entire section
- **`apply=false`** (default): dry-run only — lists candidates, nothing deleted
- Supply exactly one of `before` / `older_than_days`

---

## Common response fields

Every tool response includes:

| Field | Type | Description |
|---|---|---|
| `ok` | `boolean` | Success indicator. `false` with `error` string on failure |

## Error handling

All tools return errors as `{"ok": false, "error": "description"}`. Common
error scenarios:

- **Missing required env var**: `HARS_MEMORY_INDEX_DIR` is required and has no
  default — the server fails closed at startup
- **Index not found**: `memory_recall` returns `ok: false` with an error
  explaining the index hasn't been built yet
- **GPU guard blocks**: `memory_consolidate` returns `ok: false` when a
  GPU-exclusive workflow is running
- **MCP server unavailable**: `mcp` package (1.x) must be installed — the
  graceful stub fallback raises `RuntimeError` with a clear message
- **Legacy env prefix detected**: the server refuses to start if legacy env
  vars (e.g. `GRAPHRAG_*`) are set — see `HARS_MEMORY_LEGACY_ENV_PREFIXES`