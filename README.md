# HARS Long-Term Memory

Local, model-agnostic long-term memory service for the HARS/Cortex project.
Serves the `hars-longterm-memory` MCP server (`memory_recall`, `memory_remember`,
`memory_forget`, `memory_consolidate`, `memory_status`, and the introspection-tier
`memory_entities` / `memory_related`). LightRAG graph+vector retrieval is one
retrieval channel among several (see `retrieval/` for BM25 fusion) — the public
tool surface never names the underlying mechanism.

**Stack**: LightRAG (lightrag-hku 1.4.16) · unsloth/embeddinggemma-300m (768-dim) on CPU · NanoVectorDB (file-backed, migrating to Qdrant — see `.plans/2026-07-29_graphrag-qdrant-migration.md`) · NetworkX · Qwen3.6-27B extractor · Qwen3.5-4B query LLM.

**Package manager**: [UV](https://docs.astral.sh/uv/) — standalone project at `tools/memory/` with its own `pyproject.toml` + `uv.lock`. Fully isolated from the main workspace and the ROCm training venv.

---

## Quick start

### 1. Install dependencies

```bash
# From the project root — creates tools/memory/.venv automatically
uv sync --project tools/memory --extra dev
```

### 2. Start Qdrant (optional prepared backend)

```bash
docker compose -f docker-compose.dev.yml up hars-memory-qdrant -d
# Confirm health:
curl http://localhost:6335/readyz
```

The installed LightRAG build (lightrag-hku 1.4.16) provides
`QdrantVectorDBStorage`, the default backend since the 2026-07-30 vector
transplant migration (zero re-embedding; see
`.plans/2026-07-29_graphrag-qdrant-migration.md`). Gotcha:
`HARS_MEMORY_QDRANT_COLLECTION` is not a Qdrant collection name — LightRAG uses
it as the tenant id (`workspace_id`) written into every payload; the actual
collections are always `lightrag_vdb_{chunks,entities,relationships}`.

### 3. Configure

```bash
cp tools/memory/config/.env.example tools/memory/config/.env
# Edit .env: set DSN, Qdrant URL, model paths
```

### 4. Run indexing (GPU MUST be free)

Wait until no `vea/expert/ai_tuner/finetune/distillation` experiment is running.
Confirm the OpenAI-compatible extractor endpoint is live before starting:

```bash
curl http://localhost:8080/v1/models  # extractor
```

Then:

```bash
HARS_MEMORY_EXTRACTOR_BASE_URL=http://localhost:8080/v1 \
HARS_MEMORY_EXTRACTOR_MODEL=Qwen3.6-27B-Q4_K_M \
HARS_MEMORY_QUERY_BASE_URL=http://localhost:8080/v1 \
HARS_MEMORY_QUERY_MODEL=Qwen3.6-27B-Q4_K_M \
uv run --project tools/memory python tools/memory/server/index.py \
    --paths .plans docs \
    --db-export
```

The script **automatically refuses** to run if the GPU guard detects a running training job.

Dry-run (no LLM, just document counts):
```bash
uv run --project tools/memory python tools/memory/server/index.py \
    --paths .plans docs --dry-run
```

If either endpoint is down, start `llama-server` with a real local `.gguf` file
first. The battle-tested failure mode is:

- no `*.gguf` under the expected model cache path, so `llama-server` cannot start;
- no `/v1/models` response on the configured endpoint, so real indexing/querying cannot run;
- dry-run ingest, Postgres export, MCP status, and MCP dry-run reindex still work.

---

## Swapping models

All model bindings are in `config/.env` (or environment variables). No code changes needed.

| What to swap | Variable | Notes |
|---|---|---|
| Extraction LLM | `HARS_MEMORY_EXTRACTOR_BASE_URL` + `HARS_MEMORY_EXTRACTOR_MODEL` | GPU-exclusive; update llama-server launch cmd |
| Query LLM | `HARS_MEMORY_QUERY_BASE_URL` + `HARS_MEMORY_QUERY_MODEL` | Can be CPU if small |
| Embedder | `HARS_MEMORY_EMBED_MODEL` | Changing the model changes the vector dimension — rebuild the index with `index.py --full` |
| Vector backend | `HARS_MEMORY_VECTOR_STORAGE` | Current default: `QdrantVectorDBStorage` (since 2026-07-30); `NanoVectorDBStorage` for rollback |
| Qdrant URL | `HARS_MEMORY_QDRANT_URL` | Dedicated `hars-memory-qdrant` compose service is `http://localhost:6335` |
| Qdrant tenant | `HARS_MEMORY_QDRANT_COLLECTION` | NOT a collection name — the `workspace_id` tenant id in every payload/query filter |
| Postgres DSN | `HARS_MEMORY_POSTGRES_DSN` | hars-postgres; read-only |

After swapping embedder: run `index.py --full` to rebuild all vectors.

---

## MCP tool usage (from an AI session)

Register: already in `.mcp.json` as `hars-longterm-memory` (launched via `uv run --project tools/memory memory-mcp`).

```
# Check index status (always GPU-free)
memory_status()

# Query the knowledge graph
memory_recall(
  question="Which hypotheses were invalidated by Phase C?",
  mode="hybrid",   # local | global | hybrid | naive
  top_k=12
)

# Find entity and its neighbours
memory_entities(name="Phase C")

# Traverse subgraph from a hypothesis
memory_related(entity_id="hyp:abc-123", hops=2)

# Trigger incremental reindex (dry-run by default)
memory_consolidate(paths=[".plans", "docs"], dry_run=true)
# Actual reindex (GPU must be free):
memory_consolidate(paths=[".plans", "docs"], db_export=true, dry_run=false)
```

---

## Architecture

```
.plans/ docs/
    │
    ▼
ingest/walker.py          — glob filter + .memoryignore
ingest/chunker.py         — character-level overlap chunking
ingest/postgres_export.py — stable-ID docs from hars-postgres
    │
    ▼
server/gpu_guard.py       — refuses indexing while training runs
server/lightrag_init.py   — LightRAG wired to embeddinggemma-300m (CPU) + vector/NetworkX + llama.cpp
server/index.py           — CLI entrypoint (GPU-guarded)
    │
    ▼
LightRAG working dir      — HARS_MEMORY_INDEX_DIR (NetworkX .graphml + KV JSON; vectors here only
                              while HARS_MEMORY_VECTOR_STORAGE=NanoVectorDBStorage)
Qdrant                    — http://localhost:6335 (hars-memory-qdrant; vector backend once migrated)
    │
    ▼
plugins/hars-longterm-memory/scripts/hars_longterm_memory_mcp.py
    — 7 MCP tools: memory_recall / memory_remember / memory_forget /
      memory_consolidate / memory_status / memory_entities / memory_related
    — registered in .mcp.json (launched via uv run --project tools/memory)
```

---

## Query modes

| Mode | Use for |
|---|---|
| `hybrid` (default) | Most questions — combines local entity-graph + global community |
| `local` | Specific entity facts (what do we know about H6?) |
| `global` | Broad summarisation (ROCm page fault across all reports) |
| `naive` | Pure vector similarity fallback |

Not to be confused with `memory_recall`'s separate `context_priority` param
(`context_only=True` path only): default `merged` as of 2026-07-30, reorders
LightRAG's own graph/vector context with the additive dense+BM25 `hybrid`
channel (round-robin, fusion first) and applies supersession-aware rescoring.
Measured on the 46-query labeled eval set: recall@1 0.477→0.5324, recall@10
0.727→0.8241, ndcg@10 0.637→0.7151, mrr 0.634→0.7030, supersession error rate
0.333→0.1667 vs the pre-2026-07-30 default, no regressions. Escape hatches:
pass `context_priority="lightrag"` on any single call, or set
`HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT=lightrag` server-wide. Supersession
rescoring itself is also on by default (`HARS_MEMORY_SUPERSESSION_SCORING=1`)
— set to `0` to disable. See `config/.env.example` for the full retrieval-
tuning env var list.

---

## Evaluation

```bash
# After indexing, run gold-question harness:
uv run --project tools/memory python tools/memory/eval/check.py --mode hybrid

# Generated retrieval battle test; context-only avoids one LLM answer per case.
uv run --project tools/memory python -m tools.memory.eval.battle \
    --paths .reports .plans .session \
    --db-export \
    --cases 100 \
    --context-only \
    --report tools/memory/eval/battle_report.json

# Scale to 1000 when the 100-case run is stable:
uv run --project tools/memory python -m tools.memory.eval.battle \
    --paths .reports .plans .session \
    --db-export \
    --cases 1000 \
    --context-only \
    --report tools/memory/eval/battle_report_1000.json
```

5 gold multi-hop questions are in `eval/gold_questions.yaml`.
The gold harness exits non-zero if any question fails source-node retrieval.
The battle harness exits non-zero if any generated case fails, or if the pass
rate drops below `--min-pass-rate`.

---

## Unit tests

```bash
# From project root
uv run --project tools/memory pytest plugins/hars-longterm-memory/tests/ -v
```

---

## GPU guard

Indexing calls `tools/memory/server/gpu_guard.py::assert_gpu_free()` which
queries the HARS backend for running experiments.  If `vea/expert/ai_tuner/
finetune/distillation` is running, indexing exits immediately with a clear
error message.  **Query tools never call the GPU guard** — CPU embeddings keep
`memory_status()` and `memory_recall()` always available even mid-training.

---

## Dependency management

The isolated UV project lives at `tools/memory/`.  Key files:

| File | Purpose |
|---|---|
| `tools/memory/pyproject.toml` | Dependencies + UV project config |
| `tools/memory/uv.lock` | Pinned, reproducible lock file |
| `tools/memory/.python-version` | Python 3.13 pin |
| `tools/memory/.venv/` | Isolated venv (CPU torch only — never touches ROCm) |

To update dependencies:
```bash
# Edit pyproject.toml, then:
uv lock --project tools/memory
uv sync --project tools/memory --extra dev
```

To add a dependency:
```bash
uv add --project tools/memory <package>
```
