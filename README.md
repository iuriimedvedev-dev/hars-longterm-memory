# HARS GraphRAG

Local, model-agnostic GraphRAG service for the HARS/Cortex project.

**Stack**: LightRAG (lightrag-hku 1.4.16) · e5-large-v2 on CPU · NanoVectorDB (file-backed PoC) · NetworkX · Qwen3.6-27B extractor · Qwen3.5-4B query LLM.

**Package manager**: [UV](https://docs.astral.sh/uv/) — standalone project at `tools/graphrag/` with its own `pyproject.toml` + `uv.lock`. Fully isolated from the main workspace and the ROCm training venv.

---

## Quick start

### 1. Install dependencies

```bash
# From the project root — creates tools/graphrag/.venv automatically
uv sync --project tools/graphrag --extra dev
```

### 2. Start Qdrant (optional prepared backend)

```bash
docker compose -f docker-compose.dev.yml up hars-graphrag-qdrant -d
# Confirm health:
curl http://localhost:6335/readyz
```

The current pinned LightRAG package uses `NanoVectorDBStorage` by default because
this installed build does not include a Qdrant storage implementation. The
dedicated Qdrant service is kept ready for the next backend swap.

### 3. Configure

```bash
cp tools/graphrag/config/.env.example tools/graphrag/config/.env
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
GRAPHRAG_EXTRACTOR_BASE_URL=http://localhost:8080/v1 \
GRAPHRAG_EXTRACTOR_MODEL=Qwen3.6-27B-Q4_K_M \
GRAPHRAG_QUERY_BASE_URL=http://localhost:8080/v1 \
GRAPHRAG_QUERY_MODEL=Qwen3.6-27B-Q4_K_M \
uv run --project tools/graphrag python tools/graphrag/server/index.py \
    --paths .reports .plans .session \
    --db-export
```

The script **automatically refuses** to run if the GPU guard detects a running training job.

Dry-run (no LLM, just document counts):
```bash
uv run --project tools/graphrag python tools/graphrag/server/index.py \
    --paths .reports .plans --dry-run
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
| Extraction LLM | `GRAPHRAG_EXTRACTOR_BASE_URL` + `GRAPHRAG_EXTRACTOR_MODEL` | GPU-exclusive; update llama-server launch cmd |
| Query LLM | `GRAPHRAG_QUERY_BASE_URL` + `GRAPHRAG_QUERY_MODEL` | Can be CPU if small |
| Embedder | `GRAPHRAG_EMBED_MODEL` | Also update `qdrant.vector_size` in `graphrag.yaml` |
| Vector backend | `GRAPHRAG_VECTOR_STORAGE` | Current default: `NanoVectorDBStorage`; switch only to an installed LightRAG backend |
| Qdrant URL | `GRAPHRAG_QDRANT_URL` | Dedicated compose service is `http://localhost:6335` |
| Postgres DSN | `GRAPHRAG_POSTGRES_DSN` | hars-postgres; read-only |

After swapping embedder: run `index.py --full` to rebuild all vectors.

---

## MCP tool usage (from an AI session)

Register: already in `.mcp.json` as `hars-graphrag` (launched via `uv run --project tools/graphrag graphrag-mcp`).

```
# Check index status (always GPU-free)
graphrag_status()

# Query the knowledge graph
graphrag_query(
  question="Which hypotheses were invalidated by Phase C?",
  mode="hybrid",   # local | global | hybrid | naive
  top_k=12
)

# Find entity and its neighbours
graphrag_search_entities(name="Phase C")

# Traverse subgraph from a hypothesis
graphrag_get_subgraph(entity_id="hyp:abc-123", hops=2)

# Trigger incremental reindex (dry-run by default)
graphrag_reindex(paths=[".reports", ".plans"], dry_run=true)
# Actual reindex (GPU must be free):
graphrag_reindex(paths=[".reports", ".plans"], db_export=true, dry_run=false)
```

---

## Architecture

```
.reports/ .plans/ .session/
    │
    ▼
ingest/walker.py          — glob filter + .graphragignore
ingest/chunker.py         — character-level overlap chunking
ingest/postgres_export.py — stable-ID docs from hars-postgres
    │
    ▼
server/gpu_guard.py       — refuses indexing while training runs
server/lightrag_init.py   — LightRAG wired to e5-large (CPU) + NanoVectorDB/NetworkX + llama.cpp
server/index.py           — CLI entrypoint (GPU-guarded)
    │
    ▼
LightRAG working dir      — /tmp/hars_graphrag_lightrag/ (NetworkX .graphml + NanoVectorDB JSON)
Qdrant                    — http://localhost:6335 (prepared optional backend)
    │
    ▼
plugins/hars-graphrag/scripts/hars_graphrag_mcp.py
    — 5 MCP tools: query / search_entities / get_subgraph / status / reindex
    — registered in .mcp.json (launched via uv run --project tools/graphrag)
```

---

## Query modes

| Mode | Use for |
|---|---|
| `hybrid` (default) | Most questions — combines local entity-graph + global community |
| `local` | Specific entity facts (what do we know about H6?) |
| `global` | Broad summarisation (ROCm page fault across all reports) |
| `naive` | Pure vector similarity fallback |

---

## Evaluation

```bash
# After indexing, run gold-question harness:
uv run --project tools/graphrag python tools/graphrag/eval/check.py --mode hybrid

# Generated retrieval battle test; context-only avoids one LLM answer per case.
uv run --project tools/graphrag python -m tools.graphrag.eval.battle \
    --paths .reports .plans .session \
    --db-export \
    --cases 100 \
    --context-only \
    --report tools/graphrag/eval/battle_report.json

# Scale to 1000 when the 100-case run is stable:
uv run --project tools/graphrag python -m tools.graphrag.eval.battle \
    --paths .reports .plans .session \
    --db-export \
    --cases 1000 \
    --context-only \
    --report tools/graphrag/eval/battle_report_1000.json
```

5 gold multi-hop questions are in `eval/gold_questions.yaml`.
The gold harness exits non-zero if any question fails source-node retrieval.
The battle harness exits non-zero if any generated case fails, or if the pass
rate drops below `--min-pass-rate`.

---

## Unit tests

```bash
# From project root
uv run --project tools/graphrag pytest plugins/hars-graphrag/tests/ -v
```

---

## GPU guard

Indexing calls `tools/graphrag/server/gpu_guard.py::assert_gpu_free()` which
queries the HARS backend for running experiments.  If `vea/expert/ai_tuner/
finetune/distillation` is running, indexing exits immediately with a clear
error message.  **Query tools never call the GPU guard** — CPU embeddings keep
`graphrag_status()` and `graphrag_query()` always available even mid-training.

---

## Dependency management

The isolated UV project lives at `tools/graphrag/`.  Key files:

| File | Purpose |
|---|---|
| `tools/graphrag/pyproject.toml` | Dependencies + UV project config |
| `tools/graphrag/uv.lock` | Pinned, reproducible lock file |
| `tools/graphrag/.python-version` | Python 3.13 pin |
| `tools/graphrag/.venv/` | Isolated venv (CPU torch only — never touches ROCm) |

To update dependencies:
```bash
# Edit pyproject.toml, then:
uv lock --project tools/graphrag
uv sync --project tools/graphrag --extra dev
```

To add a dependency:
```bash
uv add --project tools/graphrag <package>
```
