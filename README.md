# HARS Long-Term Memory

Local, model-agnostic long-term memory service for the HARS/Cortex project.
Serves the `hars-longterm-memory` MCP server (`memory_recall`, `memory_remember`,
`memory_forget`, `memory_consolidate`, `memory_status`, and the introspection-tier
`memory_entities` / `memory_related`). LightRAG graph+vector retrieval is one
retrieval channel among several (see `retrieval/` for BM25 fusion) — the public
tool surface never names the underlying mechanism.

**Documentation**: see `docs/` for the MCP tool reference, configuration reference, architecture, integration guide, and release/distribution process.

**Stack**: LightRAG (lightrag-hku 1.4.16) · unsloth/embeddinggemma-300m (768-dim) on CPU · NanoVectorDB (file-backed, migrating to Qdrant — see `.plans/2026-07-29_graphrag-qdrant-migration.md`) · NetworkX · Qwen3.6-27B extractor · Qwen3.5-4B query LLM.

**Package manager**: [UV](https://docs.astral.sh/uv/) — this standalone repository has its own `pyproject.toml` + `uv.lock` and is fully isolated from Cortex and its ROCm training environment.

---

## Quick start

### 1. Install dependencies

```bash
# From the project root — creates .venv automatically
uv sync --extra dev
```

## Remote indexing service and Python SDK

Install the optional server surface and start a local service:

```bash
uv sync --extra server

export HARS_MEMORY_API_KEYS_JSON='{"replace-with-a-secret-key":"tenant-a"}'
export HARS_MEMORY_SERVICE_DATABASE_URL='sqlite:///./hars-memory-service.sqlite3'
export HARS_MEMORY_ARTIFACT_STORE_URL="file://$(pwd)/hars-memory-artifacts"

uv run hars-longterm-memory-server
```

The API key mapping is `API key -> tenant ID`. It is required; the server
fails closed when it is absent or empty. Submit work through the SDK:

```python
from pathlib import Path

from hars_memory.sdk import HarsMemoryClient

with HarsMemoryClient("http://127.0.0.1:8787", "replace-with-a-secret-key") as client:
    job = client.create_index(
        [Path("docs/architecture.md"), Path("notes.txt")],
        engine="corpus",  # CPU-only and LLM-free
        idempotency_key="architecture-v1",
    )
    completed = client.wait_job(job.id)
    version = client.get_latest_index(completed.index_id)
    client.download_artifact(completed.index_id, Path("architecture-index.tar.gz"))

    extension = client.extend_index(
        completed.index_id,
        {"new-facts.md": "# New facts\n\nAdditional evidence."},
        expected_version=version.version,
        idempotency_key="architecture-v2",
    )
    client.wait_job(extension.id)
```

`engine="corpus"` builds the portable sparse/fusion corpus index without an
LLM or GPU. `engine="lightrag"` runs the existing LightRAG indexer inside the
server worker and therefore requires the configured extractor/query services;
deploy that server on the GPU machine. Every successful job publishes a new
immutable index version. Extending never mutates the last good version.

For cloud deployments, switch only configuration:

```bash
export HARS_MEMORY_SERVICE_DATABASE_URL='postgresql+psycopg://user:password@db/hars_memory'
export HARS_MEMORY_ARTIFACT_STORE_URL='s3://bucket/prefix'
export HARS_MEMORY_S3_ENDPOINT_URL='https://s3.example.com'  # optional for AWS
export HARS_MEMORY_S3_REGION='eu-central-1'
```

Install `--extra s3` for S3-compatible artifacts. Database rows contain job
metadata, leases, state transitions, tenant ownership, and artifact references;
index blobs remain in filesystem/S3. A Qdrant-backed LightRAG version reports
its external workspace in the returned manifest and `portable=false`—the
downloaded tarball intentionally does not pretend to contain remote Qdrant
vectors.

The service image is also buildable directly:

```bash
docker build -f Dockerfile.service -t hars-memory-service .
docker run --rm -p 8787:8787 \
  -e HARS_MEMORY_API_KEYS_JSON='{"replace-with-a-secret-key":"tenant-a"}' \
  -v hars-memory-data:/data \
  hars-memory-service
```

Its HTTP surface is `/health`, `POST/GET /v1/index-jobs`, job cancellation,
latest/version descriptors, and checksum-bearing artifact downloads under
`/v1/indexes/{index_id}/versions/...`.

## GitLab CI and container registry

Every branch and merge request runs repository-wide Ruff, the complete pytest
suite (live LLM tests remain opt-in/skipped), and builds wheel/sdist artifacts.
The default branch and tags additionally build `Dockerfile.service` on the
existing `sh` runner and publish the service image to GitLab Container
Registry using GitLab's short-lived `CI_REGISTRY_*` credentials.

Published tags are:

- `$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA` — immutable provenance tag;
- `$CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA` — operator-friendly commit tag;
- `$CI_REGISTRY_IMAGE:$CI_COMMIT_REF_SLUG` — tag pipelines only;
- `$CI_REGISTRY_IMAGE:latest` — default branch only.

No personal or long-lived registry token is stored in the repository. Test
results are exposed as GitLab JUnit reports, while wheel/sdist files remain
downloadable pipeline artifacts for 30 days.

## Index/search strategy evaluation and benchmarks

Indexing parameters are represented by a validated `IndexStrategy`; the SDK
sends its name/options with a job, and every published manifest records both
the snapshot and its stable SHA-256. Only a fixed option allowlist is accepted:
clients cannot inject arbitrary environment variables, URLs, paths, or secrets.

Use a declarative matrix to compare indexing and retrieval strategies:

```bash
cp config/strategy-bench.example.yaml /tmp/my-memory-matrix.yaml
# Edit corpus_paths, queries_path, strategies, repeats, and primary_metric.
uv run memory strategy-bench \
  --config /tmp/my-memory-matrix.yaml \
  --run-id experiment-001
```

Each index strategy is built once. Matching search strategies then run with
configured warmups/repetitions. The report includes immutable strategy hashes,
corpus/query/index hashes, build time, index size, documents/chunks per second,
Recall@k, NDCG@k, MRR, query latency, variance, and a primary-metric
leaderboard. Corpus strategies are LLM/GPU-free; LightRAG strategies call the
real indexer and the existing retrieval-only `ab_bench` in isolated processes.

### Opt-in real LLM E2E

The live test is skipped during every normal `pytest` run. When both model
servers are ready:

```bash
cp config/live-llm-e2e.env.example /tmp/hars-live.env
# Fill model aliases/cache path, then export the file without committing it.
set -a
source /tmp/hars-live.env
set +a

uv run pytest -q \
  tests/e2e/test_live_llm_strategy_e2e.py \
  -m live_llm -s
```

It preflights both `/v1/models` endpoints, starts the real index service,
creates a portable LightRAG v1 through the SDK, extends it to v2, verifies v1
immutability, downloads both artifacts, then runs real MCP retrieval against
both and real query-LLM synthesis against v2. The fixture has three source
documents plus one rollback extension and uses one concurrent extraction slot.

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
cp config/.env.example config/.env
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
uv run python -m hars_memory.server.index \
    --paths .plans docs
```

The script **automatically refuses** to run if the GPU guard detects a running training job.

Dry-run (no LLM, just document counts):
```bash
uv run python -m hars_memory.server.index \
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
# Actual reindex (GPU must be free if HARS_MEMORY_GPU_GUARD_SCRIPT_PATH is configured):
memory_consolidate(paths=[".plans", "docs"], dry_run=false)
```

Postgres export is not part of this package — it moved to the consuming
project's own script, which calls `hars_memory.ingest.api.ingest_documents`
directly (see `hars_memory/ingest/api.py`).

---

## Architecture

```
.plans/ docs/
    │
    ▼
ingest/walker.py          — glob filter + .memoryignore
ingest/chunker.py         — character-level overlap chunking
ingest/api.py             — public ingest_documents() API (consuming
                              projects call this directly for their own
                              non-filesystem sources, e.g. Postgres export)
    │
    ▼
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
uv run pytest -q
```

---

## GPU guard

GPU-guarding is entirely the consuming project's concern, not something this
generic package hardcodes. `memory_consolidate` (and `cortex-scripts/update_kb.sh`,
Cortex's own indexing wrapper — not part of the installed package) call a
GPU-guard script only if `HARS_MEMORY_GPU_GUARD_SCRIPT_PATH` is set, pointing
at a Python file exposing `assert_gpu_free(api_base_url)`; if unset, no
GPU-concurrency check is performed and indexing proceeds. Cortex's own guard
(`tools/memory-config/scripts/gpu_guard.py`) queries the HARS backend for
running experiments and refuses to proceed if `vea/expert/ai_tuner/
finetune/distillation` is running. **Query tools never call the GPU guard** —
CPU embeddings keep `memory_status()` and `memory_recall()` always available
even mid-training.

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
