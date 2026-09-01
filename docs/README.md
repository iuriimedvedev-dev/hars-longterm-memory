# hars-longterm-memory documentation

Documentation index for the `hars-longterm-memory` package — a standalone,
model-agnostic long-term memory service built on LightRAG (graph+vector), Qdrant,
and CPU-only embeddings.

## Quick links

| Document | What it covers |
|---|---|
| [MCP tool reference](mcp-reference.md) | All 7 MCP tools: `memory_recall`, `memory_remember`, `memory_entities`, `memory_related`, `memory_status`, `memory_consolidate`, `memory_forget` — parameters, response shapes, examples |
| [Configuration reference](configuration.md) | All environment variables, their defaults, and semantics |
| [Architecture](architecture.md) | Module map, data flow, component relationships, retrieval pipeline |
| [Integration guide](integration-guide.md) | How to consume this package from another project — install from GitLab registry, set up MCP server, configure Qdrant, use the skill |
| [Release process](release-process.md) | How to publish a new version — PyPI package, container image, CI/CD |
| [Index service](index-service.md) | HTTP index-job service — SDK, API, deployment, artifact stores |

## Quick start

```bash
uv sync --extra dev
cp config/.env.example config/.env
# edit config/.env with your paths
uv run pytest -q
```

## Repository structure

```
hars_memory/              # Python package
  cli.py                  # console-script entrypoints
  mcp_server.py           # MCP server: 7 tools, ~2900 lines
  ingest/                 # walker, chunker, change detection
  server/                 # index.py, lightrag_init.py, embedder, reranker
  retrieval/              # BM25, flat-dense, ripgrep, fusion, supersession
  corpus/                 # LLM-free build/query pipeline
  schema/                 # entity/relation type schema
  eval/                   # gold-question harness, battle test, metrics
  service/                # HTTP index-job API, artifact stores, SDK
  sdk.py                  # synchronous HTTP client
  strategies.py           # validated index/search strategies
  scripts/                # maintenance tools
cortex-scripts/           # Cortex consumer scripts (not part of installed package)
docs/                     # this documentation
config/.env.example       # full env-var reference
tests/                    # 583+ passing tests
```