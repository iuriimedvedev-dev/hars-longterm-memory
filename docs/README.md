# hars-longterm-memory documentation

Documentation index for the `hars-longterm-memory` package — a standalone,
model-agnostic long-term memory service built on LightRAG (graph+vector), Qdrant,
and CPU-only embeddings.

## Quick links

| Document | What it covers |
|---|---|
| [Architecture](architecture.md) / [Configuration](configuration.md) / [Indexing](indexing.md) | Modules, env vars, indexing modes |
| [MCP tools](mcp-tools.md) / [Retrieval](retrieval.md) / [Evaluation](evaluation.md) / [Auth](auth.md) | Tool args, retrieval pipeline, metrics, RBAC |
| [MCP tool reference](mcp-reference.md) | All 7 MCP tools: `memory_recall`, `memory_remember`, `memory_entities`, `memory_related`, `memory_status`, `memory_consolidate`, `memory_forget` — parameters, response shapes, examples |
| [Configuration reference](configuration.md) | All environment variables, their defaults, and semantics |
| [Architecture](architecture.md) | Module map, data flow, component relationships, retrieval pipeline |
| [Index service](index-service.md) | HTTP index-job service — SDK, API, deployment, artifact stores |
| [gRPC reference](grpc-reference.md) | gRPC service — 7 RPCs, client usage, configuration, health check |

### CLI Reference

The `memory` command provides a unified interface for all subsystems:

| Subcommand | Description |
|---|---|
| `memory recall <question>` | Query the LightRAG knowledge graph |
| `memory mcp` | Start MCP server (stdio) |
| `memory server` | Start HTTP index-job service |
| `memory grpc` | Start gRPC server |
| `memory build` | Build / incrementally re-index a corpus |
| `memory query` | Search a corpus-built index |
| `memory eval` | Run retrieval metrics against a corpus |
| `memory regress` | Compare two eval reports |
| `memory status` | Print corpus manifest summary |
| `memory consolidate` | Trigger incremental reindex |
| `memory export <output>` | Export index to a tar.gz archive |
| `memory import <archive>` | Import index from a tar.gz archive |
| `memory estimate-cost <paths>` | Estimate indexing cost before running (dry-run, no API calls) |
| `memory strategy-bench` | Run strategy matrix benchmark |

Examples:

```bash
# Quick query
memory recall "What is Kubernetes?" --mode hybrid

# Start servers
memory mcp
memory server --port 8787
memory grpc --port 8788

# Corpus management
memory build --paths kb/ --index-dir /tmp/my-index
memory query --index-dir /tmp/my-index --question "how to deploy?"

# Export/import index
memory export /tmp/hars-index-backup.tar.gz
memory import /tmp/hars-index-backup.tar.gz --index-dir /tmp/new-index

# Estimate cost before indexing
memory estimate-cost kb/ --model gpt-5.6-luna --chunk-size 2048 --batch-size 256
memory estimate-cost kb/ --model deepseek-v3.2 --max-gleaning 1
```

Legacy entrypoints (`memory-index`, `memory-mcp`, `memory-grpc`, `hars-longterm-memory-server`, etc.) remain available for backward compatibility.

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
  grpc/                   # gRPC server, client, generated proto stubs
  strategies.py           # validated index/search strategies
  scripts/                # maintenance tools
cortex-scripts/           # Cortex consumer scripts (not part of installed package)
docs/                     # this documentation
config/.env.example       # full env-var reference
tests/                    # 742+ passing tests
```