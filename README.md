# hars-longterm-memory

A local, model-agnostic long-term memory service for AI agents. It indexes a
folder of Markdown/text knowledge into a [LightRAG](https://github.com/HKUDS/LightRAG)
graph plus a vector store, adds a BM25 sparse channel and optional reranking,
and serves the result through an [MCP](https://modelcontextprotocol.io) server
(`memory_recall`, `memory_remember`, ...), a CLI, a gRPC server and an HTTP
index-job service.

## Why

Agents forget between sessions. Plain vector search misses exact identifiers
(ticket ids, config keys, error codes) and cannot follow relations between
documents. This project combines:

- a knowledge graph (entities and relations extracted by an LLM you choose),
- dense vectors (CPU embeddings; NanoVectorDB or Qdrant),
- BM25 sparse retrieval and an optional live ripgrep channel for exact tokens,
- supersession-aware ranking that demotes self-declared deprecated documents,
- an evaluation harness so retrieval changes are measured, not guessed.

Every model binding (extractor LLM, query LLM, embedder, reranker) is an
environment variable; nothing is tied to a specific provider.

## Features

- Structure-aware Markdown chunking (`HARS_MEMORY_CHUNKER=markdown`) with heading
  breadcrumbs, whole tables/code fences, and chunk location metadata.
- Incremental indexing with content fingerprints and deleted-document cleanup.
- `memory index-batch`: LLM extraction through a provider Batch API (50% cheaper),
  resumable, with stall detection.
- Hybrid retrieval: graph + dense + BM25 (+ ripgrep, + optional flat dense).
- Optional HTTP or local cross-encoder reranker.
- MCP tools with `compact` and `view=lean` response modes to save agent context.
- Multi-project isolation, JWT tokens, encrypted keystore, RBAC.
- Retrieval metrics (recall@k, nDCG, MRR, supersession error rate) and a
  regression gate for CI.

## Architecture

```
 sources (md/txt/yaml)        .memoryignore, knowledge-sources.yaml
        |
     [walker] -> [chunker: token | markdown] -> [change detection]
                                                       |
                                          [LightRAG: LLM extraction]
                         +-----------------+-----------+---------------+
                         |                 |                           |
                   graph (NetworkX)  vectors (Qdrant)         BM25 + KV + fingerprints
                         |                 |                           |
                         +--------- retrieval fusion (+ ripgrep, + rerank) ---+
                                           |
              MCP (stdio)  |  CLI  |  gRPC  |  HTTP index service + SDK
```

See [docs/architecture.md](docs/architecture.md) for details.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:iuriimedvedev-dev/hars-longterm-memory.git
cd hars-longterm-memory
uv sync                 # runtime
uv sync --extra dev     # plus pytest, ruff
uv run memory --help
```

Vector storage defaults to the file-based `NanoVectorDBStorage` (no extra
service). For production use set `HARS_MEMORY_VECTOR_STORAGE=QdrantVectorDBStorage`
and `HARS_MEMORY_QDRANT_URL`.

## Quickstart

1. Configure (copy `config/.env.example` to `.env` and edit; at minimum the
   LLM endpoints, `HARS_MEMORY_INDEX_DIR` and `HARS_MEMORY_STAGING_DIR`):

   ```bash
   export HARS_MEMORY_INDEX_DIR=$HOME/.hars-memory/index
   export HARS_MEMORY_STAGING_DIR=$HOME/.hars-memory/staging
   export HARS_MEMORY_EXTRACTOR_BASE_URL=http://127.0.0.1:8080/v1   # any OpenAI-compatible server
   export HARS_MEMORY_EXTRACTOR_MODEL=my-extractor
   export HARS_MEMORY_QUERY_BASE_URL=http://127.0.0.1:8081/v1
   export HARS_MEMORY_QUERY_MODEL=my-query-model
   ```

2. Estimate the cost, then index a folder:

   ```bash
   uv run memory estimate-cost ./my-notes
   uv run memory-index --paths ./my-notes          # incremental; add --full to force
   ```

3. Query from the CLI:

   ```bash
   uv run memory recall "who owns the billing service?" --mode hybrid --top-k 10
   uv run memory recall "billing service owner" --context-only --json
   ```

4. Run the MCP server. Claude Code (`.mcp.json` in your project):

   ```json
   {
     "mcpServers": {
       "hars-memory": {
         "command": "uv",
         "args": ["run", "--project", "/path/to/hars-longterm-memory", "memory-mcp"],
         "env": {
           "HARS_MEMORY_INDEX_DIR": "/home/me/.hars-memory/index",
           "HARS_MEMORY_STAGING_DIR": "/home/me/.hars-memory/staging",
           "HARS_MEMORY_QUERY_BASE_URL": "http://127.0.0.1:8081/v1",
           "HARS_MEMORY_QUERY_MODEL": "my-query-model"
         }
       }
     }
   }
   ```

   Any MCP client that can launch a stdio command works the same way
   (command `uv`, args as above, or the `hars-longterm-memory-mcp` entrypoint).

## CLI overview

`memory` subcommands: `build`, `query`, `eval`, `regress`, `status`,
`strategy-bench`, `recall`, `mcp`, `server`, `grpc`, `consolidate`, `export`,
`import`, `migrate-index`, `index-batch`, `estimate-cost`, `auth`.
Other entrypoints: `memory-index`, `memory-mcp`, `memory-grpc`,
`memory-strategy-bench`, `hars-longterm-memory-server`.

## Configuration overview

All settings are `HARS_MEMORY_*` environment variables, grouped in
[docs/configuration.md](docs/configuration.md): LLM endpoints, embedder,
storage, chunking, retrieval tuning, reranking, auth, logging, services.

## Documentation

| Doc | Topic |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Modules and data flow |
| [docs/configuration.md](docs/configuration.md) | Every environment variable |
| [docs/indexing.md](docs/indexing.md) | Sources, chunkers, incremental and batch indexing |
| [docs/mcp-tools.md](docs/mcp-tools.md) | MCP tool reference |
| [docs/retrieval.md](docs/retrieval.md) | Hybrid retrieval, reranking, tuning |
| [docs/evaluation.md](docs/evaluation.md) | Metrics and writing your own eval cases |
| [docs/auth.md](docs/auth.md) | Keystore, JWT, projects, RBAC |
| [docs/index-service.md](docs/index-service.md) | HTTP index-job service and SDK |
| [docs/grpc-reference.md](docs/grpc-reference.md) | gRPC API |
| [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CHANGELOG.md](CHANGELOG.md) | Project |

## License

No license file has been added yet; until one is, all rights are reserved by
the author.
