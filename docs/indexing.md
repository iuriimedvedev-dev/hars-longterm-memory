# Indexing

## What gets indexed

Files under the paths you give (`--paths`) or under the roots of a knowledge
sources manifest. `.memoryignore` files (gitignore-style) are honoured by the
walker.

### Sources manifest

A YAML file owned by your knowledge repository, selected with
`HARS_MEMORY_SOURCES_MANIFEST`:

```yaml
version: 1
sources:
  - path: kb          # relative to the manifest's directory
    index: true       # ingest root (default true)
    grep: true        # live literal-search (ripgrep) root (default true)
    note: Team knowledge base.
```

A sibling `meta/knowledge-sources.local.yaml` (or `HARS_MEMORY_SOURCES_MANIFEST_LOCAL`)
is appended when present, for machine-specific roots. Missing paths are skipped
with a warning. With no manifest set, behaviour is unchanged: pass `--paths`.

## Chunkers

| `HARS_MEMORY_CHUNKER` | Behaviour |
|---|---|
| `token` (default) | LightRAG token splitter: `HARS_MEMORY_CHUNK_TOKEN_SIZE` (512) with `HARS_MEMORY_CHUNK_OVERLAP_TOKENS` (64) overlap |
| `markdown` | Structure-aware: splits at every heading (H1-H6) and prefixes a heading breadcrumb; keeps tables, code fences, lists and blockquotes whole; oversized tables are cut by row (at cell boundaries for very long rows) with the header repeated; fences are re-balanced; front matter folds into the first chunk; chunks smaller than `HARS_MEMORY_CHUNK_MIN_TOKENS` (200) are merged into a neighbour in the same top-level section. Non-Markdown files fall back to the token splitter. No overlap |

Chunk ids are `md5(chunk text)`. Switching chunker (or sizes) on an existing
index changes ids and requires re-extraction (LLM re-index) of the affected
documents. Plan for it, or build a new index directory.

## Incremental indexing

```bash
uv run memory-index --paths ./notes            # only new/changed documents
uv run memory-index --paths ./notes --full     # ignore change detection
uv run memory-index --paths ./notes --dry-run  # walk and count, no LLM calls
uv run memory-index --paths ./notes --refresh-changed
```

Change detection stores a content fingerprint per document in
`doc_fingerprints.json` (fingerprints are persisted only after a document reaches
LightRAG's `PROCESSED` status, so failed documents are retried). Documents that
disappear from disk are removed from the stores; a fail-safe guard refuses a
mass deletion that looks like a wrong path. `--refresh-changed` additionally
detects content changes for documents that were previously indexed.

`memory consolidate [--paths ...] [--dry-run]` triggers the same incremental
run (this is what the `memory_consolidate` MCP tool calls).

## Estimating cost

```bash
uv run memory estimate-cost ./notes --model <model> --input-price 0.15 --output-price 0.60
```

Dry run, no API calls: scans files, applies the chunk settings (`--chunk-size`,
`--chunk-overlap`, `--max-gleaning`) and prints the expected token usage and
cost. Prices are USD per 1M tokens; `--model` is used for a built-in price
lookup that `--input-price/--output-price` override.

## Batch indexing: `memory index-batch`

LightRAG calls the extraction LLM synchronously, once per chunk. Provider Batch
APIs (OpenAI-compatible) are half price but asynchronous, so `index-batch`
**primes LightRAG's LLM response cache** with batch answers and then runs the
normal indexing, in which every extraction call is a cache hit. State lives in
`<index>/batch_state.json`; every phase is resumable and idempotent.

| Phase | What it does |
|---|---|
| `collect` | Chunk documents and record every prompt LightRAG would send (no network). `--dry-run` prints the estimate only |
| `submit` | Write JSONL, upload, create batches. `--max-batch-tokens` splits a round; `--max-inflight-tokens` respects a provider's enqueued-token quota |
| `status` | Show state; `--refresh` polls the provider and downloads finished output |
| `apply` | Prime the cache, then run normal indexing. Failed/missing requests are answered by the normal synchronous call |
| `run` | All of the above, waiting (`--poll-seconds`, `--timeout-minutes`; exits 4 when the wait times out, resumable) |
| `cancel` | Cancel unfinished batches (manual) |
| `compare` | Compare `--index-dir` with `--other DIR` |

```bash
uv run memory index-batch collect --dry-run --paths ./notes      # estimate
uv run memory index-batch run --paths ./notes --max-cost 5.00 \
    --input-price 0.075 --output-price 0.30
```

Safety: `--max-cost` (default 0.50 USD at batch prices) aborts if the estimate is
higher. Gleaning (`HARS_MEMORY_MAX_GLEANING`) adds a second round whose prompts
depend on round-1 answers.

### Stall handling

Batches sometimes sit at 95 percent for a long time. During `run`:

- a batch whose completed+failed count has not grown for `--stall-minutes`
  (default 20, env `HARS_MEMORY_BATCH_STALL_MINUTES`, `0` disables) **and** has
  reached `--stall-min-done-fraction` (default 0.5) is cancelled;
- after a cancel, the tool waits `--cancel-wait-minutes` (default 15, env
  `HARS_MEMORY_BATCH_CANCEL_WAIT_MINUTES`) for a terminal state, then abandons
  the part;
- unanswered requests of cancelled/abandoned parts go through the synchronous
  fallback, still guarded by `--max-cost`.

The LLM endpoint and key come from the usual `HARS_MEMORY_EXTRACTOR_*` and
`HARS_MEMORY_LLM_API_KEY` variables. Keep `HARS_MEMORY_MAX_GLEANING` and the
chunker identical to any baseline index you compare against.

## Migrating an existing index: `memory migrate-index`

Back-fills chunk location metadata (`heading_path`, line range, real source path)
into an index built before that metadata existed. No LLM calls; vectors are not
re-embedded.

```bash
uv run memory migrate-index --index-dir "$HARS_MEMORY_INDEX_DIR" --root ./notes --dry-run
uv run memory migrate-index --index-dir "$HARS_MEMORY_INDEX_DIR" --root ./notes
uv run memory migrate-index ... --dedupe   # also drop chunks of duplicate documents
```

`--root` is re-walked (with `.memoryignore`) to recover real paths.

## Moving an index

```bash
uv run memory export ./snapshot.tar.gz --index-dir "$HARS_MEMORY_INDEX_DIR"
uv run memory import ./snapshot.tar.gz --index-dir /new/location [--force]
```

## LLM-free alternative

`memory build --paths ... --index-dir ...` and `memory query --question ... --mode sparse|dense|fusion`
use the lightweight `corpus` pipeline: no LLM, no graph, CPU only. Useful for
baselines and quick experiments.
