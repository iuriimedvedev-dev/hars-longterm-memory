# LightRAG version pin (`lightrag-hku`) — the index-cutover blocker

Status: **decision recorded 2026-10-17** (channel-hygiene + index-cutover audit).
Applies to the serving venv (`/home/angkira/Project/software/hars-longterm-memory/.venv`).

## Decision

**Keep the repo on the 1.4.x line: `lightrag-hku>=1.4.16,<1.5`.**

The deployment target is the **builder's version, 1.5.7** — but the repo cannot
*safely* move to 1.5.7 yet, because two of its own modules import symbols that
1.5.x removed. Pinning `==1.5.7` unpatched (the review's Gate 1) breaks the
build: measured **37 test failures** on 2026-10-17 (32 + 4 `ImportError`, plus 1
unrelated environment failure), all from removed imports in
`hars_memory/ingest/*` and `hars_memory/eval/*`. The upper bound `<1.5` is a
fail-safe: without it, the next `uv lock --upgrade` silently resolves 1.5.7 and
breaks the repo the same way (`>=1.4.16` alone is a latent time-bomb).

Move to `==1.5.7` in the same change that lands the import migration below
(and then update `tests/test_version_pin.py`).

## Why this is a cutover blocker (measured, not inferred)

The serving venv must be able to **read the store it serves**, and the LightRAG
`kv_store_doc_status.json` schema is version-sensitive. This is not
hypothetical: even the *current* live index was built by a newer LightRAG than
1.4.16 reads.

Two specimens, loaded with the **1.4.16** reader (`DocProcessingStatus(**row)`,
the exact call `JsonDocStatusStorage` makes):

| Specimen | Rows | Status values | 1.4.16-constructible | Failure |
| --- | ---: | --- | ---: | --- |
| `index_full_mdstruct_20261014` (new, 1.5.7 Colab build) | 1340 | `processed 958, failed 290, pending 17, **parsing 75**` | 290 / 1340 | `TypeError: unexpected keyword argument 'content_hash'` on **1050** rows |
| `index_compact_20260903_imported` (live) | 3158 | `processed 1604, failed 1554` | 2769 / 3158 | `content_hash` on **389** rows |

1.5.x `DocProcessingStatus` gained `content_hash` (`lightrag/base.py`); 1.4.16
has no such field. 1.5.x `DocStatus` also added the enum values `parsing` and
`analyzing`; 1.4.16's `get_status_counts()` does `counts[doc["status"]] += 1`
over a dict keyed only by the old enum, so on the new store it raises:

```
KeyError: 'parsing'      # 75 parsing rows are unknown to 1.4.16
```

Consequences of serving the new store on 1.4.16:

- `get_status_counts()` / `memory_status` → **KeyError** (crash).
- `get_docs_by_status*()` / `get_by_id()` → the 1050 `content_hash` rows are
  **silently dropped** (per-row `except (KeyError, TypeError): continue`), so
  status and delete surfaces under-report and can miss documents.
- The **dense/sparse recall path is unaffected**: `vdb_chunks.json` (7442
  vectors) and `kv_store_text_chunks.json` load and query fine under 1.4.16
  (verified). Only the management/status surfaces break.

So a 1.4.16 reader can *answer queries* but cannot faithfully *manage* a
1.5.7-built store. Since the cutover swaps in exactly such a store, reader and
builder must be aligned at **1.5.7**.

## The prerequisite: import migration (owned by `hars_memory/ingest/*`, `eval/*`)

Confirmed moved/removed between 1.4.16 and 1.5.7:

| File:line | 1.4.16 import | 1.5.7 replacement |
| --- | --- | --- |
| `hars_memory/ingest/chunker.py:1512` | `from lightrag.operate import chunking_by_token_size` | `from lightrag.chunker import chunking_by_token_size` (also `lightrag.chunker.token_size`) |
| `hars_memory/ingest/md_structure.py:817` | `from lightrag.operate import chunking_by_token_size` | `from lightrag.chunker import chunking_by_token_size` |
| `hars_memory/eval/extractor_bench.py:102` | `from lightrag.constants import DEFAULT_ENTITY_TYPES` | **removed entirely in 1.5** — define the list locally (copy 1.4.16's `lightrag.constants.DEFAULT_ENTITY_TYPES`) |

`DEFAULT_SUMMARY_LANGUAGE` still exists in `lightrag.constants` (1.5.7), so the
same line's second import is fine.

## Upgrade procedure (serving venv → 1.5.7)

```bash
cd /home/angkira/Project/software/hars-longterm-memory

# 0. Snapshot for rollback.
cp pyproject.toml /tmp/pyproject.toml.bak && cp uv.lock /tmp/uv.lock.bak

# 1. Land the import migration above (chunker.py, md_structure.py,
#    extractor_bench.py). Do NOT skip: the next step fails 37 tests otherwise.

# 2. Pin the builder's version.
$EDITOR pyproject.toml     # "lightrag-hku==1.5.7"
$EDITOR tests/test_version_pin.py   # update the pinned-decision assertion

# 3. Relock + sync.
uv lock
uv sync --all-extras --all-groups     # full dev env; the serving path only needs `uv sync`
uv run python -c "import importlib.metadata as m; assert m.version('lightrag-hku')=='1.5.7', m.version('lightrag-hku')"

# 4. Gate.
uv run pytest -q        # expect 0 failed except the known test_tokenize_russian
uv run ruff check hars_memory tests
```

The MCP service consumes this venv via `cortex/.mcp.json`
(`uv run --project ../hars-longterm-memory memory mcp`), so it picks up 1.5.7 on
its **next restart** — no service restart is performed by this audit.

## Rollback procedure (back to 1.4.16)

```bash
cd /home/angkira/Project/software/hars-longterm-memory
cp /tmp/pyproject.toml.bak pyproject.toml
cp /tmp/uv.lock.bak uv.lock
uv lock && uv sync --all-extras --all-groups
uv run python -c "import importlib.metadata as m; print('lightrag-hku', m.version('lightrag-hku'))"   # 1.4.16
uv run pytest -q
```

Rollback is pure dependency state — no index mutation, so it can be done at any
time without touching the store. If the pip cache is cold, `uv` re-downloads
1.4.16 from PyPI (network required).

## Interim mitigation for the store (if the cutover lands before the migration)

Either keep serving the **1.4.16-built** live store until 1.5.7 is in, or
normalise the new store's `kv_store_doc_status.json` for 1.4.16 (strip
`content_hash`; drop/relabel `parsing`/`analyzing` rows). Both are inferior to
aligning the version — the normalisation is lossy and must be redone on every
rebuild.
