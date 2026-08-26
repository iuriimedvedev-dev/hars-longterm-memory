# 2026-08-26 — Repo extraction and onboarding docs

## Context

`hars-longterm-memory` was extracted from the `cortex` monorepo's
`tools/memory/` on 2026-08-25 via `git filter-repo` (history preserved), and
is now a standalone repo published to `code.knotgate.com`'s GitLab Package
Registry as package `hars-longterm-memory`, project id `14`
(`hars/hars-longterm-memory`). Confirmed via the registry API
(`GET /api/v4/projects/14`) during this session: name and path match, and
the package registry currently lists versions 0.1.0, 0.1.1, 0.1.2, 0.1.3.
HEAD at the time of this session is `8f5016f6`, tagged `v0.1.3`.

Cortex consumes this package via a companion `tools/memory-config` project,
which supplies the entity schema, the GPU-guard script
(`tools/memory-config/scripts/gpu_guard.py`), and Cortex's own
Postgres-export / update-KB wrapper scripts (`cortex-scripts/` in this repo
is the historical reference copy of those wrappers — not part of the
installed wheel).

## Decisions made

- No cortex-specific code should live in this repo going forward — the
  `cortex-scripts/` directory is kept only as a working reference for what a
  consumer's own operational wrappers look like, explicitly excluded from
  the wheel (`[tool.hatch.build.targets.wheel] packages = ["hars_memory"]`).
- Added `AGENTS.md` / `CLAUDE.md` / `.session/` / `.plans/` conventions
  matching the parent cortex project's own conventions, so this repo is
  self-sufficient for an agent that has never seen the monorepo.
- Added a portable skill at `.claude/skills/hars-longterm-memory/SKILL.md`
  meant to be copied into any OTHER project's `.claude/skills/` — the
  primary deliverable of this session, grounded in a direct read of
  `mcp_server.py`'s tool catalogue, `cli.py`'s entrypoints, and every
  `os.environ`/`os.getenv` call site across `hars_memory/`.

## Verified facts (as of this session)

- `uv sync --group dev` does **not** work — `dev` is declared under
  `[project.optional-dependencies]`, not a `[dependency-groups]` table. The
  correct command is `uv sync --extra dev`.
- `uv run pytest -q`: **532 passed, 5 skipped**, 0 failed.
- Console-script entrypoints (`[project.scripts]`): `memory-index`,
  `memory-mcp`, `hars-longterm-memory-mcp` (→ `hars_memory.mcp_server:main`
  — the actual MCP server entrypoint), `memory-eval-battle`, `memory` (the
  unified `build/query/eval/regress/status` subcommand CLI).
- 7 MCP tools confirmed by reading `mcp_server.py`'s `list_tools()`/
  `call_tool()` directly: `memory_recall`, `memory_remember`,
  `memory_entities`, `memory_related`, `memory_status`, `memory_consolidate`,
  `memory_forget`.
- torch CPU-only override (`[[tool.uv.index]] name="pytorch-cpu"` +
  `[tool.uv.sources] torch = {index="pytorch-cpu"}`) confirmed present in
  `pyproject.toml`, unchanged.

## Known limitation (parked, not fixed this session)

`hars_memory/eval/battle.py` line 23 still computes
`_PROJECT_ROOT = Path(__file__).resolve().parents[3]` — the same
monorepo-depth-assumption bug that was fixed in `server/index.py` for v0.1.3
(commit `8f5016f6`, "index.py's `_PROJECT_ROOT` had the same C1 disease").
`battle.py` was not part of that fix wave. It only affects
`memory-eval-battle` invoked with relative `--paths`; not load-bearing for
the MCP server or the core CLI. See `AGENTS.md` for the full writeup and the
workaround (pass absolute `--paths`).

Also noted (not fixed): `hars_memory/cli.py`'s `_project_root()` /
`_ensure_project_root()` (used by `index_main`, `eval_battle_main`, and the
unified `memory` command) carries the identical `parents[3]` pattern. It
appears harmless in practice — the actual `from hars_memory.X import Y`
imports resolve via the installed package in `site-packages` regardless of
what this inserts into `sys.path` — but it's the same anti-pattern and worth
cleaning up alongside `battle.py`.

## Next steps

- Fix `battle.py`'s `_PROJECT_ROOT` (and `cli.py`'s matching pattern) in a
  dedicated v0.1.4 change, following the exact same fix shape as the
  `server/index.py` fix (resolve relative paths against `Path.cwd()` at
  invocation time) plus a regression test, not as a drive-by inside an
  unrelated change.
- No other open items from this session — this was a docs-only pass; no
  package code was touched.
