# Contributing

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev        # `dev` is an optional-dependencies extra, not a dependency group
uv run pytest -q           # offline suite, no LLM or network needed
uv run ruff check .        # line length 100, target py313
```

Live-LLM end-to-end tests are opt-in (`HARS_RUN_LIVE_LLM_E2E=1`); copy
`config/live-llm-e2e.env.example` to `config/live-llm-e2e.env` (git-ignored) and
point it at your own endpoints. Never commit that file.

## Conventions

- Commit messages follow Conventional Commits (`feat(scope):`, `fix(scope):`,
  `docs:`, `test:`, `refactor:`, `chore:`).
- Every behaviour change needs a test. Tests must be hermetic: no real LLM, no
  network, no machine-specific paths. Use `tmp_path` and the synthetic fixtures
  in `tests/e2e/fixtures/live_llm/`.
- Do not put confidential or third-party document content in fixtures, eval
  cases or docs. Use small synthetic documents.
- No machine-specific defaults in code: paths and endpoints come from
  `HARS_MEMORY_*` environment variables (see `docs/configuration.md`).
- New settings are documented in `config/.env.example` and `docs/configuration.md`.
- Chunk ids are `md5(chunk text)`: changes to chunking invalidate existing
  indexes, so call that out in the PR.
- `mcp` is pinned `<2.0.0` on purpose (the 2.x API renamed the decorators this
  server uses); re-verify the whole MCP server before bumping it.

## Pull requests

Keep PRs focused, run the tests and `ruff` before pushing, and describe how you
verified the change (commands and results).
