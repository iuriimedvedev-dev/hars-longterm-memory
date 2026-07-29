"""Permanent fail-closed guard against the pre-rename `GRAPHRAG_*` env prefix.

This component was renamed `hars-graphrag` -> `hars-longterm-memory` on
2026-07-29 (see .plans/2026-07-29_rename-to-hars-longterm-memory.md). All
config now reads `HARS_MEMORY_*`, with NO dual-prefix fallback — a stale
`GRAPHRAG_*` env var is silently ignored by every `os.environ.get("HARS_MEMORY_...")`
call in this codebase, which would make a server start "successfully" against
an empty/default index instead of the one the operator actually configured.

Call `refuse_if_legacy_graphrag_env()` at the very start of any entrypoint
that reads `HARS_MEMORY_*` config (the MCP server and the indexing CLI) so a
stale caller environment fails LOUDLY at startup instead of silently.

This guard is NOT a compatibility shim — it accepts nothing from the old
prefix. It only refuses to start. Keep it permanently; it is the only defence
against a stale config (an old `.env`, a cached docker-compose env, an old
shell profile export) resurfacing months after the rename.
"""

from __future__ import annotations

import os


def refuse_if_legacy_graphrag_env() -> None:
    """Raise SystemExit if any `GRAPHRAG_*` environment variable is present."""
    legacy = sorted(k for k in os.environ if k.startswith("GRAPHRAG_"))
    if legacy:
        raise SystemExit(
            "Legacy GRAPHRAG_* environment detected: " + ", ".join(legacy) + " — "
            "this component was renamed to hars-longterm-memory; use HARS_MEMORY_* "
            "instead. Refusing to start rather than silently falling back to defaults."
        )


__all__ = ["refuse_if_legacy_graphrag_env"]
