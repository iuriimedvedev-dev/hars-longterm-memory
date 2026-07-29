"""Importable console entry points for the HARS long-term memory UV project."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_project_root() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def index_main() -> None:
    _ensure_project_root()
    from tools.memory.server.index import main

    main()


def mcp_main() -> None:
    _ensure_project_root()
    script_path = _project_root() / "plugins" / "hars-longterm-memory" / "scripts" / "hars_longterm_memory_mcp.py"
    spec = importlib.util.spec_from_file_location("hars_longterm_memory_mcp", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load MCP server module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    asyncio.run(module.main())


def eval_battle_main() -> None:
    _ensure_project_root()
    from tools.memory.eval.battle import main

    main()
