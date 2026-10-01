from __future__ import annotations

from pathlib import Path

import pytest

from hars_memory import cli


def test_project_root_falls_back_for_shallow_installed_layout(
    tmp_path: Path, monkeypatch
) -> None:
    package_dir = tmp_path / "site-packages" / "hars_memory"
    package_dir.mkdir(parents=True)
    module_path = package_dir / "cli.py"
    module_path.touch()
    monkeypatch.setattr(cli, "__file__", str(module_path))

    assert cli._project_root() == package_dir.parent


def test_project_root_discovers_source_checkout(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / "checkout" / "hars_memory"
    package_dir.mkdir(parents=True)
    (tmp_path / "checkout" / "pyproject.toml").touch()
    module_path = package_dir / "cli.py"
    module_path.touch()
    monkeypatch.setattr(cli, "__file__", str(module_path))

    assert cli._project_root() == tmp_path / "checkout"


def test_recall_cli_passes_a_query_param_to_lightrag(monkeypatch, capsys) -> None:
    """Regression: `memory recall` used to call aquery(question, mode=...) -> TypeError."""
    from hars_memory import mcp_server

    seen: dict = {}

    class FakeRag:
        async def aquery(self, query, param=None, system_prompt=None):  # LightRAG >= 1.4 signature
            seen["param"] = param
            return "CONTEXT TEXT"

    async def fake_get_rag():
        return FakeRag()

    async def fake_hybrid(rag, question, top_k, ll_keywords):
        return {"enabled": False}

    monkeypatch.setattr(mcp_server, "_get_rag", fake_get_rag)
    monkeypatch.setattr(mcp_server, "_compute_hybrid_block", fake_hybrid)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["recall", "what is x", "--mode", "naive", "--top-k", "3"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "CONTEXT TEXT" in out
    assert seen["param"].mode == "naive"
    assert seen["param"].top_k == 3
    assert seen["param"].only_need_context is True
