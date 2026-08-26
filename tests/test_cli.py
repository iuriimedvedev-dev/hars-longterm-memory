from __future__ import annotations

from pathlib import Path

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
