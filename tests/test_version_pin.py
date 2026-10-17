"""Pin the LightRAG version decision (index-cutover blocker, audit 2026-10-17).

`hars_memory`'s serving code imports two symbols that `lightrag-hku` 1.5
removed/moved:

* ``chunking_by_token_size`` — `lightrag.operate` (1.4.x) → `lightrag.chunker`
  (1.5.x); imported by ``ingest/chunker.py`` and ``ingest/md_structure.py``.
* ``DEFAULT_ENTITY_TYPES`` — removed from `lightrag.constants` entirely in
  1.5; imported by ``eval/extractor_bench.py``.

A 1.5.7-built ``kv_store_doc_status.json`` is also not readable by 1.4.16
(rows carry ``content_hash``; the ``parsing``/``analyzing`` statuses are unknown
to 1.4.16's ``DocStatus``). See ``docs/VERSION_PIN.md``.

This test pins the *interim* decision: the declared specifier must EXCLUDE the
1.5 line until those imports migrate, so the repo cannot silently resolve onto
a version it cannot import. Update this test in the same commit that lands the
import migration and moves the pin to ``==1.5.7``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _declared_specifier() -> str:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for dep in data["project"]["dependencies"]:
        if dep.replace(" ", "").startswith("lightrag-hku"):
            return dep.removeprefix("lightrag-hku")
    raise AssertionError("lightrag-hku is not a declared dependency")


def test_lightrag_specifier_excludes_unmigrated_1_5() -> None:
    spec = SpecifierSet(_declared_specifier())
    assert not spec.contains("1.5.0"), (
        "lightrag-hku specifier admits the 1.5 line, but hars_memory still imports "
        "`chunking_by_token_size` from lightrag.operate and `DEFAULT_ENTITY_TYPES` "
        "from lightrag.constants (both removed in 1.5). Migrate those imports "
        "(see docs/VERSION_PIN.md) and update this test before widening the pin."
    )
    assert not spec.contains("1.5.7"), "builder's version admitted before code migration"


def test_lightrag_specifier_admits_the_working_1_4_line() -> None:
    spec = SpecifierSet(_declared_specifier())
    assert spec.contains("1.4.16"), "the version the repo currently runs must remain admitted"
