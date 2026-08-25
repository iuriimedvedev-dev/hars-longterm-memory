"""Tests for attribution-header emission (`[Document: ... | Section: ... | Date: ...]`)
on walker- and postgres-sourced documents.

Reuses ``scripts/cleanup_kb.py``'s actual ``HEADER_RE`` (rather than a
hand-copied regex) so these tests fail loudly if the two formats ever drift.
"""

from __future__ import annotations

from pathlib import Path

from hars_memory.ingest.document import HEADER_DATE_UNKNOWN, build_source_header
from hars_memory.ingest.postgres_export import (
    transform_experiment_row,
    transform_hypothesis_link_row,
    transform_hypothesis_row,
)
from hars_memory.ingest.walker import walk
from hars_memory.scripts.cleanup_kb import HEADER_RE


class TestBuildSourceHeader:
    def test_matches_cleanup_kb_header_re(self) -> None:
        header = build_source_header(document_name="foo.md", section="docs", date="2026-05-01")
        m = HEADER_RE.search(header)
        assert m is not None
        assert m.group("name").strip() == "foo.md"
        assert m.group("section").strip() == "docs"
        assert m.group("date") == "2026-05-01"

    def test_unknown_date_matches_cleanup_kb_header_re(self) -> None:
        header = build_source_header(
            document_name="foo.md", section="docs", date=HEADER_DATE_UNKNOWN
        )
        m = HEADER_RE.search(header)
        assert m is not None
        assert m.group("date") == "unknown"

    def test_matches_mcp_memory_remember_format(self) -> None:
        """Byte-for-byte structural match against the format emitted by
        hars_longterm_memory_mcp.py's memory_remember tool (not editable here —
        we only assert format compatibility)."""
        header = build_source_header(document_name="note.md", section="staging", date="2026-07-29")
        assert header.startswith("[Document: note.md | Section: staging | Date: 2026-07-29]")


class TestWalkerHeaderEmission:
    def test_real_read_prepends_header_within_first_400_chars(self, tmp_path: Path) -> None:
        root = tmp_path / ".session"
        root.mkdir()
        (root / "note.md").write_text("# Some content\n\nBody text.\n")

        docs, _ = walk([root], dry_run=False)

        assert len(docs) == 1
        doc = docs[0]
        m = HEADER_RE.search(doc.content[:400])
        assert m is not None
        assert m.group("name").strip() == "note.md"
        assert m.group("section").strip() == "session"
        # No git repo in tmp_path -> falls back to mtime -> a real date, not "unknown".
        assert m.group("date") != HEADER_DATE_UNKNOWN

    def test_section_defaults_to_docs_for_unmapped_root(self, tmp_path: Path) -> None:
        root = tmp_path / "some_custom_root"
        root.mkdir()
        (root / "note.md").write_text("content")

        docs, _ = walk([root], dry_run=False)

        m = HEADER_RE.search(docs[0].content[:400])
        assert m is not None
        assert m.group("section").strip() == "docs"

    def test_section_defaults_to_code_for_python_files(self, tmp_path: Path) -> None:
        root = tmp_path / "some_custom_root"
        root.mkdir()
        (root / "script.py").write_text("print('hi')\n")

        docs, _ = walk([root], dry_run=False)

        m = HEADER_RE.search(docs[0].content[:400])
        assert m is not None
        assert m.group("section").strip() == "code"

    def test_dry_run_does_not_prepend_header(self, tmp_path: Path) -> None:
        root = tmp_path / ".session"
        root.mkdir()
        (root / "note.md").write_text("content")

        docs, _ = walk([root], dry_run=True)

        assert HEADER_RE.search(docs[0].content) is None

    def test_frontmatter_is_passed_through_after_header(self, tmp_path: Path) -> None:
        """Frontmatter-bearing files (as in the Claude memory dir) get the
        header prepended AHEAD of the frontmatter, unmodified."""
        root = tmp_path / "memory"
        root.mkdir()
        original = "---\nname: Some Note\ndescription: test\n---\nBody\n"
        (root / "note.md").write_text(original)

        docs, _ = walk([root], dry_run=False)

        content = docs[0].content
        m = HEADER_RE.search(content[:400])
        assert m is not None
        assert m.group("section").strip() == "memory"
        # Frontmatter survives, unmodified, right after the header.
        assert content.endswith(original)


class TestPostgresHeaderEmission:
    def test_experiment_row_header(self) -> None:
        row = {"id": "42", "name": "exp-42", "status": "succeeded", "updated_at": "2026-06-01T10:00:00"}
        doc = transform_experiment_row(row)
        m = HEADER_RE.search(doc.content[:400])
        assert m is not None
        assert m.group("name").strip() == "exp:42"
        assert m.group("section").strip() == "experiment"
        assert m.group("date") == "2026-06-01"

    def test_hypothesis_row_header_falls_back_to_created_at(self) -> None:
        row = {"id": "abc", "title": "H1", "created_at": "2026-01-15T00:00:00"}
        doc = transform_hypothesis_row(row)
        m = HEADER_RE.search(doc.content[:400])
        assert m is not None
        assert m.group("section").strip() == "hypothesis"
        assert m.group("date") == "2026-01-15"

    def test_hypothesis_link_row_header_unknown_date_when_no_timestamp(self) -> None:
        row = {"id": "9", "hypothesis_id": "abc", "entity_type": "experiment", "entity_id": "42"}
        doc = transform_hypothesis_link_row(row)
        m = HEADER_RE.search(doc.content[:400])
        assert m is not None
        assert m.group("section").strip() == "hypothesis_link"
        assert m.group("date") == HEADER_DATE_UNKNOWN
