"""Structure-aware Markdown chunking: md_structure parser, chunker, manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hars_memory.ingest import md_structure as md
from hars_memory.ingest.md_structure import (
    breadcrumb_for,
    chunk_note_structured,
    estimate_tokens,
    note_manifest,
    parse_frontmatter,
    parse_note,
    write_manifest_sidecar,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestFrontmatter:
    def test_fields_and_content_start(self) -> None:
        fields, start = parse_frontmatter(_read("md_struct_note_frontmatter.md"))

        assert fields["title"] == "Phase C Evaluation Report"
        assert fields["date"] == "2026-10-01"
        assert fields["tags"] == "[metrics, eval]"
        assert start == 6  # first line after the closing fence

    def test_no_frontmatter(self) -> None:
        assert parse_frontmatter("# Just a title\n\nbody\n") == ({}, 1)

    def test_unclosed_fence_is_not_frontmatter(self) -> None:
        assert parse_frontmatter("---\ntitle: nope\n") == ({}, 1)

    def test_leading_rule_without_keys_is_not_frontmatter(self) -> None:
        assert parse_frontmatter("---\n\nsome text\n\n---\n") == ({}, 1)

    def test_title_prefers_frontmatter_over_h1_over_stem(self) -> None:
        assert parse_note(_read("md_struct_note_frontmatter.md")).title == (
            "Phase C Evaluation Report"
        )
        assert parse_note("# H1 Title\n", "dir/stem.md").title == "H1 Title"
        assert parse_note("## Only H2\n", "dir/stem.md").title == "stem"


class TestHeadingsAndFences:
    def test_hashes_inside_fences_are_not_headings(self) -> None:
        note = parse_note(_read("md_struct_note_frontmatter.md"), "n.md")

        paths = [s.heading_path for s in note.sections]
        assert ("Phase C Evaluation Report", "Phase C", "Metrics") in paths
        assert ("Phase C Evaluation Report", "Phase C", "Notes") in paths
        assert not any("also not a heading" in p for path in paths for p in path)

    def test_section_tree_lines_tables_code(self) -> None:
        note = parse_note(_read("md_struct_note_frontmatter.md"), "n.md")
        by_title = {s.title: s for s in note.sections}

        assert by_title["Metrics"].has_table is True
        assert by_title["Metrics"].has_code is False
        assert by_title["Notes"].has_table is False
        assert by_title["Notes"].has_code is True
        assert by_title["Metrics"].start_line == 13
        assert by_title["Metrics"].end_line == 20
        assert by_title["Notes"].start_line == 22
        assert by_title["Notes"].end_line == 30

    def test_single_pipe_line_is_not_a_table(self) -> None:
        note = parse_note(_read("md_struct_note_edges.md"), "e.md")

        assert all(s.has_table is False for s in note.sections)

    def test_horizontal_rule_does_not_split(self) -> None:
        chunks = chunk_note_structured(_read("md_struct_note_edges.md"))

        assert len(chunks) == 1
        assert "---" in chunks[0].text

    def test_unclosed_fence_swallows_rest(self) -> None:
        note = parse_note("# T\n\n```sh\n# not heading\nmore\n", "u.md")

        assert [s.title for s in note.sections] == ["T"]


class TestSectionChunking:
    def test_one_section_one_chunk_with_breadcrumb(self) -> None:
        chunks = chunk_note_structured(
            _read("md_struct_note_frontmatter.md"), source_path="n.md",
            min_merge_chars=0,  # disable merge-up: isolate section chunking
        )
        crumbs = [c.breadcrumb for c in chunks]

        assert "[Phase C Evaluation Report > Phase C > Metrics]" in crumbs
        metrics = next(c for c in chunks if "0.87" in c.text)
        assert metrics.text.startswith(
            "[Phase C Evaluation Report > Phase C > Metrics]\n\n"
        )
        assert metrics.heading_path == (
            "Phase C Evaluation Report", "Phase C", "Metrics",
        )
        assert metrics.section == "Metrics"
        assert metrics.section_depth == 3
        assert metrics.source_path == "n.md"
        assert all(c.chunk_index == i for i, c in enumerate(chunks))

    def test_oversized_section_splits_with_repeated_breadcrumb(self) -> None:
        text = _read("md_struct_note_oversized.md")
        chunks = chunk_note_structured(text, target_chars=400, min_merge_chars=0)

        big = [c for c in chunks if c.heading_path[-1:] == ("Big Section",)]
        assert len(big) > 1
        assert all(
            c.breadcrumb == "[Long Note > Bare Parent > Big Section]" for c in big
        )
        assert all(len(c.text) <= 400 for c in big)
        assert all(c.text.startswith("[Long Note > Bare Parent > Big Section]\n\n") for c in big)
        # Paragraphs are never cut mid-line.
        for chunk in big:
            body = chunk.text.split("\n\n", 1)[1]
            assert "PARA0" in body or "PARA1" in body or "PARA2" in body or "PARA3" in body

    def test_tiny_sibling_merges_up(self) -> None:
        chunks = chunk_note_structured(
            _read("md_struct_note_oversized.md"),
            target_chars=2000,
            min_merge_chars=200,
        )
        tiny_alone = [
            c for c in chunks
            if c.heading_path and c.heading_path[-1] == "Tiny Sibling"
            and len(c.text) < 200
        ]

        assert tiny_alone == []
        merged = next(c for c in chunks if "[Tiny Sibling]" in c.text)
        assert "two lines." in merged.text

    def test_tiny_siblings_merge_up_into_parent(self) -> None:
        chunks = chunk_note_structured(
            "# D\n\n## Alpha\n\ntiny a\n\n## Beta\n\ntiny b\n",
            target_chars=2000,
            min_merge_chars=200,
        )

        assert len(chunks) == 1
        assert chunks[0].breadcrumb == "[D]"
        assert "tiny a" in chunks[0].text and "tiny b" in chunks[0].text

    def test_unrelated_top_level_sections_stay_separate_when_large(self) -> None:
        chunks = chunk_note_structured(
            "# D\n\n## Alpha\n\n" + "a" * 300 + "\n\n## Beta\n\n" + "b" * 300 + "\n",
            target_chars=2000,
            min_merge_chars=200,
        )

        assert any(c.heading_path[-1:] == ("Alpha",) for c in chunks)
        assert any(c.heading_path[-1:] == ("Beta",) for c in chunks)

    def test_is_code_heavy_and_depth(self) -> None:
        chunks = chunk_note_structured(_read("md_struct_note_edges.md"), source_path="e.md")

        assert len(chunks) == 1
        assert chunks[0].is_code_heavy is True
        assert chunks[0].section_depth == 1

    def test_empty_input_and_bad_params(self) -> None:
        assert chunk_note_structured("  \n ") == []
        with pytest.raises(ValueError):
            chunk_note_structured("x", target_chars=0)
        with pytest.raises(ValueError):
            chunk_note_structured("x", min_merge_chars=-1)

    def test_exact_line_numbers_on_fixture(self) -> None:
        lines = _read("md_struct_note_frontmatter.md").splitlines()
        chunks = chunk_note_structured(
            _read("md_struct_note_frontmatter.md"),
            target_chars=2000,
            min_merge_chars=0,  # keep sections separate for exact assertions
        )

        for chunk in chunks:
            assert 1 <= chunk.start_line <= chunk.end_line <= len(lines)
        metrics = next(c for c in chunks if "0.87" in c.text)
        assert lines[metrics.start_line - 1] == "### Metrics"
        assert lines[metrics.end_line - 1] == "| recall | 0.74 | 0.79 |"

    def test_merged_chunk_spans_merged_sections(self) -> None:
        lines = _read("md_struct_note_frontmatter.md").splitlines()
        chunks = chunk_note_structured(_read("md_struct_note_frontmatter.md"))

        metrics = next(c for c in chunks if "0.87" in c.text)
        # The tiny Notes section (with its closing fence) merged up.
        assert lines[metrics.end_line - 1] == "```"

    def test_breadcrumb_format(self) -> None:
        assert breadcrumb_for(("Phase C", "Metrics")) == "[Phase C > Metrics]"
        assert breadcrumb_for(()) == ""

    def test_estimate_tokens(self) -> None:
        assert estimate_tokens("a" * 400) == 100
        assert estimate_tokens("  ") == 0


class TestManifest:
    def test_manifest_shape(self) -> None:
        manifest = note_manifest(
            _read("md_struct_note_frontmatter.md"), "notes/n.md"
        )

        assert manifest["title"] == "Phase C Evaluation Report"
        assert manifest["source_path"] == "notes/n.md"
        assert manifest["frontmatter"]["date"] == "2026-10-01"
        assert len(manifest["sections"]) == 5
        metrics = next(
            s for s in manifest["sections"]
            if s["heading_path"][-1:] == ["Metrics"]
        )
        assert metrics == {
            "heading_path": ["Phase C Evaluation Report", "Phase C", "Metrics"],
            "level": 3,
            "start_line": 13,
            "end_line": 20,
            "chars": metrics["chars"],
            "est_tokens": metrics["chars"] // 4,
            "has_table": True,
            "has_code": False,
        }

    def test_manifest_line_ranges_cover_note_without_overlap(self) -> None:
        text = _read("md_struct_note_frontmatter.md")
        manifest = note_manifest(text)
        sections = manifest["sections"]

        assert sections[0]["start_line"] == 1
        assert sections[-1]["end_line"] == len(text.splitlines())
        for prev, nxt in zip(sections, sections[1:]):
            assert prev["end_line"] < nxt["start_line"]

    def test_sidecar_writer(self, tmp_path: Path) -> None:
        manifest = note_manifest("# T\n\n## A\n\nbody\n", "t.md")
        path = write_manifest_sidecar(tmp_path, "doc:1/a", manifest)

        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8"))["title"] == "T"


class TestAdapterAndSwitch:
    def test_adapter_record_shape(self) -> None:
        class FakeTok:
            @staticmethod
            def encode(text: str) -> list[int]:
                return list(text.encode("utf-8"))

        text = (
            "[Document: note.md | Section: all | Date: 2026-10-01]\n\n"
            + _read("md_struct_note_frontmatter.md")
        )
        records = md.md_struct_chunking_func(
            FakeTok(), text, None, False, 100, 1200
        )

        assert records
        assert [r["chunk_order_index"] for r in records] == list(range(len(records)))
        first = records[0]
        assert set(first) >= {
            "tokens", "content", "chunk_order_index", "heading_path",
            "section", "section_depth", "is_code_heavy", "start_line",
            "end_line", "source_path",
        }
        assert first["source_path"] == "note.md"
        # Line numbers address the file on disk (walker header subtracted).
        assert all(r["start_line"] >= 1 for r in records)

    def test_adapter_delegates_non_markdown(self) -> None:
        import sys
        import types

        class FakeTok:
            @staticmethod
            def encode(text: str) -> list[int]:
                return list(text.encode("utf-8"))

        calls: list[str] = []
        stub_operate = types.ModuleType("lightrag.operate")

        def fake_split(tokenizer: object, content: str, *args: object) -> list[dict]:
            calls.append(content)
            return [{"tokens": 1, "content": content, "chunk_order_index": 0}]

        stub_operate.chunking_by_token_size = fake_split  # type: ignore[attr-defined]
        stub_lightrag = types.ModuleType("lightrag")
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setitem(sys.modules, "lightrag", stub_lightrag)
        monkeypatch.setitem(sys.modules, "lightrag.operate", stub_operate)
        try:
            records = md.md_struct_chunking_func(
                FakeTok(), "plain prose " * 500, None, False, 100, 1200
            )
        finally:
            monkeypatch.undo()

        assert calls == ["plain prose " * 500]
        assert records == [
            {"tokens": 1, "content": "plain prose " * 500, "chunk_order_index": 0}
        ]

    def test_legacy_default_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HARS_MEMORY_CHUNKER_MODE", raising=False)
        monkeypatch.delenv("HARS_MEMORY_CHUNKER", raising=False)
        from hars_memory.server.lightrag_init import resolve_chunker

        # Unset everything -> legacy token splitter, byte-identical default.
        assert resolve_chunker() == "token"

    def test_mode_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.server.lightrag_init import resolve_chunker

        monkeypatch.setenv("HARS_MEMORY_CHUNKER_MODE", "md_struct")
        assert resolve_chunker() == "md_struct"
        monkeypatch.setenv("HARS_MEMORY_CHUNKER_MODE", "legacy")
        monkeypatch.setenv("HARS_MEMORY_CHUNKER", "markdown")
        assert resolve_chunker() == "token"  # MODE wins over CHUNKER
        monkeypatch.delenv("HARS_MEMORY_CHUNKER_MODE", raising=False)
        monkeypatch.setenv("HARS_MEMORY_CHUNKER", "markdown")
        assert resolve_chunker() == "markdown"  # pre-existing switch untouched

    def test_invalid_mode_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_CHUNKER_MODE", "bogus")
        from hars_memory.server.lightrag_init import resolve_chunker

        with pytest.raises(ValueError, match="HARS_MEMORY_CHUNKER_MODE"):
            resolve_chunker()
