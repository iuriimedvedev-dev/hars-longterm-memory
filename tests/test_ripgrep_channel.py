"""Unit tests for tools/memory/retrieval/ripgrep_channel.py.

GPU-free, no LLM calls, no network, and — per this session's constraints —
never touches `/home/user/.local/share/hars-graphrag/index_gemma_v4`
(the live, actively-mutating LightRAG index): every test here either runs
`rg` against synthetic `tmp_path` fixtures or, for the availability-gating
tests, monkeypatches `shutil.which`/PATH rather than depending on whether
the real system binary happens to be installed.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tools.memory.retrieval import ripgrep_channel as rgc
from tools.memory.retrieval.ripgrep_channel import (
    HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV,
    RipgrepSearchHit,
    check_availability,
    default_roots,
    extract_terms,
    search,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestExtractTerms:
    """Item: identifier-term extraction from a natural-language question."""

    def test_extracts_identifiers_from_question(self) -> None:
        terms = extract_terms(
            "What happened in phase_c1_lora_safe and does A2S32 relate to vea_native?"
        )
        assert terms == ["phase_c1_lora_safe", "A2S32", "vea_native"]

    def test_plain_question_yields_no_terms(self) -> None:
        assert extract_terms("How is training going in general?") == []

    def test_ll_keywords_contribute_identifier_shaped_terms(self) -> None:
        terms = extract_terms("How is training going?", ll_keywords=["A2S32"])
        assert terms == ["A2S32"]

    def test_ll_keywords_plain_words_are_rejected(self) -> None:
        # A plain-English keyword must not sneak a noisy substring search in
        # through the keyword side-channel — the whole point of this
        # channel is exact identifiers, not keyword recall.
        terms = extract_terms("How is training going?", ll_keywords=["training", "results"])
        assert terms == []

    def test_question_and_keywords_are_merged_and_deduped(self) -> None:
        terms = extract_terms(
            "What about A2S32?", ll_keywords=["a2s32", "phase_c1_lora_safe"]
        )
        assert terms == ["A2S32", "phase_c1_lora_safe"]

    def test_empty_question_no_keywords_yields_no_terms(self) -> None:
        assert extract_terms("") == []


class TestAvailabilityFailSoft:
    """Item: the rg-missing fail-soft path — never an exception."""

    def test_available_when_binary_present(self) -> None:
        availability = check_availability()
        # ripgrep is a hard prerequisite of this dev environment; assert the
        # *shape* of a positive result rather than hardcoding a path.
        if availability.available:
            assert availability.binary_path
            assert availability.reason is None

    def test_unavailable_when_binary_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rgc.shutil, "which", lambda _name: None)
        availability = check_availability()
        assert availability.available is False
        assert availability.binary_path is None
        assert "not found on PATH" in availability.reason

    def test_search_returns_unavailable_result_not_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path / "note.md", "Contains A2S32 somewhere.\n")
        monkeypatch.setattr(rgc.shutil, "which", lambda _name: None)

        result = search("What about A2S32?", roots=[tmp_path])

        assert result.available is False
        assert result.hits == []
        assert result.unavailable_reason is not None
        assert result.query_terms == ("A2S32",)

    def test_search_never_raises_on_binary_vanishing_mid_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate the race: available at check_availability() time, but the
        # exec itself raises FileNotFoundError.
        _write(tmp_path / "note.md", "A2S32 mentioned here.\n")

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise FileNotFoundError("rg vanished")

        monkeypatch.setattr(rgc.subprocess, "run", _boom)

        result = search("What about A2S32?", roots=[tmp_path])

        assert result.available is True  # binary WAS found at the gate check
        assert result.hits == []


class TestNoIdentifierTermsReturnsNothing:
    """Item: 'no identifier terms -> return nothing' behaviour."""

    def test_plain_question_returns_no_hits_without_invoking_rg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path / "note.md", "Some prose about training results.\n")

        def _fail_if_called(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("rg must not be invoked when there are no identifier terms")

        monkeypatch.setattr(rgc.subprocess, "run", _fail_if_called)

        result = search("How is training going in general today?", roots=[tmp_path])

        assert result.hits == []
        assert result.query_terms == ()


@pytest.mark.skipif(
    check_availability().available is False, reason="ripgrep (rg) not installed"
)
class TestSearchAgainstSyntheticTree:
    """End-to-end `search()` against a real, synthetic tmp_path tree — the
    real `rg` binary is exercised here, but never against the live index.
    """

    def _corpus(self, tmp_path: Path) -> Path:
        _write(
            tmp_path / ".reports" / "report_a.md",
            "# Report A\n\nExperiment A2S32 stabilized the gate at step 80.\n"
            "A2S32 also appears again here for emphasis.\n",
        )
        _write(
            tmp_path / ".session" / "session_b.md",
            "# Session B\n\nWe discussed phase_c1_lora_safe briefly, "
            "no relation to a2s32-suffix tokens.\n",
        )
        _write(
            tmp_path / "docs" / "guide.md",
            "# Guide\n\nGeneral training guidance, no identifiers here.\n",
        )
        return tmp_path

    def test_finds_whole_token_identifier_hit(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        result = search("What happened with A2S32?", roots=[corpus], top_k=5)

        assert result.available is True
        assert result.hits, "expected at least one hit for A2S32"
        top = result.hits[0]
        assert top.file_path.endswith("report_a.md")
        assert top.term_hits.get("A2S32", 0) >= 2

    def test_substring_hit_scores_lower_than_whole_token_hit(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        result = search("Anything about A2S32?", roots=[corpus], top_k=10)

        by_file = {Path(h.file_path).name: h for h in result.hits}
        assert "report_a.md" in by_file
        assert "session_b.md" in by_file
        # report_a.md has two WHOLE-TOKEN "A2S32" hits; session_b.md has one
        # SUBSTRING hit ("a2s32-suffix"). Whole-token must outrank substring.
        assert by_file["report_a.md"].score > by_file["session_b.md"].score

    def test_no_match_returns_empty_hits_not_error(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        result = search("What about ZZZ99notfound?", roots=[corpus], top_k=5)
        assert result.available is True
        assert result.hits == []

    def test_result_hit_shape_mirrors_bm25_hit_fields(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        result = search("What happened with A2S32?", roots=[corpus], top_k=5)
        assert result.hits
        hit = result.hits[0]
        assert isinstance(hit, RipgrepSearchHit)
        # Same first-four-field shape as bm25_index.BM25SearchHit.
        assert isinstance(hit.chunk_id, str) and hit.chunk_id.startswith("file:")
        assert isinstance(hit.score, float)
        assert isinstance(hit.content, str)
        assert isinstance(hit.file_path, str)

    def test_timeout_returns_empty_result_not_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = self._corpus(tmp_path)

        import subprocess as _subprocess

        def _timeout(*_args: object, **_kwargs: object) -> None:
            raise _subprocess.TimeoutExpired(cmd="rg", timeout=0.001)

        monkeypatch.setattr(rgc.subprocess, "run", _timeout)

        result = search("What happened with A2S32?", roots=[corpus], timeout_seconds=0.001)
        assert result.hits == []
        assert result.timed_out is True

    def test_co_occurrence_of_multiple_terms_boosts_score(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "single.md",
            "A2S32 appears once here, nothing else related.\n",
        )
        _write(
            tmp_path / "both.md",
            "A2S32 and vea_native both appear together in this note.\n",
        )
        result = search(
            "How do A2S32 and vea_native relate?", roots=[tmp_path], top_k=5
        )
        by_name = {Path(h.file_path).name: h for h in result.hits}
        assert by_name["both.md"].score > by_name["single.md"].score

    def test_ignores_memoryignore_excluded_files(self, tmp_path: Path) -> None:
        _write(tmp_path / "keep.md", "A2S32 lives here.\n")
        _write(tmp_path / "secret_a2s32.md", "A2S32 also lives here, but secretly.\n")
        _write(tmp_path / ".memoryignore", "secret_*\n")

        result = search("What about A2S32?", roots=[tmp_path], top_k=10)

        matched_names = {Path(h.file_path).name for h in result.hits}
        assert "keep.md" in matched_names
        assert "secret_a2s32.md" not in matched_names

    def test_legacy_graphragignore_warns_and_is_not_applied(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write(tmp_path / "keep.md", "A2S32 lives here.\n")
        _write(tmp_path / "secret_a2s32.md", "A2S32 also lives here.\n")
        _write(tmp_path / ".graphragignore", "secret_*\n")

        with caplog.at_level("WARNING"):
            result = search("What about A2S32?", roots=[tmp_path], top_k=10)

        matched_names = {Path(h.file_path).name for h in result.hits}
        # Legacy ignore file's pattern is NOT applied -> the "secret" file is
        # still found (loud warning, not a silent exclusion).
        assert "secret_a2s32.md" in matched_names
        assert any("legacy" in record.message.lower() for record in caplog.records)

    def test_excludes_venv_and_outputs_directories(self, tmp_path: Path) -> None:
        _write(tmp_path / "real.md", "A2S32 in the real corpus.\n")
        _write(tmp_path / ".venv" / "lib" / "site.md", "A2S32 in a venv artifact.\n")
        _write(tmp_path / "outputs" / "run1" / "log.md", "A2S32 in build output.\n")

        result = search("What about A2S32?", roots=[tmp_path], top_k=10)

        matched_names = {Path(h.file_path).name for h in result.hits}
        assert matched_names == {"real.md"}

    def test_measured_latency_is_reported_and_reasonable(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        result = search("What happened with A2S32?", roots=[corpus], top_k=5)
        assert result.latency_seconds >= 0.0
        assert result.latency_seconds < 2.0  # generous upper bound for a tiny fixture


class TestScoreFormula:
    """Item: the score formula — component-level checks against
    ripgrep_channel's internal helpers (documented in the module docstring).
    """

    def test_recency_multiplier_is_one_at_zero_age(self) -> None:
        now = time.time()
        assert rgc._recency_multiplier(now, now=now) == pytest.approx(1.0)

    def test_recency_multiplier_floors_for_old_files(self) -> None:
        now = time.time()
        very_old = now - rgc.RECENCY_HORIZON_DAYS * 86_400.0 * 10
        assert rgc._recency_multiplier(very_old, now=now) == pytest.approx(rgc.RECENCY_FLOOR)

    def test_recency_multiplier_is_monotonic_in_age(self) -> None:
        now = time.time()
        fresher = rgc._recency_multiplier(now - 1 * 86_400.0, now=now)
        older = rgc._recency_multiplier(now - 100 * 86_400.0, now=now)
        assert fresher > older

    def test_whole_token_match_detection(self) -> None:
        line = "Experiment A2S32 stabilized the gate."
        start = line.index("A2S32")
        end = start + len("A2S32")
        assert rgc._is_whole_token_match(line, start, end) is True

    def test_substring_match_detection(self) -> None:
        # The match is embedded inside a longer word on both sides, so it is
        # a substring hit, not a whole-token hit.
        line = "prefixa2s32suffix"
        start = line.index("a2s32")
        end = start + len("a2s32")
        assert rgc._is_whole_token_match(line, start, end) is False

    def test_path_type_weight_prioritizes_curated_sections(self, tmp_path: Path) -> None:
        session_file = tmp_path / ".session" / "note.md"
        code_file = tmp_path / "tools" / "memory" / "thing.py"
        roots = [tmp_path]
        assert rgc._path_type_weight(session_file, roots) > rgc._path_type_weight(
            code_file, roots
        )

    def test_path_type_weight_for_memory_dir_root(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        memory_dir = tmp_path / "claude_memory"
        memory_file = memory_dir / "note.md"
        roots = [project_root, memory_dir]
        assert rgc._path_type_weight(memory_file, roots) == pytest.approx(
            rgc._MEMORY_DIR_PATH_WEIGHT
        )


class TestGlobConventionsMatchWalker:
    """Drift guard: this module intentionally DUPLICATES (not imports)
    walker.py's private include/exclude glob constants, translated into
    rg's glob syntax (see ripgrep_channel.py's module docstring). This test
    asserts the two stay semantically equivalent — a future edit to
    walker.py's globs that isn't mirrored here will fail this test rather
    than silently diverging the two channels' file universes.
    """

    def test_include_globs_cover_the_same_extensions(self) -> None:
        from tools.memory.ingest import walker

        walker_extensions = {g.removeprefix("**/*") for g in walker._DEFAULT_INCLUDE_GLOBS}
        rg_extensions = {g.removeprefix("*") for g in rgc._INCLUDE_GLOBS_RG}
        assert walker_extensions == rg_extensions

    def test_exclude_globs_cover_the_same_names(self) -> None:
        from tools.memory.ingest import walker

        def _normalize(pattern: str) -> str:
            return pattern.strip("*/").removeprefix(".")

        walker_names = {_normalize(g) for g in walker._DEFAULT_EXCLUDE_GLOBS}
        rg_names = {_normalize(g) for g in rgc._EXCLUDE_GLOBS_RG}
        # rg's set is a superset (it adds .venv/ and outputs/ — see module
        # docstring "Search roots and filters" for why).
        assert walker_names <= rg_names

    def test_ignore_file_name_matches_walker(self) -> None:
        from tools.memory.ingest import walker

        assert rgc._DEFAULT_IGNORE_FILE == walker._DEFAULT_IGNORE_FILE
        assert rgc._LEGACY_IGNORE_FILE == walker._LEGACY_IGNORE_FILE


class TestDefaultRoots:
    def test_memory_dir_env_unset_yields_project_root_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, raising=False)
        roots = default_roots(tmp_path)
        assert roots == [tmp_path.resolve()]

    def test_memory_dir_env_set_is_appended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        monkeypatch.setenv(HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, str(memory_dir))
        roots = default_roots(tmp_path / "project")
        assert roots == [(tmp_path / "project").resolve(), memory_dir.resolve()]

    def test_memory_dir_not_duplicated_if_same_as_project_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, str(tmp_path))
        roots = default_roots(tmp_path)
        assert roots == [tmp_path.resolve()]
