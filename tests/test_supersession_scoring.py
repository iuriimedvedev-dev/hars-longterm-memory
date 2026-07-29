"""Unit tests for tools/memory/retrieval/supersession.py and its wiring
into tools/memory/retrieval/fusion.py behind HARS_MEMORY_SUPERSESSION_SCORING.

Regression fixtures below are drawn from real chunk content read directly out
of the live index (kv_store_text_chunks.json, index_gemma_v4 snapshot
2026-07-12) — see the module docstring of supersession.py for the full
false-positive investigation these fixtures encode (project_droid_mapping_v3
and project_b1_sidecar_family_falsified deliberately must NOT trigger the
marker penalty despite containing "wrong"/"falsified" verbiage about a
DIFFERENT, already-corrected artifact).
"""

from __future__ import annotations

from datetime import date

import pytest


class TestExtractChunkDate:
    def test_parses_iso_date_from_header(self) -> None:
        from tools.memory.retrieval.supersession import extract_chunk_date

        content = "[Document: foo.md | Section: session | Date: 2026-04-18]\n\nBody text."
        assert extract_chunk_date(content) == date(2026, 4, 18)

    def test_unknown_date_returns_none(self) -> None:
        from tools.memory.retrieval.supersession import extract_chunk_date

        content = "[Document: foo.md | Section: memory | Date: unknown]\n\nBody text."
        assert extract_chunk_date(content) is None

    def test_missing_header_returns_none(self) -> None:
        from tools.memory.retrieval.supersession import extract_chunk_date

        # Non-chunk-000 chunks carry no header at all — this is the expected,
        # majority case (only chunk-000 of each doc gets the header).
        assert extract_chunk_date("...continuation of a chunk with no header...") is None

    def test_empty_content_returns_none(self) -> None:
        from tools.memory.retrieval.supersession import extract_chunk_date

        assert extract_chunk_date("") is None


class TestMarkerPenalty:
    def test_deprecated_name_field_triggers_penalty(self) -> None:
        from tools.memory.retrieval.supersession import apply_supersession_scoring
        from tools.memory.retrieval.fusion import FusedChunk

        # Real content shape (trimmed) from project_hires_hypothesis_disproven.md.
        content = (
            "[Document: project_hires_hypothesis_disproven.md | Section: memory | Date: unknown]\n\n"
            "---\n"
            "name: DEPRECATED — hi-res hypothesis was token-Jaccard artifact\n"
            "description: Prior claim was a metric artifact.\n"
            "---\n"
            "**THIS MEMORY IS DEPRECATED.**\n\nThe 2026-04-17 conclusion was wrong."
        )
        chunk = FusedChunk(
            chunk_id="c1", fused_score=0.9, dense_score=0.9, sparse_score=None,
            dense_norm=0.9, sparse_norm=0.0, content=content,
            file_path="project_hires_hypothesis_disproven.md",
        )
        out = apply_supersession_scoring([chunk], enable_recency_discount=False)
        assert out[0].fused_score == pytest.approx(0.9 * 0.3)

    def test_slug_containing_falsified_word_does_not_trigger(self) -> None:
        """b1-sidecar-family-falsified: the marker is mid-slug, describing the
        TOPIC (a sidecar family that was falsified), not a self-verdict on
        this note. Must NOT be penalized — this is the CORRECT doc for sq03.
        """
        from tools.memory.retrieval.supersession import apply_supersession_scoring
        from tools.memory.retrieval.fusion import FusedChunk

        content = (
            "[Document: project_b1_sidecar_family_falsified.md | Section: memory | Date: unknown]\n\n"
            "---\n"
            "name: b1-sidecar-family-falsified\n"
            "description: Layer-24 gated-residual VEA sidecar runtime-falsified 2026-07-09\n"
            "---\n"
            "As of 2026-07-09, the B1/A2.5 gated-residual sidecar family is falsified at runtime."
        )
        chunk = FusedChunk(
            chunk_id="c2", fused_score=0.8, dense_score=0.8, sparse_score=None,
            dense_norm=0.8, sparse_norm=0.0, content=content,
            file_path="project_b1_sidecar_family_falsified.md",
        )
        out = apply_supersession_scoring([chunk], enable_recency_discount=False)
        assert out[0].fused_score == pytest.approx(0.8)

    def test_body_text_describing_a_different_artifact_as_wrong_does_not_trigger(self) -> None:
        """project_droid_mapping_v3.md: "the old v2 map was fundamentally
        wrong" describes a DIFFERENT, already-superseded artifact — this doc
        is itself the CORRECT, current one for sq05. Must NOT be penalized.
        """
        from tools.memory.retrieval.supersession import apply_supersession_scoring
        from tools.memory.retrieval.fusion import FusedChunk

        content = (
            "[Document: project_droid_mapping_v3.md | Section: memory | Date: unknown]\n\n"
            "---\n"
            "name: DROID cadene-hi-res mapping v3 (authoritative)\n"
            "description: How cadene episode_index maps to hi-res folders\n"
            "---\n"
            "The old v2 map was index-based and fundamentally wrong. All downstream "
            "artifacts built on v2 are invalid."
        )
        chunk = FusedChunk(
            chunk_id="c3", fused_score=0.95, dense_score=0.95, sparse_score=None,
            dense_norm=0.95, sparse_norm=0.0, content=content,
            file_path="project_droid_mapping_v3.md",
        )
        out = apply_supersession_scoring([chunk], enable_recency_discount=False)
        assert out[0].fused_score == pytest.approx(0.95)

    def test_plain_chunk_with_no_marker_is_unaffected(self) -> None:
        from tools.memory.retrieval.supersession import apply_supersession_scoring
        from tools.memory.retrieval.fusion import FusedChunk

        content = "[Document: some_report.md | Section: session | Date: 2026-05-01]\n\nNormal body text."
        chunk = FusedChunk(
            chunk_id="c4", fused_score=0.6, dense_score=0.6, sparse_score=None,
            dense_norm=0.6, sparse_norm=0.0, content=content, file_path="some_report.md",
        )
        out = apply_supersession_scoring([chunk], enable_recency_discount=False)
        assert out[0].fused_score == pytest.approx(0.6)

    def test_marker_scan_is_scoped_to_header_zone_not_whole_body(self) -> None:
        """A marker word appearing only far into a long chunk body (well past
        the header+frontmatter+opening-sentence zone) must not trigger —
        mirrors the real corpus finding that late-body mentions are about
        OTHER artifacts, not a self-verdict.
        """
        from tools.memory.retrieval.supersession import (
            _SCAN_ZONE_CHARS,
            apply_supersession_scoring,
        )
        from tools.memory.retrieval.fusion import FusedChunk

        padding = "x" * (_SCAN_ZONE_CHARS + 200)
        content = f"[Document: foo.md | Section: reports | Date: 2026-01-01]\n\n{padding}\nDEPRECATED"
        chunk = FusedChunk(
            chunk_id="c5", fused_score=0.7, dense_score=0.7, sparse_score=None,
            dense_norm=0.7, sparse_norm=0.0, content=content, file_path="foo.md",
        )
        out = apply_supersession_scoring([chunk], enable_recency_discount=False)
        assert out[0].fused_score == pytest.approx(0.7)


class TestRecencyDiscount:
    def test_unknown_date_chunk_is_never_discounted(self) -> None:
        from tools.memory.retrieval.supersession import apply_supersession_scoring
        from tools.memory.retrieval.fusion import FusedChunk

        content = "[Document: foo.md | Section: memory | Date: unknown]\n\nCurrent living note."
        chunk = FusedChunk(
            chunk_id="c6", fused_score=0.5, dense_score=0.5, sparse_score=None,
            dense_norm=0.5, sparse_norm=0.0, content=content, file_path="foo.md",
        )
        out = apply_supersession_scoring(
            [chunk], enable_marker_penalty=False, reference_date=date(2026, 7, 29)
        )
        assert out[0].fused_score == pytest.approx(0.5)

    def test_old_dated_chunk_is_discounted_but_bounded(self) -> None:
        from tools.memory.retrieval.supersession import (
            MAX_RECENCY_DISCOUNT,
            apply_supersession_scoring,
        )
        from tools.memory.retrieval.fusion import FusedChunk

        # Far older than RECENCY_SATURATION_DAYS relative to the reference date
        # -> ramp saturates at the hard cap, never below it.
        content = "[Document: old.md | Section: session | Date: 2020-01-01]\n\nStale-by-age report."
        chunk = FusedChunk(
            chunk_id="c7", fused_score=1.0, dense_score=1.0, sparse_score=None,
            dense_norm=1.0, sparse_norm=0.0, content=content, file_path="old.md",
        )
        out = apply_supersession_scoring(
            [chunk], enable_marker_penalty=False, reference_date=date(2026, 7, 29)
        )
        assert out[0].fused_score == pytest.approx(1.0 * (1.0 - MAX_RECENCY_DISCOUNT))

    def test_recent_dated_chunk_is_discounted_less_than_older_one(self) -> None:
        from tools.memory.retrieval.supersession import apply_supersession_scoring
        from tools.memory.retrieval.fusion import FusedChunk

        ref = date(2026, 7, 29)
        recent = FusedChunk(
            chunk_id="recent", fused_score=1.0, dense_score=1.0, sparse_score=None,
            dense_norm=1.0, sparse_norm=0.0,
            content="[Document: r.md | Section: session | Date: 2026-07-01]\n\nBody.",
            file_path="r.md",
        )
        older = FusedChunk(
            chunk_id="older", fused_score=1.0, dense_score=1.0, sparse_score=None,
            dense_norm=1.0, sparse_norm=0.0,
            content="[Document: o.md | Section: session | Date: 2025-01-01]\n\nBody.",
            file_path="o.md",
        )
        out = apply_supersession_scoring(
            [recent, older], enable_marker_penalty=False, reference_date=ref
        )
        by_id = {c.chunk_id: c for c in out}
        assert by_id["recent"].fused_score > by_id["older"].fused_score

    def test_future_or_same_day_date_is_not_discounted(self) -> None:
        from tools.memory.retrieval.supersession import apply_supersession_scoring
        from tools.memory.retrieval.fusion import FusedChunk

        content = "[Document: t.md | Section: session | Date: 2026-07-29]\n\nBody."
        chunk = FusedChunk(
            chunk_id="c8", fused_score=0.4, dense_score=0.4, sparse_score=None,
            dense_norm=0.4, sparse_norm=0.0, content=content, file_path="t.md",
        )
        out = apply_supersession_scoring(
            [chunk], enable_marker_penalty=False, reference_date=date(2026, 7, 29)
        )
        assert out[0].fused_score == pytest.approx(0.4)


class TestApplySupersessionScoringReordersAndIsPure:
    def test_reorders_after_penalty_drops_a_chunk_below_a_competitor(self) -> None:
        from tools.memory.retrieval.supersession import apply_supersession_scoring
        from tools.memory.retrieval.fusion import FusedChunk

        deprecated = FusedChunk(
            chunk_id="dep", fused_score=0.9, dense_score=0.9, sparse_score=None,
            dense_norm=0.9, sparse_norm=0.0,
            content=(
                "[Document: dep.md | Section: memory | Date: unknown]\n\n---\n"
                "name: DEPRECATED — old claim\ndescription: x\n---\n"
                "**THIS MEMORY IS DEPRECATED.**"
            ),
            file_path="dep.md",
        )
        current = FusedChunk(
            chunk_id="cur", fused_score=0.5, dense_score=0.5, sparse_score=None,
            dense_norm=0.5, sparse_norm=0.0,
            content="[Document: cur.md | Section: memory | Date: unknown]\n\nCurrent finding.",
            file_path="cur.md",
        )
        out = apply_supersession_scoring([deprecated, current], enable_recency_discount=False)
        assert [c.chunk_id for c in out] == ["cur", "dep"]

    def test_empty_list_returns_empty_list(self) -> None:
        from tools.memory.retrieval.supersession import apply_supersession_scoring

        assert apply_supersession_scoring([]) == []

    def test_input_list_is_not_mutated(self) -> None:
        from tools.memory.retrieval.supersession import apply_supersession_scoring
        from tools.memory.retrieval.fusion import FusedChunk

        content = (
            "[Document: dep.md | Section: memory | Date: unknown]\n\n---\n"
            "name: DEPRECATED — old claim\n---\n**THIS MEMORY IS DEPRECATED.**"
        )
        chunk = FusedChunk(
            chunk_id="dep", fused_score=0.9, dense_score=0.9, sparse_score=None,
            dense_norm=0.9, sparse_norm=0.0, content=content, file_path="dep.md",
        )
        original = [chunk]
        apply_supersession_scoring(original)
        assert original[0].fused_score == 0.9  # frozen dataclass, unchanged


class TestFusionEnvGating:
    """Item: fuse() only applies supersession scoring when explicitly enabled."""

    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.delenv("HARS_MEMORY_SUPERSESSION_SCORING", raising=False)
        dense = {
            "dep": ChannelHit(
                score=0.9,
                content=(
                    "[Document: dep.md | Section: memory | Date: unknown]\n\n---\n"
                    "name: DEPRECATED — old claim\n---\n**THIS MEMORY IS DEPRECATED.**"
                ),
                file_path="dep.md",
            ),
            "cur": ChannelHit(
                score=0.5,
                content="[Document: cur.md | Section: memory | Date: unknown]\n\nCurrent.",
                file_path="cur.md",
            ),
        }
        fused = fuse(dense, {}, alpha=1.0)
        # Without the flag, raw dense ranking wins: "dep" (0.9) stays on top.
        assert fused[0].chunk_id == "dep"

    def test_enabled_via_env_reorders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_SCORING", "1")
        # THREE dense hits (not two): with min-max normalization a two-item
        # channel's weaker hit always normalizes to exactly 0.0 (same as "no
        # hit"), which would make this assertion pass/fail for the wrong
        # reason. A third, genuinely-weakest hit ("floor") absorbs that floor
        # instead — same fixture-design note as fusion.py's own
        # TestFusionMath in test_bm25_retrieval.py.
        dense = {
            "dep": ChannelHit(
                score=0.9,
                content=(
                    "[Document: dep.md | Section: memory | Date: unknown]\n\n---\n"
                    "name: DEPRECATED — old claim\n---\n**THIS MEMORY IS DEPRECATED.**"
                ),
                file_path="dep.md",
            ),
            "cur": ChannelHit(
                score=0.5,
                content="[Document: cur.md | Section: memory | Date: unknown]\n\nCurrent.",
                file_path="cur.md",
            ),
            "floor": ChannelHit(score=0.1, content="floor", file_path="floor.md"),
        }
        fused = fuse(dense, {}, alpha=1.0)
        assert fused[0].chunk_id == "cur"
