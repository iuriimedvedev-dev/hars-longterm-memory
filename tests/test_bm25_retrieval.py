"""Unit tests for tools/memory/retrieval — BM25 sparse index, tokenizer,
and dense/sparse fusion. GPU-free, no LLM calls, no network.

Covers the acceptance criteria from the hybrid-retrieval feature spec:
- identifier tokenization preserves whole identifiers (`phase_c1_lora_safe`,
  `A2S32`) rather than stemming/stripping them away.
- fusion math is correct at alpha=0 (pure sparse) and alpha=1 (pure dense).
- the on-disk BM25 index cache invalidates when the source
  kv_store_text_chunks.json mtime changes, and is reused when it doesn't.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tools.memory.retrieval.tokenizer import (
    extract_identifier_terms,
    looks_like_identifier,
    tokenize_identifiers,
)


class TestTokenizerPreservesIdentifiers:
    """Item: tokenization must not destroy the exact-identifier signal."""

    def test_whole_identifiers_survive_as_single_tokens(self) -> None:
        for identifier in ("A2S32", "phase_c1_lora_safe", "vea_native", "hyp:7f62cfb4"):
            tokens = tokenize_identifiers(identifier)
            assert identifier.casefold() in tokens, (
                f"whole identifier {identifier!r} must survive as one token; got {tokens}"
            )

    def test_sub_tokens_are_additive_not_replacing(self) -> None:
        tokens = tokenize_identifiers("phase_c1_lora_safe")
        assert "phase_c1_lora_safe" in tokens
        assert {"phase", "c1", "lora", "safe"} <= set(tokens)

    def test_alnum_code_splits_on_digit_letter_boundary_but_keeps_whole(self) -> None:
        tokens = tokenize_identifiers("A2S32")
        assert "a2s32" in tokens
        assert {"a2", "s32"} <= set(tokens)

    def test_whole_token_match_is_case_symmetric(self) -> None:
        # The primary guarantee: a query typed in different case than the
        # corpus must still match on the WHOLE identifier — casefold is
        # applied identically on both sides (tokenizer.py module docstring
        # item 4). This is the guarantee `looks_like_identifier`/BM25 exact
        # lookup actually relies on.
        assert "a2s32" in tokenize_identifiers("A2S32")
        assert "a2s32" in tokenize_identifiers("a2s32")
        assert "phase_c1_lora_safe" in tokenize_identifiers("Phase_C1_Lora_Safe")
        assert "phase_c1_lora_safe" in tokenize_identifiers("phase_c1_lora_safe")

    def test_underscore_split_sub_tokens_are_case_symmetric(self) -> None:
        # Delimiter-based sub-tokens (underscore/hyphen/colon/dot/slash) do not
        # depend on case at all, so they ARE fully symmetric.
        assert tokenize_identifiers("Phase_C1_Lora_Safe") == tokenize_identifiers(
            "phase_c1_lora_safe"
        )

    def test_camel_case_sub_split_is_not_recoverable_from_all_lowercase(self) -> None:
        # Documented limitation, not a bug: "A2S32" carries a real camelCase
        # boundary (digit '2' -> uppercase 'S') that "a2s32" structurally does
        # not contain — an all-lowercase string has no case-transition
        # information to split on. Sub-token recall differs; the whole-token
        # exact match (asserted above) does not.
        assert tokenize_identifiers("A2S32") != tokenize_identifiers("a2s32")
        assert {"a2", "s32"} <= set(tokenize_identifiers("A2S32"))
        assert set(tokenize_identifiers("a2s32")) == {"a2s32"}

    def test_no_stemming_digits_and_underscores_survive(self) -> None:
        tokens = tokenize_identifiers("phase_c1_lora_safe")
        # A stemmer/aggressive-normalizer would strip the digit or the
        # underscores; neither is acceptable for identifier retrieval.
        assert any("c1" in t for t in tokens)
        assert "phase_c1_lora_safe" in tokens  # underscores intact

    def test_prose_tokenizes_without_crashing_on_punctuation(self) -> None:
        text = "See A2S32 (also phase_c1_lora_safe), e.g. vea_native — done."
        tokens = tokenize_identifiers(text)
        assert "a2s32" in tokens
        assert "phase_c1_lora_safe" in tokens
        assert "vea_native" in tokens

    def test_empty_text_returns_empty_list(self) -> None:
        assert tokenize_identifiers("") == []


class TestIdentifierDetection:
    def test_recognizes_known_identifier_shapes(self) -> None:
        for identifier in ("A2S32", "phase_c1_lora_safe", "vea_native", "hyp:7f62cfb4"):
            assert looks_like_identifier(identifier), identifier

    def test_rejects_plain_words(self) -> None:
        for word in ("training", "the", "model", "results"):
            assert not looks_like_identifier(word), word

    def test_extract_identifier_terms_from_question(self) -> None:
        question = "What happened in phase_c1_lora_safe and does A2S32 relate to vea_native?"
        terms = extract_identifier_terms(question)
        assert terms == ["phase_c1_lora_safe", "A2S32", "vea_native"]

    def test_extract_identifier_terms_empty_for_plain_question(self) -> None:
        assert extract_identifier_terms("What happened in training yesterday?") == []


class TestFusionMath:
    """Item: fusion math correct at alpha=0 (pure sparse) and alpha=1 (pure dense).

    Fixture uses THREE hits per channel (not two): with min-max normalization,
    a two-item channel's weaker hit always normalizes to exactly 0.0 — the
    same value a chunk with NO hit in that channel defaults to — which makes a
    2-item fixture unable to distinguish "weak real signal" from "no signal"
    in assertions. A 3rd, genuinely-weakest hit absorbs that floor instead.
    """

    def _sample_hits(self):
        from tools.memory.retrieval.fusion import ChannelHit

        dense_hits = {
            "chunk-a": ChannelHit(score=0.9, content="dense top", file_path="a.md"),
            "chunk-b": ChannelHit(score=0.5, content="dense mid", file_path="b.md"),
            "chunk-d": ChannelHit(score=0.1, content="dense floor", file_path="d.md"),
        }
        sparse_hits = {
            "chunk-b": ChannelHit(score=10.0, content="sparse top", file_path="b.md"),
            "chunk-c": ChannelHit(score=5.0, content="sparse mid", file_path="c.md"),
            "chunk-e": ChannelHit(score=1.0, content="sparse floor", file_path="e.md"),
        }
        return dense_hits, sparse_hits

    def test_alpha_one_is_pure_dense_ranking(self) -> None:
        from tools.memory.retrieval.fusion import fuse

        dense_hits, sparse_hits = self._sample_hits()
        fused = fuse(dense_hits, sparse_hits, alpha=1.0)
        by_id = {c.chunk_id: c for c in fused}
        # chunk-a has the highest dense score -> must rank first, exactly as a
        # dense-only search would, regardless of chunk-b's dominant sparse score.
        assert fused[0].chunk_id == "chunk-a"
        assert fused[0].fused_score == pytest.approx(fused[0].dense_norm)
        # A chunk with genuine (if middling) dense signal must outrank a chunk
        # the dense channel never returned at all, even though the latter
        # (chunk-c) has strong sparse support — sparse contributes nothing at
        # alpha=1.
        assert by_id["chunk-b"].fused_score > by_id["chunk-c"].fused_score
        assert by_id["chunk-c"].fused_score == 0.0  # no dense hit -> zero at alpha=1

    def test_alpha_zero_is_pure_sparse_ranking(self) -> None:
        from tools.memory.retrieval.fusion import fuse

        dense_hits, sparse_hits = self._sample_hits()
        fused = fuse(dense_hits, sparse_hits, alpha=0.0)
        by_id = {c.chunk_id: c for c in fused}
        # chunk-b has the highest sparse score -> must rank first at alpha=0,
        # regardless of chunk-a's dominant dense score.
        assert fused[0].chunk_id == "chunk-b"
        assert fused[0].fused_score == pytest.approx(fused[0].sparse_norm)
        # A chunk with genuine sparse signal (chunk-c) must outrank a chunk
        # the sparse channel never returned at all (chunk-a) — dense
        # contributes nothing at alpha=0.
        assert by_id["chunk-c"].fused_score > by_id["chunk-a"].fused_score
        assert by_id["chunk-a"].fused_score == 0.0  # no sparse hit -> zero at alpha=0

    def test_alpha_half_blends_both_channels(self) -> None:
        from tools.memory.retrieval.fusion import fuse

        dense_hits, sparse_hits = self._sample_hits()
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        by_id = {c.chunk_id: c for c in fused}
        assert by_id["chunk-b"].fused_score == pytest.approx(
            0.5 * by_id["chunk-b"].dense_norm + 0.5 * by_id["chunk-b"].sparse_norm
        )

    def test_invalid_alpha_raises(self) -> None:
        from tools.memory.retrieval.fusion import fuse

        dense_hits, sparse_hits = self._sample_hits()
        with pytest.raises(ValueError, match="alpha"):
            fuse(dense_hits, sparse_hits, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            fuse(dense_hits, sparse_hits, alpha=-0.1)

    def test_tied_scores_normalize_to_neutral_midpoint(self) -> None:
        from tools.memory.retrieval.fusion import ChannelHit, fuse

        dense_hits = {
            "chunk-a": ChannelHit(score=0.5, content="x", file_path="a.md"),
            "chunk-b": ChannelHit(score=0.5, content="y", file_path="b.md"),
        }
        fused = fuse(dense_hits, {}, alpha=1.0)
        assert all(c.dense_norm == pytest.approx(0.5) for c in fused)

    def test_empty_channels_do_not_crash(self) -> None:
        from tools.memory.retrieval.fusion import fuse

        assert fuse({}, {}, alpha=0.5) == []


class TestRipgrepFusion:
    """Item: ripgrep-channel fusion wiring (retrieval/fusion.py's
    `apply_ripgrep_gate` / `fuse_ripgrep_as_third_weight` /
    `ripgrep_channel_enabled`). See `apply_ripgrep_gate`'s own module-level
    design note in fusion.py for the measured rationale these tests pin.
    """

    def _sample_fused(self):
        from tools.memory.retrieval.fusion import ChannelHit, fuse

        dense_hits = {
            "chunk-a": ChannelHit(score=0.9, content="dense top", file_path="a.md"),
            "chunk-b": ChannelHit(score=0.5, content="dense mid", file_path="b.md"),
        }
        sparse_hits = {
            "chunk-b": ChannelHit(score=10.0, content="sparse top", file_path="b.md"),
            "chunk-c": ChannelHit(score=5.0, content="sparse mid", file_path="c.md"),
        }
        return fuse(dense_hits, sparse_hits, alpha=0.5)

    def test_empty_ripgrep_hits_is_a_pure_passthrough(self) -> None:
        from tools.memory.retrieval.fusion import apply_ripgrep_gate

        fused = self._sample_fused()
        gated = apply_ripgrep_gate(fused, {}, top_k=2)
        assert gated == fused[:2]

    def test_injection_never_evicts_appends_past_top_k(self) -> None:
        from tools.memory.retrieval.fusion import ChannelHit, apply_ripgrep_gate

        fused = self._sample_fused()
        ripgrep_hits = {
            "off_index.md": ChannelHit(score=4.0, content="found only by rg", file_path="off_index.md"),
        }
        gated = apply_ripgrep_gate(fused, ripgrep_hits, top_k=2, enable_boost=False)
        # The real top-2 candidates are unchanged, in the same order/score.
        assert [c.chunk_id for c in gated[:2]] == [c.chunk_id for c in fused[:2]]
        assert gated[:2] == fused[:2]
        # The off-index file is appended, not substituted for anything.
        assert len(gated) == 3
        assert gated[2].chunk_id == "ripgrep:off_index.md"
        assert gated[2].file_path == "off_index.md"
        assert gated[2].ripgrep_score == 4.0

    def test_matching_is_by_basename_not_chunk_id(self) -> None:
        """The whole reason for basename matching: ripgrep's chunk_id
        (file_stable_id of whatever path it was given) essentially never
        equals a real dense/sparse chunk_id -- see fusion.py's design note.
        A ripgrep hit whose FILE PATH's basename matches an existing fused
        chunk's file_path basename must be treated as the SAME document even
        though the two `chunk_id`s are completely different strings.
        """
        from tools.memory.retrieval.fusion import ChannelHit, apply_ripgrep_gate

        fused = self._sample_fused()
        # "b.md" is already in the fused pool (chunk-b); ripgrep reports it
        # under an absolute path with a totally different chunk_id scheme.
        ripgrep_hits = {
            "b.md": ChannelHit(score=6.0, content="rg found b.md too", file_path="/abs/path/to/b.md"),
        }
        gated = apply_ripgrep_gate(fused, ripgrep_hits, top_k=3, enable_boost=True)
        matched = next(c for c in gated if c.chunk_id == "chunk-b")
        assert matched.ripgrep_score == 6.0
        # No new "ripgrep:b.md" entry was injected -- it was recognized as
        # the SAME document as chunk-b, not a distinct off-index one.
        assert not any(c.chunk_id == "ripgrep:b.md" for c in gated)
        assert len(gated) == 3  # still exactly the fused pool, no injection

    def test_boost_default_off_leaves_fused_score_untouched(self) -> None:
        from tools.memory.retrieval.fusion import ChannelHit, apply_ripgrep_gate

        fused = self._sample_fused()
        by_id = {c.chunk_id: c for c in fused}
        ripgrep_hits = {"b.md": ChannelHit(score=6.0, content="x", file_path="b.md")}
        gated = apply_ripgrep_gate(fused, ripgrep_hits, top_k=3)  # enable_boost defaults False
        matched = next(c for c in gated if c.chunk_id == "chunk-b")
        assert matched.fused_score == pytest.approx(by_id["chunk-b"].fused_score)
        assert matched.ripgrep_score == 6.0  # still recorded, even though score is untouched

    def test_boost_enabled_increases_fused_score_but_stays_bounded(self) -> None:
        from tools.memory.retrieval.fusion import (
            RIPGREP_BOOST_CAP,
            ChannelHit,
            apply_ripgrep_gate,
        )

        fused = self._sample_fused()
        by_id = {c.chunk_id: c for c in fused}
        # A huge raw ripgrep score must still saturate at RIPGREP_BOOST_CAP,
        # never overwhelm a real dense/sparse-earned fused_score.
        ripgrep_hits = {"b.md": ChannelHit(score=1_000_000.0, content="x", file_path="b.md")}
        gated = apply_ripgrep_gate(fused, ripgrep_hits, top_k=3, enable_boost=True)
        matched = next(c for c in gated if c.chunk_id == "chunk-b")
        assert matched.fused_score == pytest.approx(by_id["chunk-b"].fused_score + RIPGREP_BOOST_CAP)

    def test_max_injected_caps_off_index_appends(self) -> None:
        from tools.memory.retrieval.fusion import ChannelHit, apply_ripgrep_gate

        fused = self._sample_fused()
        ripgrep_hits = {
            f"off_{i}.md": ChannelHit(score=float(10 - i), content="x", file_path=f"off_{i}.md")
            for i in range(5)
        }
        gated = apply_ripgrep_gate(fused, ripgrep_hits, top_k=2, max_injected=2)
        injected = [c for c in gated if c.chunk_id.startswith("ripgrep:")]
        assert len(injected) == 2
        # Highest-scoring off-index hits win the limited slots.
        assert {c.chunk_id for c in injected} == {"ripgrep:off_0.md", "ripgrep:off_1.md"}

    def test_injection_disabled_never_appends(self) -> None:
        from tools.memory.retrieval.fusion import ChannelHit, apply_ripgrep_gate

        fused = self._sample_fused()
        ripgrep_hits = {"off_index.md": ChannelHit(score=9.0, content="x", file_path="off_index.md")}
        gated = apply_ripgrep_gate(fused, ripgrep_hits, top_k=2, enable_injection=False)
        assert len(gated) == 2
        assert not any(c.chunk_id.startswith("ripgrep:") for c in gated)

    def test_channel_enabled_env_default_and_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.memory.retrieval.fusion import (
            HARS_MEMORY_RIPGREP_CHANNEL_ENV,
            ripgrep_channel_enabled,
        )

        monkeypatch.delenv(HARS_MEMORY_RIPGREP_CHANNEL_ENV, raising=False)
        assert ripgrep_channel_enabled() is True  # measured default -- see fusion.py's comment

        monkeypatch.setenv(HARS_MEMORY_RIPGREP_CHANNEL_ENV, "0")
        assert ripgrep_channel_enabled() is False

        monkeypatch.setenv(HARS_MEMORY_RIPGREP_CHANNEL_ENV, "1")
        assert ripgrep_channel_enabled() is True


class TestRipgrepThirdWeightRejectedVariant:
    """Item: `fuse_ripgrep_as_third_weight` -- kept importable as a measured,
    falsifiable rejected alternative (see fusion.py's design note), not
    deleted. These tests pin its (deliberately naive, chunk_id-keyed)
    behavior, not endorse it for production use.
    """

    def test_weights_must_sum_to_one(self) -> None:
        from tools.memory.retrieval.fusion import fuse_ripgrep_as_third_weight

        with pytest.raises(ValueError, match="sum to 1.0"):
            fuse_ripgrep_as_third_weight({}, {}, {}, 0.5, 0.5, 0.5)

    def test_negative_weight_rejected(self) -> None:
        from tools.memory.retrieval.fusion import fuse_ripgrep_as_third_weight

        with pytest.raises(ValueError, match=">= 0.0"):
            fuse_ripgrep_as_third_weight({}, {}, {}, 1.2, -0.1, -0.1)

    def test_ripgrep_hit_with_non_overlapping_chunk_id_becomes_a_new_entry(self) -> None:
        """Demonstrates the id-space mismatch this variant was rejected for:
        a ripgrep hit keyed by its OWN chunk_id (not a real dense/sparse
        chunk_id) never merges with an existing entry -- it always shows up
        as a brand-new, dense_norm=sparse_norm=0 entry, exactly the
        distortion described in fusion.py's design note.
        """
        from tools.memory.retrieval.fusion import ChannelHit, fuse_ripgrep_as_third_weight

        dense_hits = {"chunk-a": ChannelHit(score=0.9, content="x", file_path="a.md")}
        ripgrep_hits = {
            "file:deadbeef0000": ChannelHit(score=5.0, content="rg only", file_path="fresh.md"),
        }
        fused = fuse_ripgrep_as_third_weight(dense_hits, {}, ripgrep_hits, 0.7, 0.2, 0.1)
        by_id = {c.chunk_id: c for c in fused}
        assert by_id["file:deadbeef0000"].dense_norm == 0.0
        assert by_id["file:deadbeef0000"].sparse_norm == 0.0
        assert by_id["file:deadbeef0000"].ripgrep_score == 5.0
        assert by_id["file:deadbeef0000"].fused_score > 0.0  # contributes via alpha_ripgrep alone


class TestBM25IndexCache:
    """Item: index cache invalidates on kv_store_text_chunks.json mtime change."""

    def _write_chunks(self, working_dir: Path, chunks: dict[str, dict]) -> None:
        (working_dir / "kv_store_text_chunks.json").write_text(
            json.dumps(chunks), encoding="utf-8"
        )

    def _sample_chunks(self) -> dict[str, dict]:
        return {
            "chunk-1": {
                "content": "Experiment A2S32 stabilized the gate at step 80.",
                "file_path": "hyp_a2s32.md",
            },
            "chunk-2": {
                "content": "phase_c1_lora_safe ran for 3000 steps without drift.",
                "file_path": "exp_phase_c1_lora_safe.md",
            },
            "chunk-3": {
                "content": "vea_native mode encodes video tokens at runtime.",
                "file_path": "feedback_vea_native.md",
            },
        }

    def test_missing_source_raises_unavailable(self, tmp_path: Path) -> None:
        from tools.memory.retrieval.bm25_index import (
            BM25IndexUnavailableError,
            get_or_build_index,
        )

        with pytest.raises(BM25IndexUnavailableError):
            get_or_build_index(str(tmp_path), str(tmp_path / "cache"))

    def test_build_index_indexes_expected_chunk_count(self, tmp_path: Path) -> None:
        from tools.memory.retrieval.bm25_index import build_index

        self._write_chunks(tmp_path, self._sample_chunks())
        index, build_seconds = build_index(str(tmp_path))
        assert index.chunk_count == 3
        assert build_seconds >= 0.0

    def test_search_finds_verbatim_identifier(self, tmp_path: Path) -> None:
        from tools.memory.retrieval.bm25_index import build_index

        self._write_chunks(tmp_path, self._sample_chunks())
        index, _ = build_index(str(tmp_path))
        hits = index.search("A2S32", top_k=3)
        assert hits, "expected at least one hit for A2S32"
        assert hits[0].chunk_id == "chunk-1"
        assert "A2S32" in hits[0].content

    def test_get_or_build_index_caches_on_disk(self, tmp_path: Path) -> None:
        from tools.memory.retrieval.bm25_index import get_or_build_index

        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        self._write_chunks(working_dir, self._sample_chunks())

        _index1, stats1 = get_or_build_index(str(working_dir), str(cache_dir))
        assert stats1.cache_hit is False
        assert stats1.chunk_count == 3
        assert (cache_dir / "bm25_cache_meta.json").is_file()

        _index2, stats2 = get_or_build_index(str(working_dir), str(cache_dir))
        assert stats2.cache_hit is True
        assert stats2.chunk_count == 3

    def test_cache_invalidates_on_mtime_change(self, tmp_path: Path) -> None:
        from tools.memory.retrieval.bm25_index import get_or_build_index

        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"
        self._write_chunks(working_dir, self._sample_chunks())

        _index1, stats1 = get_or_build_index(str(working_dir), str(cache_dir))
        assert stats1.cache_hit is False

        # Rewrite with new content AND force a distinct mtime (some filesystems
        # have coarse mtime granularity) — this simulates a re-ingested index.
        richer_chunks = self._sample_chunks()
        richer_chunks["chunk-4"] = {
            "content": "A new chunk added after re-ingest.",
            "file_path": "new.md",
        }
        self._write_chunks(working_dir, richer_chunks)
        future = time.time() + 5
        os.utime(chunks_file, (future, future))

        index2, stats2 = get_or_build_index(str(working_dir), str(cache_dir))
        assert stats2.cache_hit is False, "mtime changed — must rebuild, not reuse stale cache"
        assert index2.chunk_count == 4

    def test_reloaded_index_still_searches_correctly(self, tmp_path: Path) -> None:
        """The disk-persisted (cache_hit=True) path must return identical
        results to a fresh in-memory build — proves save/load round-trips the
        sparse matrix + chunk metadata correctly, not just the chunk count."""
        from tools.memory.retrieval.bm25_index import get_or_build_index

        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        self._write_chunks(working_dir, self._sample_chunks())

        index1, stats1 = get_or_build_index(str(working_dir), str(cache_dir))
        assert stats1.cache_hit is False
        hits1 = index1.search("phase_c1_lora_safe", top_k=3)

        index2, stats2 = get_or_build_index(str(working_dir), str(cache_dir))
        assert stats2.cache_hit is True
        hits2 = index2.search("phase_c1_lora_safe", top_k=3)

        assert [h.chunk_id for h in hits1] == [h.chunk_id for h in hits2]
        assert [h.score for h in hits1] == pytest.approx([h.score for h in hits2])
