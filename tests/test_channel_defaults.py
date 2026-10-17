"""Pin the code defaults of the optional retrieval channels.

``HARS_MEMORY_RIPGREP_CHANNEL`` and ``HARS_MEMORY_FLAT_CHANNEL`` each have a
code default in ``hars_memory/retrieval/fusion.py`` and an override in the
live serving env (``cortex/.mcp.json``).  The audit found the two silently
disagreeing for ripgrep (code default ON, live env ``=0``); these tests make
the code default explicit so a future edit cannot reintroduce the mismatch
without failing here.
"""

from __future__ import annotations

import pytest

from hars_memory.retrieval import fusion


def test_ripgrep_channel_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Code default is OFF, matching ``cortex/.mcp.json``'s
    ``HARS_MEMORY_RIPGREP_CHANNEL=0`` (audit 2026-10-17)."""
    assert fusion._RIPGREP_CHANNEL_DEFAULT == "0"

    monkeypatch.delenv(fusion.HARS_MEMORY_RIPGREP_CHANNEL_ENV, raising=False)
    assert fusion.ripgrep_channel_enabled() is False

    monkeypatch.setenv(fusion.HARS_MEMORY_RIPGREP_CHANNEL_ENV, "1")
    assert fusion.ripgrep_channel_enabled() is True

    monkeypatch.setenv(fusion.HARS_MEMORY_RIPGREP_CHANNEL_ENV, "0")
    assert fusion.ripgrep_channel_enabled() is False


def test_ripgrep_channel_env_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env override wins over the code default, both directions."""
    for raw, expected in (("1", True), ("true", True), ("0", False), ("off", False)):
        monkeypatch.setenv(fusion.HARS_MEMORY_RIPGREP_CHANNEL_ENV, raw)
        assert fusion.ripgrep_channel_enabled() is expected


def test_flat_channel_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flat dense stays backend-disabled by default (redundant with the dense
    channel; see ``fusion.py``'s flat-channel DECISION note)."""
    assert fusion._FLAT_CHANNEL_DEFAULT == "0"

    monkeypatch.delenv(fusion.HARS_MEMORY_FLAT_CHANNEL_ENV, raising=False)
    assert fusion.flat_channel_enabled() is False
