"""Tests for server/gpu_guard.py's fail-CLOSED-on-unreachable-backend behaviour.

No real network calls — ``requests.get`` is monkeypatched to simulate an
unreachable backend, a busy GPU, and a free GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools.memory.server.gpu_guard import (
    GpuBusyError,
    GpuGuardUnavailableError,
    assert_gpu_free,
    check_gpu_concurrency,
)


def _raise_connection_error(*_args: object, **_kwargs: object) -> None:
    import requests

    raise requests.ConnectionError("backend unreachable")


class TestCheckGpuConcurrencyFailsClosed:
    def test_unreachable_backend_raises_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "tools.memory.server.gpu_guard.requests.get", _raise_connection_error
        )
        with pytest.raises(GpuGuardUnavailableError):
            check_gpu_concurrency("http://localhost:8765")

    def test_unreachable_backend_with_override_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "tools.memory.server.gpu_guard.requests.get", _raise_connection_error
        )
        result = check_gpu_concurrency(
            "http://localhost:8765", allow_unreachable_backend=True
        )
        assert result is None

    def test_reachable_backend_no_running_workflow_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = []
        monkeypatch.setattr(
            "tools.memory.server.gpu_guard.requests.get", lambda *a, **k: resp
        )
        result = check_gpu_concurrency("http://localhost:8765")
        assert result is None

    def test_reachable_backend_busy_gpu_returns_tuple(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = [
            {"id": "exp-1", "workflow_type": "vea", "status": "running"}
        ]
        monkeypatch.setattr(
            "tools.memory.server.gpu_guard.requests.get", lambda *a, **k: resp
        )
        result = check_gpu_concurrency("http://localhost:8765")
        assert result == ("exp-1", "vea")


class TestAssertGpuFree:
    def test_unreachable_backend_raises_gpu_guard_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "tools.memory.server.gpu_guard.requests.get", _raise_connection_error
        )
        with pytest.raises(GpuGuardUnavailableError):
            assert_gpu_free("http://localhost:8765")

    def test_unreachable_backend_override_proceeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "tools.memory.server.gpu_guard.requests.get", _raise_connection_error
        )
        # Must not raise.
        assert_gpu_free("http://localhost:8765", allow_unreachable_backend=True)

    def test_busy_gpu_raises_gpu_busy_error_even_with_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """allow_unreachable_backend must only affect the unreachable case,
        never mask an actually-busy GPU."""
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = [
            {"id": "exp-1", "workflow_type": "expert", "status": "running"}
        ]
        monkeypatch.setattr(
            "tools.memory.server.gpu_guard.requests.get", lambda *a, **k: resp
        )
        with pytest.raises(GpuBusyError):
            assert_gpu_free("http://localhost:8765", allow_unreachable_backend=True)
