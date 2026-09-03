import httpx
import pytest

from hars_memory.server.adapter_mode_guard import (
    ADAPTER_HEALTH_URL_ENV,
    assert_local_mode_adapter_if_configured,
)


def _fake_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload, request=httpx.Request("GET", "http://x/health"))


def test_no_op_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ADAPTER_HEALTH_URL_ENV, raising=False)

    def _boom(*_args: object, **_kwargs: object) -> httpx.Response:
        raise AssertionError("must not make a network call when the guard is not configured")

    monkeypatch.setattr(httpx, "get", _boom)
    assert_local_mode_adapter_if_configured()  # must not raise


def test_passes_when_adapter_reports_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ADAPTER_HEALTH_URL_ENV, "http://127.0.0.1:4243/health")
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: _fake_response({"ok": True, "llmProvider": "local"})
    )
    assert_local_mode_adapter_if_configured()  # must not raise


def test_refuses_when_adapter_reports_server_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ADAPTER_HEALTH_URL_ENV, "http://127.0.0.1:4243/health")
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: _fake_response({"ok": True, "llmProvider": "server"})
    )
    with pytest.raises(SystemExit):
        assert_local_mode_adapter_if_configured()


def test_refuses_when_llm_provider_field_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ADAPTER_HEALTH_URL_ENV, "http://127.0.0.1:4243/health")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_response({"ok": True}))
    with pytest.raises(SystemExit):
        assert_local_mode_adapter_if_configured()


def test_refuses_when_adapter_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ADAPTER_HEALTH_URL_ENV, "http://127.0.0.1:4243/health")

    def _raise(*_args: object, **_kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _raise)
    with pytest.raises(SystemExit):
        assert_local_mode_adapter_if_configured()


def test_refuses_on_non_2xx_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ADAPTER_HEALTH_URL_ENV, "http://127.0.0.1:4243/health")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_response({}, status_code=500))
    with pytest.raises(SystemExit):
        assert_local_mode_adapter_if_configured()
