"""Tests for the bounded retry/backoff wrapping ``make_llm_func``'s LLM call
in ``server/lightrag_init.py``.

Context: neither of LightRAG 1.4.16's own decorators retries this call path.
``lightrag.llm.openai.openai_complete_if_cache`` has a tenacity ``@retry`` on
``RateLimitError | APIConnectionError | APITimeoutError | InvalidResponseError``,
but this module never calls that function — it hand-rolls its own httpx-based
``llm_func`` instead. The only decorator that actually wraps our call
(``lightrag.utils.priority_limit_async_func_call``, source of the "Error in
decorated function" log line) is a concurrency limiter with multi-layer
*timeout* protection, not a retry mechanism — on any exception it just logs
and propagates. So a single transient failure (e.g. llama-server briefly
refusing connections during a prompt-cache reorganisation pass) used to fail
the whole document's merge stage with no retry at all.

These tests substitute ``httpx.AsyncClient`` with one bound to an
``httpx.MockTransport`` so no real network/process is involved.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tools.memory.server.lightrag_init import make_llm_func

# Captured before any test monkeypatches ``httpx.AsyncClient`` — the fake
# client factory below constructs real ``httpx.AsyncClient`` instances (just
# pinned to a mock transport), so it must not call back into itself.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _client_factory_using_transport(
    transport: httpx.MockTransport,
) -> Callable[..., httpx.AsyncClient]:
    """Build an ``httpx.AsyncClient`` factory pinned to a mock transport.

    Mirrors ``make_llm_func``'s call shape (``httpx.AsyncClient(timeout=...)``)
    but routes all requests through ``transport`` instead of the network.
    """

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    return _factory


def _chat_response(content: str = "ok") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _make_llm_func_for_test() -> object:
    return make_llm_func(
        base_url="http://localhost:8083/v1",
        model="gemma-4-12b",
        max_tokens=64,
        temperature=0.1,
    )


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink retry backoff so retry tests run in milliseconds, not seconds.

    Production defaults are 1s / 8s; these values just need to be > 0 so the
    retry/backoff code path (not merely "retry with zero delay") is exercised.
    """
    monkeypatch.setenv("HARS_MEMORY_LLM_RETRY_BACKOFF_INITIAL_SECONDS", "0.001")
    monkeypatch.setenv("HARS_MEMORY_LLM_RETRY_BACKOFF_MAX_SECONDS", "0.002")


class TestMakeLlmFuncRetriesTransientFailures:
    def test_retries_connection_error_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                # Simulates llama-server refusing/dropping a connection mid
                # prompt-cache reorganisation pass.
                raise httpx.ConnectError("connection refused", request=request)
            return _chat_response("recovered")

        monkeypatch.setattr(
            httpx, "AsyncClient", _client_factory_using_transport(httpx.MockTransport(handler))
        )

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert result == "recovered"
        assert calls["n"] == 3

    def test_retries_503_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(503, text="server busy")
            return _chat_response("ok-after-503")

        monkeypatch.setattr(
            httpx, "AsyncClient", _client_factory_using_transport(httpx.MockTransport(handler))
        )

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert result == "ok-after-503"
        assert calls["n"] == 2

    def test_retries_429_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(429, text="rate limited")
            return _chat_response("ok-after-429")

        monkeypatch.setattr(
            httpx, "AsyncClient", _client_factory_using_transport(httpx.MockTransport(handler))
        )

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert result == "ok-after-429"
        assert calls["n"] == 2

    def test_logs_a_warning_per_retry_attempt(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectError("connection refused", request=request)
            return _chat_response("recovered")

        monkeypatch.setattr(
            httpx, "AsyncClient", _client_factory_using_transport(httpx.MockTransport(handler))
        )

        llm_func = _make_llm_func_for_test()
        with caplog.at_level(logging.WARNING, logger="tools.memory.server.lightrag_init"):
            asyncio.run(llm_func("hello"))  # type: ignore[operator]

        retry_records = [r for r in caplog.records if "transient failure" in r.message]
        assert len(retry_records) == 2  # 2 failures before the 3rd (successful) attempt
        assert "gemma-4-12b" in retry_records[0].message
        assert "retrying in" in retry_records[0].message


class TestMakeLlmFuncDoesNotRetryFatalErrors:
    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
    def test_does_not_retry_non_429_4xx(
        self, monkeypatch: pytest.MonkeyPatch, status_code: int
    ) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(status_code, text="fatal client error")

        monkeypatch.setattr(
            httpx, "AsyncClient", _client_factory_using_transport(httpx.MockTransport(handler))
        )

        llm_func = _make_llm_func_for_test()

        with pytest.raises(RuntimeError, match="LLM endpoint unavailable"):
            asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert calls["n"] == 1  # no retry attempted — this can never succeed

    def test_runtime_error_chains_the_original_httpx_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

        monkeypatch.setattr(
            httpx, "AsyncClient", _client_factory_using_transport(httpx.MockTransport(handler))
        )

        llm_func = _make_llm_func_for_test()

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)
        assert exc_info.value.__cause__.response.status_code == 401


class TestMakeLlmFuncGivesUpAfterBoundedAttempts:
    def test_gives_up_after_max_attempts_and_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            # Permanently failing endpoint: must give up, not hang or retry
            # indefinitely.
            raise httpx.ConnectError("connection refused", request=request)

        monkeypatch.setattr(
            httpx, "AsyncClient", _client_factory_using_transport(httpx.MockTransport(handler))
        )
        monkeypatch.setenv("HARS_MEMORY_LLM_RETRY_ATTEMPTS", "3")

        llm_func = _make_llm_func_for_test()

        with pytest.raises(RuntimeError, match="LLM endpoint unavailable"):
            asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert calls["n"] == 3  # exactly the configured cap — bounded, not infinite

    def test_retry_attempts_env_var_of_one_disables_retrying(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectError("connection refused", request=request)

        monkeypatch.setattr(
            httpx, "AsyncClient", _client_factory_using_transport(httpx.MockTransport(handler))
        )
        monkeypatch.setenv("HARS_MEMORY_LLM_RETRY_ATTEMPTS", "1")

        llm_func = _make_llm_func_for_test()

        with pytest.raises(RuntimeError, match="LLM endpoint unavailable"):
            asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert calls["n"] == 1
