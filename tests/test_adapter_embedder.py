import httpx
import pytest

from hars_memory.server.adapter_embedder import (
    adapter_embedding_dimension,
    make_adapter_embedding_func,
)

ADAPTER_URL_ENV = "HARS_MEMORY_EMBED_ADAPTER_URL"


def _fake_sync_response(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://x/embeddings"))


class _FakeAsyncClient:
    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[dict] = []

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, url: str, json: dict) -> httpx.Response:
        self.calls.append({"url": url, "json": json})
        return self._responder(json)


def test_dimension_probe_reads_vector_length(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ADAPTER_URL_ENV, "http://127.0.0.1:4243/internal/embeddings/v1/embeddings")
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _fake_sync_response({"data": [{"embedding": [0.0] * 384}]}),
    )

    assert adapter_embedding_dimension() == 384


def test_dimension_probe_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ADAPTER_URL_ENV, raising=False)

    with pytest.raises(RuntimeError, match="HARS_MEMORY_EMBED_BACKEND=adapter"):
        adapter_embedding_dimension()


@pytest.mark.asyncio
async def test_embed_func_sends_batches_and_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ADAPTER_URL_ENV, "http://127.0.0.1:4243/internal/embeddings/v1/embeddings")
    monkeypatch.setenv("HARS_MEMORY_EMBED_MODEL", "local-model")

    fake_client_holder: list[_FakeAsyncClient] = []

    def responder(sent_json: dict) -> httpx.Response:
        texts = sent_json["input"]
        return _fake_sync_response(
            {"data": [{"embedding": [float(len(t)), 0.0]} for t in texts]}
        )

    def make_client(*_args: object, **_kwargs: object) -> _FakeAsyncClient:
        client = _FakeAsyncClient(responder)
        fake_client_holder.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", make_client)

    embed = make_adapter_embedding_func(batch_size=2)
    result = await embed(["a", "bb", "ccc"])

    assert result.shape == (3, 2)
    assert result[0].tolist() == [1.0, 0.0]
    assert result[1].tolist() == [2.0, 0.0]
    assert result[2].tolist() == [3.0, 0.0]
    # batch_size=2 over 3 texts -> two upstream calls (2 + 1), not three or one
    assert len(fake_client_holder[0].calls) == 2
    assert fake_client_holder[0].calls[0]["json"]["model"] == "local-model"


def test_embed_func_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ADAPTER_URL_ENV, raising=False)

    with pytest.raises(RuntimeError, match="HARS_MEMORY_EMBED_BACKEND=adapter"):
        make_adapter_embedding_func()
