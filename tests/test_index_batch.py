"""Tests for `memory index-batch` (ingest/batch.py) with a fake Batch backend.

No network and no model downloads: the embedder and the synchronous extraction
LLM are replaced by deterministic fakes, but LightRAG itself is real, so the
tests prove the point of the design — answers written into the LLM response
cache by `apply` are really found by LightRAG's own cache lookup.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hars_memory import cli
from hars_memory.ingest import batch
from hars_memory.ingest.document import Document, SourceKind

DOC_A = (
    "# Alpha guide\n\nThe alpha service talks to the beta service.\n\n"
    "## Install\n\nInstall alpha with the installer.\n\n"
    "## Usage\n\n| flag | meaning |\n| --- | --- |\n| -v | verbose |\n\nUse alpha daily.\n"
)
DOC_B = "# Beta notes\n\n## Install\n\nBeta needs a database.\n\n## Ops\n\nRestart beta nightly.\n"

EXTRACTION = (
    "entity<|#|>Alpha<|#|>Concept<|#|>The alpha service\n"
    "entity<|#|>Beta<|#|>Concept<|#|>The beta service\n"
    "relation<|#|>Alpha<|#|>Beta<|#|>talks to<|#|>alpha talks to beta\n"
    "<|COMPLETE|>"
)


def _doc(name: str, content: str) -> Document:
    return Document(
        doc_id=f"doc-{name}",
        content=content,
        source_kind=SourceKind.MARKDOWN,
        source_path=f"/fake/{name}.md",
        metadata={"relative_path": f"{name}.md"},
    )


@pytest.fixture()
def docs() -> list[Document]:
    return [_doc("alpha", DOC_A), _doc("beta", DOC_B)]


class FakeBackend:
    """In-memory Batch API: answers every request with EXTRACTION (or an error)."""

    def __init__(self, fail_every: int = 0, finish_after_polls: int = 1) -> None:
        self.files: dict[str, str] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self.uploads = 0
        self.creates = 0
        self.fail_every = fail_every
        self.finish_after_polls = finish_after_polls
        self.polls: dict[str, int] = {}

    def upload(self, path: Path) -> str:
        self.uploads += 1
        file_id = f"file-{self.uploads}"
        self.files[file_id] = path.read_text(encoding="utf-8")
        return file_id

    def create(self, file_id: str, metadata: dict[str, str]) -> dict[str, Any]:
        self.creates += 1
        batch_id = f"batch-{self.creates}"
        self.batches[batch_id] = {"file_id": file_id, "metadata": metadata}
        return {"id": batch_id, "status": "validating", "counts": {}}

    def get(self, batch_id: str) -> dict[str, Any]:
        self.polls[batch_id] = self.polls.get(batch_id, 0) + 1
        if self.polls[batch_id] < self.finish_after_polls:
            return {"id": batch_id, "status": "in_progress", "counts": {}}
        info = self.batches[batch_id]
        out_id = f"out-{batch_id}"
        if out_id not in self.files:
            self.files[out_id] = self._answer(self.files[info["file_id"]])
        return {
            "id": batch_id, "status": "completed", "output_file_id": out_id,
            "error_file_id": None, "counts": {"total": 1, "completed": 1, "failed": 0},
        }

    def read(self, file_id: str) -> str:
        return self.files[file_id]

    def _answer(self, jsonl: str) -> str:
        out = []
        for n, line in enumerate(jsonl.splitlines()):
            req = json.loads(line)
            assert req["method"] == "POST" and req["url"] == "/v1/chat/completions"
            assert req["body"]["messages"][-1]["role"] == "user"
            if self.fail_every and n % self.fail_every == 0:
                out.append(json.dumps({
                    "custom_id": req["custom_id"], "response": None,
                    "error": {"code": "boom", "message": "failed"},
                }))
                continue
            body = {
                "choices": [{"message": {"role": "assistant", "content": EXTRACTION}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
            out.append(json.dumps({
                "custom_id": req["custom_id"], "error": None,
                "response": {"status_code": 200, "body": body},
            }))
        return "\n".join(out) + "\n"


@pytest.fixture()
def lightrag_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Real LightRAG, fake embedder + fake sync LLM (counting calls)."""
    # import_module (sys.modules) rather than `from hars_memory.server import ...`:
    # other tests reload these modules and the package attribute can be stale.
    embedder = importlib.import_module("hars_memory.server.embedder")
    lightrag_init = importlib.import_module("hars_memory.server.lightrag_init")

    sync_calls: list[str] = []

    async def fake_embed(texts: list[str], **_: object) -> np.ndarray:
        return np.array([[float(len(t) % 7), 1.0, 0.5, float(len(t) % 3)] for t in texts], dtype=np.float32)

    async def fake_llm(prompt: str, system_prompt: str | None = None, **_: object) -> str:
        sync_calls.append(prompt[:20])
        return EXTRACTION

    monkeypatch.setattr(embedder, "embedding_dimension", lambda *_a, **_k: 4)
    monkeypatch.setattr(embedder, "validate_embedder_against_index", lambda *a, **k: None)
    monkeypatch.setattr(embedder, "make_embedding_func", lambda **k: fake_embed)
    monkeypatch.setattr(lightrag_init, "make_llm_func", lambda **k: fake_llm)
    monkeypatch.delenv("HARS_MEMORY_RERANK_MODEL", raising=False)
    monkeypatch.setenv("HARS_MEMORY_CHUNKER", "markdown")
    monkeypatch.setenv("HARS_MEMORY_CHUNK_TOKEN_SIZE", "64")
    monkeypatch.setenv("HARS_MEMORY_EXTRACTOR_MODEL", "fake/model")
    monkeypatch.setenv("HARS_MEMORY_MAX_GLEANING", "1")
    yield sync_calls
    from lightrag.kg.shared_storage import finalize_share_data

    finalize_share_data()


def _index(tmp_path: Path, name: str = "idx") -> Path:
    return tmp_path / name


def _run_all(index_dir: Path, docs: list[Document], backend: FakeBackend, **kw: Any) -> dict[str, Any]:
    while batch.next_round(batch.BatchState.load(index_dir)) is not None:
        batch.collect(index_dir, docs, **kw)
        batch.submit(index_dir, backend)
        batch.wait(index_dir, backend, poll_seconds=0, timeout_seconds=5, sleep=lambda _s: None)
    return batch.apply(index_dir, docs)


def test_request_key_matches_lightrag_cache_key() -> None:
    from lightrag.utils import compute_args_hash, generate_cache_key

    history = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    key, joined, messages = batch.request_key("user text", "system text", history)
    expected = generate_cache_key(
        "default", "extract",
        compute_args_hash("user text\nsystem text\n" + json.dumps(history, ensure_ascii=False)),
    )
    assert key == expected
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "user text"
    assert joined.startswith("user text\nsystem text\n")


def test_collect_dry_run_writes_nothing_and_estimates(lightrag_env, docs, tmp_path) -> None:
    idx = _index(tmp_path)
    rnd = batch.collect(idx, docs, dry_run=True)
    assert rnd is not None and rnd.round == 1 and rnd.requests >= 2
    assert rnd.est_input_tokens > 0 and rnd.est_cost_usd > 0
    assert not (idx / batch.STATE_FILENAME).exists()
    assert not (idx / batch.BATCH_DIRNAME / "_collect").exists()


def test_cost_guard_aborts_before_anything_is_written(lightrag_env, docs, tmp_path) -> None:
    idx = _index(tmp_path)
    with pytest.raises(batch.CostGuardError):
        batch.collect(idx, docs, max_cost_usd=0.0)
    assert not (idx / batch.STATE_FILENAME).exists()


def test_collect_round1_only_has_first_pass_prompts(lightrag_env, docs, tmp_path) -> None:
    idx = _index(tmp_path)
    rnd = batch.collect(idx, docs)
    rows = [json.loads(line) for line in (idx / batch.BATCH_DIRNAME / rnd.requests_file).read_text().splitlines()]
    assert len(rows) == rnd.requests
    assert all(r["round"] == 1 and len(r["messages"]) == 2 for r in rows)
    assert all(r["chunk_id"].startswith("chunk-") for r in rows)
    state = batch.BatchState.load(idx)
    assert state.gleaning == 1 and state.chunker == "markdown" and state.model == "fake/model"
    assert batch.next_round(state) is None  # round 1 not finished => gleaning not collectable yet


def test_collect_is_idempotent_per_round(lightrag_env, docs, tmp_path) -> None:
    idx = _index(tmp_path)
    batch.collect(idx, docs)
    assert batch.collect(idx, docs) is None  # nothing new until round 1 finishes


def test_submit_is_idempotent_and_resumable(lightrag_env, docs, tmp_path) -> None:
    idx, backend = _index(tmp_path), FakeBackend()
    batch.collect(idx, docs)
    first = batch.submit(idx, backend)
    assert len(first) == 1 and backend.creates == 1
    assert batch.submit(idx, backend) == []  # second call: already submitted
    assert backend.creates == 1 and backend.uploads == 1
    state = batch.BatchState.load(idx)  # state survives a "restart"
    assert state.rounds["1"].parts[0].batch_id == "batch-1"


def test_submit_splits_parts_by_request_cap(lightrag_env, docs, tmp_path) -> None:
    idx, backend = _index(tmp_path), FakeBackend()
    rnd = batch.collect(idx, docs)
    batch.submit(idx, backend, max_requests_per_batch=2)
    parts = batch.BatchState.load(idx).rounds["1"].parts
    assert len(parts) == -(-rnd.requests // 2)
    assert sum(p.requests for p in parts) == rnd.requests


def test_full_run_primes_cache_so_no_extraction_hits_the_sync_llm(lightrag_env, docs, tmp_path) -> None:
    idx, backend = _index(tmp_path), FakeBackend()
    stats = _run_all(idx, docs, backend)
    assert lightrag_env == []  # the synchronous LLM was never called
    assert stats["sync_llm_calls"] == 0
    assert stats["cache_hits"] == stats["total_requests"] == stats["answered"]
    assert set(batch.BatchState.load(idx).rounds) == {"1", "2"}  # gleaning round was collected too
    s = batch.index_stats(idx)
    assert s["docs_processed"] == 2 and s["entities"] >= 2 and s["relations"] >= 1
    assert s["chunks_with_heading_path"] == s["chunks"] > 0
    assert s["llm_cache_by_type"].get("extract") == stats["total_requests"]


def test_gleaning_zero_collects_a_single_round(lightrag_env, docs, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HARS_MEMORY_MAX_GLEANING", "0")
    idx = _index(tmp_path)
    stats = _run_all(idx, docs, FakeBackend())
    assert set(batch.BatchState.load(idx).rounds) == {"1"}
    assert lightrag_env == [] and stats["sync_llm_calls"] == 0


def test_error_lines_fall_back_to_the_sync_llm_and_are_counted(lightrag_env, docs, tmp_path) -> None:
    idx = _index(tmp_path)
    stats = _run_all(idx, docs, FakeBackend(fail_every=2))
    assert stats["error_lines"] > 0
    assert stats["expected_sync_fallback"] == stats["total_requests"] - stats["answered"] > 0
    assert len(lightrag_env) > 0  # the failed ones were answered synchronously
    assert batch.index_stats(idx)["docs_processed"] == 2


def test_apply_refuses_while_batches_are_unfinished(lightrag_env, docs, tmp_path) -> None:
    idx, backend = _index(tmp_path), FakeBackend(finish_after_polls=99)
    batch.collect(idx, docs)
    batch.submit(idx, backend)
    state = batch.refresh(idx, backend)
    assert state.rounds["1"].parts[0].status == "in_progress"
    with pytest.raises(batch.BatchError, match="not all finished"):
        batch.apply(idx, docs)


def test_wait_times_out_and_leaves_resumable_state(lightrag_env, docs, tmp_path) -> None:
    idx, backend = _index(tmp_path), FakeBackend(finish_after_polls=99)
    batch.collect(idx, docs)
    batch.submit(idx, backend)
    state = batch.wait(idx, backend, poll_seconds=0, timeout_seconds=0, sleep=lambda _s: None)
    assert not state.rounds["1"].terminal()
    backend.finish_after_polls = 1  # provider finishes later; resume with the same state
    state = batch.wait(idx, backend, poll_seconds=0, timeout_seconds=5, sleep=lambda _s: None)
    assert state.rounds["1"].terminal()
    assert state.rounds["1"].parts[0].usage["prompt_tokens"] > 0


def test_document_set_change_is_rejected(lightrag_env, docs, tmp_path) -> None:
    idx = _index(tmp_path)
    batch.collect(idx, docs)
    with pytest.raises(batch.BatchError, match="document set differs"):
        batch.collect(idx, docs[:1])


def test_status_and_cost_report(lightrag_env, docs, tmp_path) -> None:
    idx, backend = _index(tmp_path), FakeBackend()
    batch.collect(idx, docs)
    batch.submit(idx, backend)
    state = batch.wait(idx, backend, poll_seconds=0, timeout_seconds=5, sleep=lambda _s: None)
    text = batch.format_status(state)
    assert "batch-1" in text and "completed" in text
    cost = batch.actual_cost(state)
    assert cost["prompt_tokens"] > 0
    assert cost["sync_equivalent_usd"] == pytest.approx(2 * cost["batch_cost_usd"])


def test_cli_collect_cost_guard_exit_code(lightrag_env, docs, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_batch_documents", lambda args: docs)
    with pytest.raises(SystemExit) as exc:
        cli.main(["index-batch", "collect", "--index-dir", str(_index(tmp_path)), "--max-cost", "0"])
    assert exc.value.code == 3
    assert "exceeds --max-cost" in capsys.readouterr().err


def test_provider_model_strips_only_the_openai_routing_prefix() -> None:
    assert batch.provider_model("openai/gpt-6-luna") == "gpt-6-luna"
    assert batch.provider_model("gpt-6-luna") == "gpt-6-luna"
    assert batch.provider_model("fake/model") == "fake/model"


def test_token_split_and_inflight_window_submit_parts_in_waves(lightrag_env, docs, tmp_path) -> None:
    idx, backend = _index(tmp_path), FakeBackend()
    rnd = batch.collect(idx, docs)
    per_part = rnd.est_input_tokens // 3  # forces several parts
    created = batch.submit(idx, backend, max_tokens_per_batch=per_part, max_inflight_tokens=per_part)
    parts = batch.BatchState.load(idx).rounds["1"].parts
    assert len(parts) >= 2
    assert len(created) == 1 and backend.creates == 1  # the rest wait for capacity
    state = batch.wait(
        idx, backend, poll_seconds=0, timeout_seconds=5, sleep=lambda _s: None,
        after_poll=lambda: batch.submit(idx, backend, max_tokens_per_batch=per_part, max_inflight_tokens=per_part),
    )
    assert state.rounds["1"].terminal() and backend.creates == len(parts)


def test_cli_refuses_live_and_foreign_index_dirs(tmp_path, monkeypatch) -> None:
    from hars_memory import cli

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "kv_store_doc_status.json").write_text("{}")
    with pytest.raises(SystemExit, match="did not create"):
        cli.main(["index-batch", "collect", "--dry-run", "--index-dir", str(foreign)])

    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setenv("HARS_MEMORY_LIVE_INDEX_DIR", str(live))
    with pytest.raises(SystemExit, match="live index"):
        cli.main(["index-batch", "run", "--index-dir", str(live)])

    default_live = Path("~/.local/share/hars-longterm-memory/index").expanduser()
    with pytest.raises(SystemExit, match="live index"):
        cli.main(["index-batch", "apply", "--index-dir", str(default_live)])

    # a dir index-batch itself started stays usable (resumable)
    (foreign / "batch_state.json").write_text("{}")
    cli._check_not_live_index(foreign.resolve())
