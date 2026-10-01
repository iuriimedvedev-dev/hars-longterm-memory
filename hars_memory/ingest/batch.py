"""Batch-API extraction for the LightRAG index (`memory index-batch`).

Why this exists
---------------
LightRAG calls the extraction LLM synchronously, once per chunk (plus one
"gleaning" call).  Provider Batch APIs are half price but asynchronous, so they
cannot be plugged in as ``llm_model_func``.  Instead this module *primes
LightRAG's own LLM response cache*:

* ``lightrag.utils.use_llm_func_with_cache`` keys every extraction call by
  ``md5(user_prompt + "\\n" + system_prompt + "\\n" + json(history))`` under
  ``default:extract:<md5>`` in ``kv_store_llm_response_cache.json`` and, with
  ``enable_llm_cache_for_entity_extract`` (default on), returns the cached
  answer without calling the LLM.
* So: build every prompt LightRAG *would* send (``collect``), run them through
  the Batch API (``submit``), write the answers into that cache (``apply``),
  then run the normal indexing — every extraction call is a cache hit and the
  rest of the pipeline (merge, embeddings, graph, chunk metadata) is untouched.

Phases (each resumable and idempotent; state in ``<index>/batch_state.json``)
------------------------------------------------------------------------------
``collect``  chunk the documents with the configured chunker and run LightRAG's
             own ``extract_entities`` with a *recording* LLM that returns a stub.
             No network.  Round 1 = first-pass extraction (depends only on the
             chunk).  Round 2 = the gleaning pass; its prompt embeds the round-1
             answer, so it can only be collected after round 1 has completed.
``submit``   write the OpenAI batch JSONL, upload it, create the batch(es).
``status``   poll; download the output of finished batches.
``apply``    prime the cache and run the normal indexing.  Requests that failed
             (error line, missing, expired batch) are simply not primed, so
             LightRAG answers them with the normal synchronous LLM call; the
             counts are reported.

``entity_extract_max_gleaning`` is read from ``HARS_MEMORY_MAX_GLEANING``
exactly as the synchronous indexer does (LightRAG performs at most ONE
gleaning pass).  Keep it identical to the baseline index when comparing.

The only coupling to LightRAG internals is the documented adapter in
``_collect_requests``: ``rag.chunking_func`` + ``operate.extract_entities`` with
an in-memory cache stand-in.  If the cache-key derivation ever drifts, the
collector raises (the stand-in sees a key it did not derive itself) instead of
silently priming keys LightRAG would never look up.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Protocol

logger = logging.getLogger("memory.index_batch")

STATE_FILENAME: Final[str] = "batch_state.json"
BATCH_DIRNAME: Final[str] = "batch"
STATE_VERSION: Final[int] = 1
DEFAULT_ENDPOINT: Final[str] = "/v1/chat/completions"
# OpenAI Batch API hard limits are 50 000 requests / 200 MB per file; stay under.
MAX_REQUESTS_PER_BATCH: Final[int] = 50_000
MAX_BYTES_PER_BATCH: Final[int] = 150 * 1024 * 1024
# Prices are USD per 1M tokens AT BATCH RATES (the 50% discount already applied).
DEFAULT_INPUT_PRICE: Final[float] = 0.05
DEFAULT_OUTPUT_PRICE: Final[float] = 0.25
# Output tokens per request are unknown before the first call; deliberately
# pessimistic (a reasoning model also bills its hidden reasoning tokens).
DEFAULT_EST_OUTPUT_TOKENS: Final[int] = 1500
DEFAULT_MAX_COST_USD: Final[float] = 0.50
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "failed", "expired", "cancelled"}
)
# Answer returned by the recording LLM; a request whose history contains it was
# derived from an answer we do not have yet (gleaning of an unanswered chunk).
_STUB: Final[str] = "<|COMPLETE|>"


class BatchError(RuntimeError):
    """Raised for user-facing batch-phase failures (bad state, cost guard, ...)."""


class CostGuardError(BatchError):
    """The estimated cost exceeds ``--max-cost``."""


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class BatchBackend(Protocol):
    """Minimal Batch API surface; the real one wraps the OpenAI SDK, tests fake it."""

    def upload(self, path: Path) -> str: ...

    def create(self, file_id: str, metadata: dict[str, str]) -> dict[str, Any]: ...

    def get(self, batch_id: str) -> dict[str, Any]: ...

    def read(self, file_id: str) -> str: ...


class OpenAIBatchBackend:
    """Batch backend over the OpenAI SDK (works against a LiteLLM proxy too)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        completion_window: str = "24h",
    ) -> None:
        import openai

        self._client = openai.OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
        self._endpoint = endpoint
        self._window = completion_window

    def upload(self, path: Path) -> str:
        with path.open("rb") as fh:
            return self._client.files.create(file=fh, purpose="batch").id

    def create(self, file_id: str, metadata: dict[str, str]) -> dict[str, Any]:
        batch = self._client.batches.create(
            input_file_id=file_id,
            endpoint=self._endpoint,  # type: ignore[arg-type]
            completion_window=self._window,  # type: ignore[arg-type]
            metadata=metadata,
        )
        return _batch_dict(batch)

    def get(self, batch_id: str) -> dict[str, Any]:
        return _batch_dict(self._client.batches.retrieve(batch_id))

    def read(self, file_id: str) -> str:
        return self._client.files.content(file_id).text


def _batch_dict(batch: Any) -> dict[str, Any]:
    counts = getattr(batch, "request_counts", None)
    errors = getattr(batch, "errors", None)
    error_list = getattr(errors, "data", None) or []
    return {
        "id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "counts": {
            "total": getattr(counts, "total", None),
            "completed": getattr(counts, "completed", None),
            "failed": getattr(counts, "failed", None),
        },
        "errors": [getattr(e, "message", str(e)) for e in error_list][:5],
    }


def make_backend_from_env() -> OpenAIBatchBackend:
    """Backend for the extractor endpoint (HARS_MEMORY_EXTRACTOR_BASE_URL / LLM_API_KEY)."""
    base_url = os.environ.get("HARS_MEMORY_EXTRACTOR_BASE_URL", "")
    api_key = os.environ.get("HARS_MEMORY_LLM_API_KEY", "")
    if not base_url or not api_key:
        raise BatchError(
            "HARS_MEMORY_EXTRACTOR_BASE_URL and HARS_MEMORY_LLM_API_KEY must be set to submit batches"
        )
    return OpenAIBatchBackend(base_url, api_key)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


@dataclass
class BatchRequest:
    """One extraction call LightRAG would make: key == LightRAG's cache key."""

    key: str
    chunk_id: str
    round: int
    prompt: str  # the joined string LightRAG hashes (stored as ``original_prompt``)
    messages: list[dict[str, str]]
    est_input_tokens: int = 0


def request_key(
    user_prompt: str, system_prompt: str | None, history: list[dict[str, str]] | None
) -> tuple[str, str, list[dict[str, str]]]:
    """Return (cache_key, joined_prompt, messages) exactly as LightRAG derives them.

    Mirrors ``lightrag.utils.use_llm_func_with_cache`` (cache_type ``extract``,
    mode ``default``); messages mirror ``server/lightrag_init.make_llm_func``.
    """
    from lightrag.utils import compute_args_hash, generate_cache_key

    parts = [p for p in (user_prompt, system_prompt) if p]
    if history:
        parts.append(json.dumps(history, ensure_ascii=False))
    joined = "\n".join(parts)
    key = generate_cache_key("default", "extract", compute_args_hash(joined))
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_prompt})
    return key, joined, messages


class _RecordingCache:
    """In-memory stand-in for ``llm_response_cache`` during ``collect``.

    ``get_by_id`` serves known answers; every miss makes LightRAG call the
    recording LLM and then ``upsert`` here, which is where the chunk id lands.
    """

    def __init__(self, answers: dict[str, str]) -> None:
        self.global_config = {"enable_llm_cache_for_entity_extract": True, "enable_llm_cache": True}
        self._answers = answers
        self.pending: dict[str, BatchRequest] = {}
        self._seen: dict[str, tuple[str, list[dict[str, str]], bool]] = {}

    def record_call(self, key: str, joined: str, messages: list[dict[str, str]], derived: bool) -> None:
        self._seen[key] = (joined, messages, derived)

    async def get_by_id(self, key: str) -> dict[str, Any] | None:
        if key in self._answers:
            return {"return": self._answers[key], "create_time": 0}
        return None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        for key, entry in data.items():
            seen = self._seen.get(key)
            if seen is None:
                raise BatchError(
                    f"cache key {key} was not derived by the recorder: LightRAG's cache-key "
                    "derivation changed; batch priming would never hit. Refusing to continue."
                )
            joined, messages, derived = seen
            if derived:
                continue  # depends on a stub answer; collectable only after its upstream round
            self.pending[key] = BatchRequest(
                key=key,
                chunk_id=entry.get("chunk_id") or "",
                round=2 if len(messages) > 2 else 1,
                prompt=joined,
                messages=messages,
            )


async def _collect_async(
    docs: list[Any], answers: dict[str, str], scratch: Path
) -> tuple[list[BatchRequest], dict[str, Any]]:
    """Chunk *docs* and record the extraction calls LightRAG would issue."""
    from lightrag.operate import extract_entities
    from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

    from hars_memory.server.index import _unique_file_key
    from hars_memory.server.lightrag_init import create_lightrag

    rag = create_lightrag(working_dir=str(scratch))
    cache = _RecordingCache(answers)

    async def recorder(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **_: object,
    ) -> str:
        key, joined, messages = request_key(prompt, system_prompt, history_messages)
        derived = any(m.get("content") == _STUB for m in (history_messages or []))
        cache.record_call(key, joined, messages, derived)
        return _STUB

    chunks: dict[str, dict[str, Any]] = {}
    for doc in docs:
        content = sanitize_text_for_encoding(doc.content)
        result = rag.chunking_func(  # type: ignore[attr-defined]
            rag.tokenizer,  # type: ignore[attr-defined]
            content,
            None,
            False,
            rag.chunk_overlap_token_size,  # type: ignore[attr-defined]
            rag.chunk_token_size,  # type: ignore[attr-defined]
        )
        if inspect.isawaitable(result):
            result = await result
        for dp in result:
            chunks[compute_mdhash_id(dp["content"], prefix="chunk-")] = {
                **dp,
                "full_doc_id": doc.doc_id,
                "file_path": _unique_file_key(doc),
            }
    global_config = asdict(rag)  # type: ignore[call-overload]
    global_config["llm_model_func"] = recorder
    await extract_entities(chunks, global_config=global_config, llm_response_cache=cache)

    tokenizer = rag.tokenizer  # type: ignore[attr-defined]
    requests = list(cache.pending.values())
    for req in requests:
        req.est_input_tokens = sum(len(tokenizer.encode(m["content"])) for m in req.messages)
    meta = {
        "chunks": len(chunks),
        "gleaning": int(global_config.get("entity_extract_max_gleaning", 0)),
        "model": os.environ.get("HARS_MEMORY_EXTRACTOR_MODEL", ""),
    }
    return requests, meta


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class Part:
    part: int
    requests: int
    input_file: str
    est_input_tokens: int = 0
    file_id: str | None = None
    batch_id: str | None = None
    status: str = "pending"  # pending -> submitted statuses from the backend
    output_file_id: str | None = None
    error_file_id: str | None = None
    counts: dict[str, Any] = field(default_factory=dict)
    submitted_at: float | None = None
    completed_at: float | None = None
    downloaded: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class RoundState:
    round: int
    requests: int
    est_input_tokens: int
    est_cost_usd: float
    requests_file: str
    parts: list[Part] = field(default_factory=list)

    def terminal(self) -> bool:
        return bool(self.parts) and all(p.status in TERMINAL_STATUSES and p.downloaded for p in self.parts)

    def submitted(self) -> bool:
        return all(p.batch_id for p in self.parts)


@dataclass
class BatchState:
    version: int = STATE_VERSION
    model: str = ""
    endpoint: str = DEFAULT_ENDPOINT
    chunker: str = "token"
    gleaning: int = 0
    doc_ids: list[str] = field(default_factory=list)
    prices: dict[str, float] = field(default_factory=dict)
    rounds: dict[str, RoundState] = field(default_factory=dict)
    applied: dict[str, Any] = field(default_factory=dict)

    # -- persistence --
    @classmethod
    def load(cls, index_dir: Path) -> BatchState:
        path = index_dir / STATE_FILENAME
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = cls(**{k: v for k, v in raw.items() if k != "rounds"})
        for name, rnd in raw.get("rounds", {}).items():
            parts = [Part(**p) for p in rnd.pop("parts", [])]
            state.rounds[name] = RoundState(**rnd, parts=parts)
        return state

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        tmp = index_dir / (STATE_FILENAME + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, index_dir / STATE_FILENAME)


def batch_dir(index_dir: Path) -> Path:
    return index_dir / BATCH_DIRNAME


def estimate_cost(
    requests: int,
    input_tokens: int,
    *,
    input_price: float,
    output_price: float,
    est_output_tokens: int,
) -> float:
    return (input_tokens * input_price + requests * est_output_tokens * output_price) / 1_000_000


# ---------------------------------------------------------------------------
# Phase: collect
# ---------------------------------------------------------------------------


def _output_content(body: dict[str, Any]) -> str:
    """Answer text from a chat-completion body, normalised like ``make_llm_func``."""
    import re

    from lightrag.utils import remove_think_tags

    message = ((body.get("choices") or [{}])[0]).get("message") or {}
    content = str(message.get("content") or "")
    reasoning = message.get("reasoning_content")
    if not content.strip() and reasoning:
        content = str(reasoning)
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return remove_think_tags(content)


def load_answers(index_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    """Answers from every downloaded ``*.output.jsonl`` plus {ok, error_lines} counts."""
    answers: dict[str, str] = {}
    counts = {"ok": 0, "error_lines": 0}
    for path in sorted(batch_dir(index_dir).glob("round-*.output.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            response = row.get("response") or {}
            text = ""
            if row.get("error") is None and int(response.get("status_code", 0)) == 200:
                text = _output_content(response.get("body") or {})
            if text.strip():
                answers[row["custom_id"]] = text
                counts["ok"] += 1
            else:
                counts["error_lines"] += 1
    return answers, counts


def _processed_doc_ids(index_dir: Path) -> set[str]:
    path = index_dir / "kv_store_doc_status.json"
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {k for k, v in data.items() if str(v.get("status", "")).lower().endswith("processed")}


def next_round(state: BatchState) -> int | None:
    """Next round to collect, or None when nothing is left to collect."""
    if "1" not in state.rounds:
        return 1
    if state.gleaning > 0 and "2" not in state.rounds and state.rounds["1"].terminal():
        return 2
    return None


def collect(
    index_dir: Path,
    docs: list[Any],
    *,
    input_price: float = DEFAULT_INPUT_PRICE,
    output_price: float = DEFAULT_OUTPUT_PRICE,
    est_output_tokens: int = DEFAULT_EST_OUTPUT_TOKENS,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    dry_run: bool = False,
) -> RoundState | None:
    """Collect the next round's requests; persist them unless *dry_run*.

    Raises CostGuardError (nothing written) when the estimate exceeds *max_cost_usd*.
    Returns None when there is nothing to collect.
    """
    state = BatchState.load(index_dir)
    doc_ids = sorted(d.doc_id for d in docs)
    if state.doc_ids and state.doc_ids != doc_ids:
        raise BatchError(
            "the document set differs from the one this index's batch_state.json was collected for; "
            "use a fresh --index-dir (or delete batch_state.json and batch/) to start over"
        )
    rnd = next_round(state)
    if rnd is None:
        return None
    todo = [d for d in docs if d.doc_id not in _processed_doc_ids(index_dir)]
    answers, _ = load_answers(index_dir)
    scratch = batch_dir(index_dir) / "_collect"
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        requests, meta = asyncio.run(_collect_async(todo, answers, scratch))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    requests = [r for r in requests if r.round == rnd and r.key not in answers]
    requests.sort(key=lambda r: r.key)

    in_tokens = sum(r.est_input_tokens for r in requests)
    cost = estimate_cost(
        len(requests), in_tokens,
        input_price=input_price, output_price=output_price, est_output_tokens=est_output_tokens,
    )
    state_cost = sum(r.est_cost_usd for r in state.rounds.values()) + cost
    logger.info(
        "collect round %d: %d request(s), ~%d input tokens, estimated cost $%.4f (cumulative $%.4f, cap $%.2f)",
        rnd, len(requests), in_tokens, cost, state_cost, max_cost_usd,
    )
    if state_cost > max_cost_usd:
        raise CostGuardError(
            f"estimated cost ${state_cost:.4f} exceeds --max-cost ${max_cost_usd:.2f}; nothing submitted"
        )
    result = RoundState(
        round=rnd, requests=len(requests), est_input_tokens=in_tokens, est_cost_usd=round(cost, 6),
        requests_file=f"round-{rnd}.requests.jsonl",
    )
    if dry_run:
        return result

    bdir = batch_dir(index_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    with (bdir / result.requests_file).open("w", encoding="utf-8") as fh:
        for r in requests:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    state.chunker = os.environ.get("HARS_MEMORY_CHUNKER", "token").strip().lower() or "token"
    state.gleaning = int(meta["gleaning"])
    state.model = meta["model"] or state.model
    state.doc_ids = doc_ids
    state.prices = {"input": input_price, "output": output_price, "est_output_tokens": est_output_tokens}
    state.rounds[str(rnd)] = result
    state.save(index_dir)
    return result


# ---------------------------------------------------------------------------
# Phase: submit / status
# ---------------------------------------------------------------------------


def provider_model(model: str) -> str:
    """Model id as the provider's Batch API knows it.

    A LiteLLM proxy routes ``openai/gpt-6-luna`` for chat calls, but its batch
    endpoints hand the uploaded file to the provider as is, which rejects the
    ``openai/`` routing prefix ("model is not supported by the Batch API").
    """
    return model.split("/", 1)[1] if model.startswith("openai/") else model


def _body(model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int,
          max_tokens_param: str) -> dict[str, Any]:
    return {"model": provider_model(model), "messages": messages, "temperature": temperature, max_tokens_param: max_tokens}


def submit(
    index_dir: Path,
    backend: BatchBackend,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_tokens_param: str = "max_completion_tokens",
    max_requests_per_batch: int = MAX_REQUESTS_PER_BATCH,
    max_bytes_per_batch: int = MAX_BYTES_PER_BATCH,
    max_tokens_per_batch: int = 0,
    max_inflight_tokens: int = 0,
) -> list[Part]:
    """Upload + create batches for every collected-but-unsubmitted round. Idempotent.

    *max_tokens_per_batch* splits a round into parts of at most that many
    (estimated) input tokens; *max_inflight_tokens* submits parts only while the
    estimated input tokens of submitted-but-unfinished parts stay under the cap
    (providers enforce an "enqueued tokens" quota per model; 0 disables both).
    Call again (``run`` does, after every poll) to submit the parts that waited.
    """
    state = BatchState.load(index_dir)
    if not state.rounds:
        raise BatchError("nothing collected yet: run `index-batch collect` first")
    model = model or state.model or os.environ.get("HARS_MEMORY_EXTRACTOR_MODEL", "")
    if not model:
        raise BatchError("no model: set HARS_MEMORY_EXTRACTOR_MODEL")
    temperature = (
        temperature if temperature is not None
        else float(os.environ.get("HARS_MEMORY_EXTRACTOR_TEMPERATURE", "0.1"))
    )
    max_tokens = max_tokens or int(os.environ.get("HARS_MEMORY_EXTRACTOR_MAX_TOKENS", "8192"))
    bdir = batch_dir(index_dir)
    created: list[Part] = []
    for name in sorted(state.rounds):
        rnd = state.rounds[name]
        if not rnd.parts and rnd.requests:
            _split_parts(bdir, rnd, model, temperature, max_tokens, max_tokens_param,
                         max_requests_per_batch, max_bytes_per_batch, max_tokens_per_batch)
            state.save(index_dir)
        for part in rnd.parts:
            if part.batch_id:
                continue
            inflight = sum(
                p.est_input_tokens for r in state.rounds.values() for p in r.parts
                if p.batch_id and p.status not in TERMINAL_STATUSES
            )
            if max_inflight_tokens and inflight and inflight + part.est_input_tokens > max_inflight_tokens:
                continue  # wait for in-flight parts to finish; a later call submits it
            if not part.file_id:
                part.file_id = backend.upload(bdir / part.input_file)
                state.save(index_dir)
            info = backend.create(part.file_id, {"hars_index": index_dir.name, "round": name, "part": str(part.part)})
            part.batch_id = info["id"]
            part.status = info["status"]
            part.submitted_at = time.time()
            state.save(index_dir)
            created.append(part)
            logger.info("submitted round %s part %d: batch %s (%d requests)", name, part.part, part.batch_id, part.requests)
    return created


def _split_parts(
    bdir: Path, rnd: RoundState, model: str, temperature: float, max_tokens: int,
    max_tokens_param: str, max_requests: int, max_bytes: int, max_tokens_in: int = 0,
) -> None:
    """Write the upload JSONL(s) for *rnd*, split to stay under the API limits."""
    rows = [json.loads(line) for line in (bdir / rnd.requests_file).read_text(encoding="utf-8").splitlines() if line]
    group: list[str] = []
    size = tokens = 0

    def flush() -> None:
        nonlocal group, size, tokens
        if not group:
            return
        idx = len(rnd.parts)
        name = f"round-{rnd.round}.part-{idx}.input.jsonl"
        (bdir / name).write_text("".join(group), encoding="utf-8")
        rnd.parts.append(Part(part=idx, requests=len(group), input_file=name, est_input_tokens=tokens))
        group, size, tokens = [], 0, 0

    for row in rows:
        line = json.dumps(
            {
                "custom_id": row["key"],
                "method": "POST",
                "url": DEFAULT_ENDPOINT,
                "body": _body(model, row["messages"], temperature, max_tokens, max_tokens_param),
            },
            ensure_ascii=False,
        ) + "\n"
        row_tokens = int(row.get("est_input_tokens", 0))
        if group and (
            len(group) >= max_requests
            or size + len(line.encode()) > max_bytes
            or (max_tokens_in and tokens + row_tokens > max_tokens_in)
        ):
            flush()
        group.append(line)
        size += len(line.encode())
        tokens += row_tokens
    flush()


def refresh(index_dir: Path, backend: BatchBackend) -> BatchState:
    """Poll every submitted, non-finished part; download output of finished ones."""
    state = BatchState.load(index_dir)
    bdir = batch_dir(index_dir)
    for name, rnd in state.rounds.items():
        for part in rnd.parts:
            if not part.batch_id or part.downloaded:
                continue
            if part.status not in TERMINAL_STATUSES:
                info = backend.get(part.batch_id)
                part.status = info["status"]
                part.output_file_id = info.get("output_file_id")
                part.error_file_id = info.get("error_file_id")
                part.counts = info.get("counts") or {}
                part.errors = info.get("errors") or []
            if part.status in TERMINAL_STATUSES:
                part.completed_at = part.completed_at or time.time()
                stem = f"round-{name}.part-{part.part}"
                if part.output_file_id:
                    text = backend.read(part.output_file_id)
                    (bdir / f"{stem}.output.jsonl").write_text(text, encoding="utf-8")
                    part.usage = _sum_usage(text)
                if part.error_file_id:
                    (bdir / f"{stem}.errors.jsonl").write_text(backend.read(part.error_file_id), encoding="utf-8")
                part.downloaded = True
        state.save(index_dir)
    return state


def _sum_usage(output_text: str) -> dict[str, int]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
    for line in output_text.splitlines():
        if not line.strip():
            continue
        body = ((json.loads(line).get("response") or {}).get("body") or {})
        u = body.get("usage") or {}
        usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
        usage["reasoning_tokens"] += int(
            ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
        )
        usage["cached_tokens"] += int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    return usage


def actual_cost(state: BatchState) -> dict[str, float]:
    """Measured token usage across all downloaded batches and its cost at batch prices."""
    prompt = completion = 0
    for rnd in state.rounds.values():
        for part in rnd.parts:
            prompt += part.usage.get("prompt_tokens", 0)
            completion += part.usage.get("completion_tokens", 0)
    ip = state.prices.get("input", DEFAULT_INPUT_PRICE)
    op = state.prices.get("output", DEFAULT_OUTPUT_PRICE)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "batch_cost_usd": (prompt * ip + completion * op) / 1_000_000,
        "sync_equivalent_usd": 2 * (prompt * ip + completion * op) / 1_000_000,
    }


def format_status(state: BatchState) -> str:
    lines = [
        f"model={state.model or '?'} chunker={state.chunker} gleaning={state.gleaning} docs={len(state.doc_ids)}"
    ]
    for name in sorted(state.rounds):
        rnd = state.rounds[name]
        lines.append(
            f"round {name}: {rnd.requests} request(s), est ${rnd.est_cost_usd:.4f}, "
            f"{'done' if rnd.terminal() else 'in progress'}"
        )
        for p in rnd.parts:
            lines.append(
                f"  part {p.part}: batch={p.batch_id or '-'} status={p.status} "
                f"counts={p.counts or '-'} usage={p.usage or '-'}"
                + (f" errors={p.errors}" if p.errors else "")
            )
    if state.applied:
        lines.append(f"applied: {json.dumps(state.applied)}")
    return "\n".join(lines)


def wait(
    index_dir: Path,
    backend: BatchBackend,
    *,
    poll_seconds: float,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    after_poll: Callable[[], object] | None = None,
) -> BatchState:
    """Poll until every part is terminal or the timeout passes.

    *after_poll* runs after each refresh (``run`` uses it to submit parts that
    waited for in-flight capacity).
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = refresh(index_dir, backend)
        if after_poll is not None:
            after_poll()
            state = BatchState.load(index_dir)
        pending = [p for r in state.rounds.values() for p in r.parts if not p.downloaded]
        if not pending:
            return state
        if time.monotonic() >= deadline:
            return state
        logger.info("waiting: %s", ", ".join(f"{p.batch_id}={p.status}" for p in pending))
        sleep(poll_seconds)


# ---------------------------------------------------------------------------
# Phase: apply
# ---------------------------------------------------------------------------


async def _prime_and_index(index_dir: Path, docs: list[Any], answers: dict[str, str]) -> dict[str, Any]:
    from lightrag.utils import statistic_data

    from hars_memory.server.index import _insert_all_batches
    from hars_memory.server.lightrag_init import create_lightrag

    requests: dict[str, BatchRequest] = {}
    for path in sorted(batch_dir(index_dir).glob("round-*.requests.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                requests[row["key"]] = BatchRequest(**row)

    rag = create_lightrag(working_dir=str(index_dir))
    await rag.initialize_storages()  # type: ignore[attr-defined]
    entries = {
        key: {
            "return": text,
            "cache_type": "extract",
            "chunk_id": requests[key].chunk_id or None,
            "original_prompt": requests[key].prompt,
            "queryparam": None,
        }
        for key, text in answers.items()
        if key in requests
    }
    if entries:
        await rag.llm_response_cache.upsert(entries)  # type: ignore[attr-defined]
        await rag.llm_response_cache.index_done_callback()  # type: ignore[attr-defined]
    before = dict(statistic_data)
    batch_size = int(os.environ.get("HARS_MEMORY_INSERT_BATCH_SIZE", "10"))
    try:
        await _insert_all_batches(rag, docs, batch_size)
    finally:
        await rag.finalize_storages()  # type: ignore[attr-defined]
        # LightRAG keeps storage namespaces in process-global state; drop it so a
        # later phase / another index in the same process starts clean.
        from lightrag.kg.shared_storage import finalize_share_data

        finalize_share_data()
    hits = statistic_data.get("llm_cache", 0) - before.get("llm_cache", 0)
    calls = statistic_data.get("llm_call", 0) - before.get("llm_call", 0)
    from hars_memory.ingest.migrate import backfill_after_ingest

    backfill_after_ingest(getattr(rag, "working_dir", None), docs)
    return {"primed": len(entries), "cache_hits": hits, "sync_llm_calls": calls}


def apply(index_dir: Path, docs: list[Any]) -> dict[str, Any]:
    """Prime LightRAG's cache from downloaded answers and run the normal indexing."""
    state = BatchState.load(index_dir)
    if not state.rounds:
        raise BatchError("nothing to apply: run collect/submit first")
    if next_round(state) is not None or not all(r.terminal() for r in state.rounds.values()):
        raise BatchError(
            "batches are not all finished (or the gleaning round is not collected yet); "
            "run `index-batch status` / `collect` / `submit` first"
        )
    answers, counts = load_answers(index_dir)
    total_requests = sum(r.requests for r in state.rounds.values())
    stats = asyncio.run(_prime_and_index(index_dir, docs, answers))
    stats.update(
        total_requests=total_requests,
        answered=counts["ok"],
        error_lines=counts["error_lines"],
        expected_sync_fallback=total_requests - counts["ok"],
    )
    state.applied = stats
    state.save(index_dir)
    return stats


# ---------------------------------------------------------------------------
# Comparison of two indexes (sync vs batch smoke)
# ---------------------------------------------------------------------------


def index_stats(index_dir: Path) -> dict[str, Any]:
    """Read-only counts from an index directory (never opens it with LightRAG)."""
    import networkx as nx

    def load(name: str) -> dict[str, Any]:
        path = index_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    chunks = load("kv_store_text_chunks.json")
    cache = load("kv_store_llm_response_cache.json")
    status = load("kv_store_doc_status.json")
    graph_path = index_dir / "graph_chunk_entity_relation.graphml"
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    if graph_path.is_file():
        graph = nx.read_graphml(graph_path)
        nodes = {str(n) for n in graph.nodes}
        edges = {tuple(sorted((str(a), str(b)))) for a, b in graph.edges}  # type: ignore[misc]
    by_type: dict[str, int] = {}
    for entry in cache.values():
        kind = str(entry.get("cache_type", "?"))
        by_type[kind] = by_type.get(kind, 0) + 1
    return {
        "docs": len(status),
        "docs_processed": sum(1 for v in status.values() if str(v.get("status", "")).lower().endswith("processed")),
        "chunks": len(chunks),
        "chunks_with_heading_path": sum(1 for c in chunks.values() if c.get("heading_path")),
        "chunks_with_lines": sum(1 for c in chunks.values() if c.get("start_line")),
        "entities": len(nodes),
        "relations": len(edges),
        "llm_cache_by_type": by_type,
        "_nodes": nodes,
        "_edges": edges,
    }


def format_comparison(a_dir: str | Path | None, b_dir: Path) -> str:
    if not a_dir:
        raise BatchError("compare needs --other DIR (the index to compare --index-dir against)")
    a, b = index_stats(Path(a_dir).expanduser()), index_stats(b_dir)

    def jaccard(x: set, y: set) -> float:
        return len(x & y) / len(x | y) if x | y else 1.0

    lines = [f"{'':28}{'A (--other)':>14}{'B (--index-dir)':>18}"]
    for key in ("docs", "docs_processed", "chunks", "chunks_with_heading_path", "chunks_with_lines", "entities", "relations"):
        lines.append(f"{key:28}{a[key]:>14}{b[key]:>18}")
    lines.append(f"{'llm_cache_by_type':28}{json.dumps(a['llm_cache_by_type']):>14}{json.dumps(b['llm_cache_by_type']):>18}")
    lines.append(f"entity name overlap (Jaccard): {jaccard(a['_nodes'], b['_nodes']):.3f}")
    lines.append(f"relation overlap (Jaccard):    {jaccard(a['_edges'], b['_edges']):.3f}")
    return "\n".join(lines)
