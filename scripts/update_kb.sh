#!/usr/bin/env bash
# Incremental knowledge-base update: merge staged agent notes (+ optional extra
# paths) into the live long-term memory index using the local gemma-4-12b extractor.
#
# Usage:
#   tools/memory/scripts/update_kb.sh [extra paths...]
#     no args  -> index staging notes only
#     paths    -> also index the given dirs/files (e.g. .reports .session)
#
# GPU is needed ONLY for the duration of this run (extractor LLM). Queries stay
# LLM-free. index.py refuses to run if a training experiment is active (gpu_guard).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HF_HOME="${HF_HOME:-/mnt/datasets/models/.hf_home}"
STAGING="${HARS_MEMORY_STAGING_DIR:-/mnt/datasets/graphrag/staging}"
INDEX="${HARS_MEMORY_INDEX_DIR:-/home/user/.local/share/hars-graphrag/index_gemma_v4}"
PORT=8083
BIN="$HOME/llama.cpp/build-rocm/bin/llama-server"
GGUF=$(find -L "$HF_HOME/hub/models--unsloth--gemma-4-12b-it-GGUF" -name 'gemma-4-12b-it-Q4_K_M.gguf' | head -1)
LOG=/tmp/memory_update_kb.log

note_count=$(find "$STAGING" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)
if [[ $note_count -eq 0 && $# -eq 0 ]]; then
    echo "nothing to update: no staged notes and no extra paths"; exit 0
fi
echo "staged notes: $note_count | extra paths: $*"

cleanup() { pkill -9 -f "llama-server.*$PORT" 2>/dev/null || true; }
trap cleanup EXIT

# --- extractor concurrency: DO NOT raise beyond 6 -------------------------
# Raising --parallel needs a measured (not calculated) peak VRAM number from
# a live indexing run at the CTX_SIZE computed below (rocm-smi --showmeminfo
# vram during the run). The current KV-cache footprint (~1.5 GiB @ old
# ctx 61440/6) is a CALCULATION from GGUF metadata, not a measurement, and
# headroom shrinks as both --parallel and CTX_SIZE grow. Leave at 6.
PARALLEL=6

# --- per-slot context sizing (fixes latent ctx-size/--parallel bug) -------
# llama.cpp DIVIDES --ctx-size across --parallel slots — it is NOT shared.
# This exact invariant is already documented (and was violated here) in:
#   tools/memory/server/lightrag_init.py:270-272 ("max_total_tokens ...
#     MUST be <= the llama-server per-slot context (-c / --parallel)")
#   tools/memory/config/.env.example:14-16 (same invariant)
# We compute CTX_SIZE from the same numbers LightRAG is actually configured
# with below, so --parallel can never again drift out of agreement with
# --ctx-size without the arithmetic visibly changing too.
#
# LightRAG's max_total_tokens (HARS_MEMORY_LLM_MAX_TOKEN_SIZE) is measured with
# this project's _ByteTokenizer (tools/memory/server/lightrag_init.py:30-44),
# which counts UTF-8 BYTES, not real gemma/llama.cpp tokens — despite the
# "TOKEN" in the env var name. We must convert bytes -> real tokens before
# comparing against the server's per-slot token budget.
#
# Byte->token calibration: LightRAG's entity-extraction system prompt is
# ~15,900 chars and was measured (this session) at ~4,000 real tokens on
# this exact build/model -> ~4 bytes/token for plain English prose. Our
# indexed corpus (agent notes, .session/.reports markdown with tables, code
# blocks, YAML, numbers) tokenizes MORE densely than prose, so we use a
# conservative 3 bytes/token floor (not the observed 4) to avoid
# under-provisioning context for denser real content.
HARS_MEMORY_LLM_MAX_TOKEN_SIZE=24576   # bytes (see rationale above); single source of truth, reused below
HARS_MEMORY_EXTRACTOR_MAX_TOKENS=4096  # max output tokens/request; single source of truth, reused below
SYSTEM_PROMPT_TOKENS=4000           # measured constant prefix (see calibration above)
BYTES_PER_TOKEN_CONSERVATIVE=3
CHAT_TEMPLATE_OVERHEAD_TOKENS=512   # jinja role wrapping / special tokens safety margin

INPUT_TOKENS_EST=$(( (HARS_MEMORY_LLM_MAX_TOKEN_SIZE + BYTES_PER_TOKEN_CONSERVATIVE - 1) / BYTES_PER_TOKEN_CONSERVATIVE ))
REQUIRED_PER_SLOT=$(( SYSTEM_PROMPT_TOKENS + INPUT_TOKENS_EST + HARS_MEMORY_EXTRACTOR_MAX_TOKENS + CHAT_TEMPLATE_OVERHEAD_TOKENS ))
# Round up to a 1024-token boundary for a clean, auditable number.
REQUIRED_PER_SLOT=$(( ((REQUIRED_PER_SLOT + 1023) / 1024) * 1024 ))
CTX_SIZE=$(( REQUIRED_PER_SLOT * PARALLEL ))

echo "ctx-size sizing: system_prompt=${SYSTEM_PROMPT_TOKENS}t + input<=${INPUT_TOKENS_EST}t (${HARS_MEMORY_LLM_MAX_TOKEN_SIZE}B @ ${BYTES_PER_TOKEN_CONSERVATIVE}B/tok) + output=${HARS_MEMORY_EXTRACTOR_MAX_TOKENS}t + overhead=${CHAT_TEMPLATE_OVERHEAD_TOKENS}t -> required/slot=${REQUIRED_PER_SLOT}t x parallel=${PARALLEL} -> ctx-size=${CTX_SIZE}"

echo "starting extractor (gemma-4-12b, $PARALLEL slots, ctx-size=$CTX_SIZE)..."
# Prompt-prefix caching for the ~4,000-token constant LightRAG extraction
# system prompt (byte-identical on every chunk, ~85% of every request):
#
# --cache-prompt (longest-common-PREFIX reuse against a slot's own last
#   prompt) is ALREADY ON BY DEFAULT in this build ("(default: enabled)" per
#   --help) and needs no flag from us. This is the mechanism that actually
#   serves our workload: constant system prompt at position 0 + unique
#   variable chunk content after it is EXACTLY a leading-prefix match.
#   Verified live (smoke test, see .session notes): two identical requests
#   to the extractor showed usage.prompt_tokens_details.cached_tokens > 0 on
#   the second request — reuse is confirmed working with zero extra flags.
#
# --kv-unified: a single shared KV buffer across all sequences instead of a
#   fixed per-slot split. --help: "(default: enabled if number of slots is
#   auto)" — we pass --parallel 6 explicitly (not auto), so without this flag
#   kv-unified would default OFF. Enabling it lets slots dynamically share
#   the pool rather than each being hard-walled at ctx/parallel; it does NOT
#   remove the need for CTX_SIZE to cover worst-case uniform load across all
#   slots (computed above) — it only improves utilization when load is uneven.
#
# --cache-reuse is DELIBERATELY NOT SET: confirmed via smoke test that this
#   build logs "cache_reuse is not supported by this context, it will be
#   disabled" whenever --kv-unified is active — the two flags are mutually
#   exclusive in this build, contrary to the assumption that both could be
#   combined. Since --cache-prompt (above) already covers our exact
#   prefix-then-suffix request shape, we keep --kv-unified for its memory-
#   sharing benefit and drop --cache-reuse rather than ship a flag the
#   server silently ignores.
HSA_XNACK=0 HSA_OVERRIDE_GFX_VERSION=12.0.1 setsid nohup "$BIN" \
    --model "$GGUF" --alias gemma-4-12b --host 127.0.0.1 --port $PORT \
    --n-gpu-layers 99 --ctx-size "$CTX_SIZE" --parallel "$PARALLEL" -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --jinja --reasoning-budget 0 \
    --kv-unified \
    > /tmp/memory_update_llm.log 2>&1 < /dev/null &
for _ in $(seq 60); do
    sleep 3
    curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null && break
done
curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null || {
    echo "extractor failed to start — see /tmp/memory_update_llm.log"; exit 1; }

PATHS=("$STAGING")
[[ $# -gt 0 ]] && PATHS+=("$@")

echo "indexing into $INDEX ..."
cd "$ROOT"
HARS_MEMORY_VECTOR_STORAGE=NanoVectorDBStorage \
HARS_MEMORY_INDEX_DIR="$INDEX" \
HARS_MEMORY_EXTRACTOR_BASE_URL="http://localhost:$PORT/v1" \
HARS_MEMORY_EXTRACTOR_MODEL=gemma-4-12b \
HARS_MEMORY_EXTRACTOR_MAX_TOKENS="$HARS_MEMORY_EXTRACTOR_MAX_TOKENS" \
HARS_MEMORY_LLM_MAX_TOKEN_SIZE="$HARS_MEMORY_LLM_MAX_TOKEN_SIZE" \
HARS_MEMORY_CHUNK_TOKEN_SIZE=2048 \
HARS_MEMORY_CHUNK_OVERLAP_TOKENS=256 \
HARS_MEMORY_MAX_PARALLEL_INSERT="$PARALLEL" \
HARS_MEMORY_INSERT_BATCH_SIZE=99999 \
HARS_MEMORY_LLM_MAX_ASYNC="$PARALLEL" \
HARS_MEMORY_MAX_GLEANING=0 \
HARS_MEMORY_LLM_TIMEOUT_SECONDS=600 \
LLM_TIMEOUT=600 \
HARS_MEMORY_EMBED_MODEL=unsloth/embeddinggemma-300m \
HARS_MEMORY_EMBED_DEVICE=cpu \
HARS_MEMORY_EMBED_LOCAL_FILES_ONLY=1 \
HF_HOME="$HF_HOME" \
PYTHONPATH="$ROOT" \
uv run --project tools/memory python tools/memory/server/index.py \
    --paths "${PATHS[@]}" > "$LOG" 2>&1

echo "indexing done; archiving ingested notes..."
mkdir -p "$STAGING/ingested"
find "$STAGING" -maxdepth 1 -name '*.md' -exec mv {} "$STAGING/ingested/" \;
echo "KB update complete. Log: $LOG"
