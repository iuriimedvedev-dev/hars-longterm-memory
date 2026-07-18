#!/usr/bin/env bash
# Incremental knowledge-base update: merge staged agent notes (+ optional extra
# paths) into the live GraphRAG index using the local gemma-4-12b extractor.
#
# Usage:
#   tools/graphrag/scripts/update_kb.sh [extra paths...]
#     no args  -> index staging notes only
#     paths    -> also index the given dirs/files (e.g. .reports .session)
#
# GPU is needed ONLY for the duration of this run (extractor LLM). Queries stay
# LLM-free. index.py refuses to run if a training experiment is active (gpu_guard).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HF_HOME="${HF_HOME:-/mnt/datasets/models/.hf_home}"
STAGING="${GRAPHRAG_STAGING_DIR:-/mnt/datasets/graphrag/staging}"
INDEX="${GRAPHRAG_WORKING_DIR:-/mnt/datasets/graphrag/index_gemma_v4}"
PORT=8083
BIN="$HOME/llama.cpp/build-rocm/bin/llama-server"
GGUF=$(find -L "$HF_HOME/hub/models--unsloth--gemma-4-12b-it-GGUF" -name 'gemma-4-12b-it-Q4_K_M.gguf' | head -1)
LOG=/tmp/graphrag_update_kb.log

note_count=$(find "$STAGING" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)
if [[ $note_count -eq 0 && $# -eq 0 ]]; then
    echo "nothing to update: no staged notes and no extra paths"; exit 0
fi
echo "staged notes: $note_count | extra paths: $*"

cleanup() { pkill -9 -f "llama-server.*$PORT" 2>/dev/null || true; }
trap cleanup EXIT

echo "starting extractor (gemma-4-12b, 6 slots)..."
HSA_XNACK=0 HSA_OVERRIDE_GFX_VERSION=12.0.1 setsid nohup "$BIN" \
    --model "$GGUF" --alias gemma-4-12b --host 127.0.0.1 --port $PORT \
    --n-gpu-layers 99 --ctx-size 61440 --parallel 6 -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --jinja --reasoning-budget 0 \
    > /tmp/graphrag_update_llm.log 2>&1 < /dev/null &
for _ in $(seq 60); do
    sleep 3
    curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null && break
done
curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null || {
    echo "extractor failed to start — see /tmp/graphrag_update_llm.log"; exit 1; }

PATHS=("$STAGING")
[[ $# -gt 0 ]] && PATHS+=("$@")

echo "indexing into $INDEX ..."
cd "$ROOT"
GRAPHRAG_VECTOR_STORAGE=NanoVectorDBStorage \
GRAPHRAG_WORKING_DIR="$INDEX" \
GRAPHRAG_EXTRACTOR_BASE_URL="http://localhost:$PORT/v1" \
GRAPHRAG_EXTRACTOR_MODEL=gemma-4-12b \
GRAPHRAG_EXTRACTOR_MAX_TOKENS=4096 \
GRAPHRAG_LLM_MAX_TOKEN_SIZE=24576 \
GRAPHRAG_CHUNK_TOKEN_SIZE=2048 \
GRAPHRAG_CHUNK_OVERLAP_TOKENS=256 \
GRAPHRAG_MAX_PARALLEL_INSERT=6 \
GRAPHRAG_INSERT_BATCH_SIZE=99999 \
GRAPHRAG_LLM_MAX_ASYNC=6 \
GRAPHRAG_MAX_GLEANING=0 \
GRAPHRAG_LLM_TIMEOUT_SECONDS=600 \
LLM_TIMEOUT=600 \
GRAPHRAG_EMBED_MODEL=unsloth/embeddinggemma-300m \
GRAPHRAG_EMBED_DEVICE=cpu \
GRAPHRAG_EMBED_LOCAL_FILES_ONLY=1 \
HF_HOME="$HF_HOME" \
PYTHONPATH="$ROOT" \
uv run --project tools/graphrag python tools/graphrag/server/index.py \
    --paths "${PATHS[@]}" > "$LOG" 2>&1

echo "indexing done; archiving ingested notes..."
mkdir -p "$STAGING/ingested"
find "$STAGING" -maxdepth 1 -name '*.md' -exec mv {} "$STAGING/ingested/" \;
echo "KB update complete. Log: $LOG"
