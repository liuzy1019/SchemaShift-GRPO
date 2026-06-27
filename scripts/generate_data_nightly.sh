#!/bin/bash
# nightly_32b_generate.sh
# Run 32B vLLM + full data generation (500 train + 100 val)
# Scheduled to run at late night when GPUs are free

set -euo pipefail

PROJECT_ROOT="/mnt/data2/liuzhanyi/livemcp-grpo"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/data"
mkdir -p "$LOG_DIR"

VLLM_LOG="$LOG_DIR/vllm_32b_nightly.log"
GEN_LOG="$LOG_DIR/generate_32b_nightly.log"

# ---- cleanup function: kill vLLM + orphaned children ----
_cleanup_vllm() {
    local exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleanup: stopping vLLM..." | tee -a "$GEN_LOG" 2>/dev/null || true

    # Kill the main vLLM process tree
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        # Wait up to 10s for graceful shutdown
        for _ in $(seq 1 10); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        # Force kill if still alive
        kill -KILL "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        echo "  vLLM PID $VLLM_PID stopped" | tee -a "$GEN_LOG" 2>/dev/null || true
    fi

    # Kill any orphaned VLLM::EngineCore processes left on the GPU
    VLLM_ORPHANS=$(ps -eo pid,comm --no-headers 2>/dev/null | awk '/VLLM::EngineCore/{print $1}' || true)
    if [ -n "$VLLM_ORPHANS" ]; then
        echo "  Killing orphaned VLLM::EngineCore: $VLLM_ORPHANS" | tee -a "$GEN_LOG" 2>/dev/null || true
        for pid in $VLLM_ORPHANS; do
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi

    exit $exit_code
}
trap _cleanup_vllm EXIT INT TERM

# Bypass flashinfer JIT compilation (CUDA 11.8 + GCC 12.3 incompatible with flashinfer 0.6.4 JIT)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen/Qwen3-32B \
    --served-model-name Qwen3-32B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.82 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --port 8001 \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID" | tee -a "$GEN_LOG"

# ---- Step 2: Wait for vLLM to be ready ----
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for vLLM to be ready..." | tee -a "$GEN_LOG"
MAX_WAIT=600  # 10 minutes
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "  vLLM ready after ${WAITED}s" | tee -a "$GEN_LOG"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "  ERROR: vLLM process died. Check $VLLM_LOG" | tee -a "$GEN_LOG"
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "  ERROR: vLLM not ready after ${MAX_WAIT}s" | tee -a "$GEN_LOG"
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
fi

# ---- Step 3: Generate training data ----
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating 500+100 OVAL data..." | tee -a "$GEN_LOG"

python scripts/generate_data.py \
    --count 500 \
    --val-count 100 \
    --domain all \
    --model Qwen3-32B-Instruct \
    --api-base http://localhost:8001/v1 \
    2>&1 | tee -a "$GEN_LOG"

GEN_EXIT_CODE=${PIPESTATUS[0]}

if [ $GEN_EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done! Data saved to data/train.parquet and data/val.parquet" | tee -a "$GEN_LOG"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generation FAILED with exit code $GEN_EXIT_CODE. Check $GEN_LOG" | tee -a "$GEN_LOG"
fi

exit $GEN_EXIT_CODE
