#!/bin/bash
# Data-parallel task generation: splits work across N GPUs.
# Each GPU runs an independent model copy generating a shard of tasks.
#
# Usage:
#   bash scripts/generate_data_parallel.sh --count 500 --model models/Qwen3-8B
#   bash scripts/generate_data_parallel.sh --gpus 0,1,2,3 --count 100 --domain calendar
#   GPU_COUNT=4 bash scripts/generate_data_parallel.sh --count 200
#
# Output:
#   data/train.parquet, data/val.parquet  (merged from all shards)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- Parse --gpus and --output-dir from args, extract count/val-count ----
GPU_ARG=""
OUTPUT_DIR="${OUTPUT_DIR:-data}"
GEN_ARGS=()
TRAIN_COUNT=500
VAL_COUNT=50
PREV=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)      GPU_ARG="$2"; shift 2 ;;
        --gpus=*)    GPU_ARG="${1#*=}"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --count)     TRAIN_COUNT="$2"; shift 2 ;;  # extract, don't forward
        --count=*)   TRAIN_COUNT="${1#*=}"; shift ;;
        --val-count) VAL_COUNT="$2"; shift 2 ;;     # extract, don't forward
        --val-count=*) VAL_COUNT="${1#*=}"; shift ;;
        *)           GEN_ARGS+=("$1"); shift ;;
    esac
done

# ---- GPU detection ----
GPU_FREE_ONLY=1 . scripts/gpu_config.sh ${GPU_ARG}

# Calculate per-GPU split counts
PER_GPU_TRAIN=$(( (TRAIN_COUNT + GPU_COUNT - 1) / GPU_COUNT ))
PER_GPU_VAL=$(( (VAL_COUNT + GPU_COUNT - 1) / GPU_COUNT ))

echo "============================================"
echo "Data-Parallel Task Generation"
echo "============================================"
echo "GPUs:     ${GPU_COUNT}x ${GPU_TIER} (${GPU_IDS})"
echo "Train:    ${TRAIN_COUNT} total (${PER_GPU_TRAIN} per GPU)"
echo "Val:      ${VAL_COUNT} total (${PER_GPU_VAL} per GPU)"
echo "Args:     ${GEN_ARGS[*]}"
echo "Output:   ${OUTPUT_DIR}/train.parquet, ${OUTPUT_DIR}/val.parquet"
echo "============================================"

echo "Per GPU: train=${PER_GPU_TRAIN}, val=${PER_GPU_VAL}"

# ---- Launch parallel processes ----
TMPDIR_PARALLEL="${TMPDIR:-/tmp}/livemcp_gen_parallel_$$"
mkdir -p "${TMPDIR_PARALLEL}" "${OUTPUT_DIR}"

PIDS=()
for ((i=0; i<GPU_COUNT; i++)); do
    GPU_ID="${GPU_INDEX_ARRAY[$i]}"
    SHARD_TRAIN="${TMPDIR_PARALLEL}/shard_${i}_train.parquet"
    SHARD_VAL="${TMPDIR_PARALLEL}/shard_${i}_val.parquet"
    SHARD_LOG="${TMPDIR_PARALLEL}/shard_${i}.log"
    SHARD_SEED=$((42 + i * 10000))

    echo "[shard $i] GPU=${GPU_ID}, train=${PER_GPU_TRAIN}, val=${PER_GPU_VAL}, seed=${SHARD_SEED}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 scripts/generate_data.py \
        --count "${PER_GPU_TRAIN}" \
        --val-count "${PER_GPU_VAL}" \
        --seed "${SHARD_SEED}" \
        --output "${SHARD_TRAIN}" \
        --val-output "${SHARD_VAL}" \
        --log-file "${SHARD_LOG}" \
        --device 0 \
        "${GEN_ARGS[@]}" &
    PIDS+=($!)
done

# ---- Wait for all processes ----
echo ""
echo "Waiting for ${GPU_COUNT} processes..."
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    wait "$pid" || {
        echo "[shard $i] PID ${pid} FAILED (exit=$?)" >&2
        FAILED=$((FAILED + 1))
    }
done

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "ERROR: ${FAILED}/${GPU_COUNT} shards failed. Check logs in ${TMPDIR_PARALLEL}/"
    exit 1
fi

# ---- Merge shards ----
echo ""
echo "All shards complete. Merging..."

python3 -c "
import pandas as pd, sys
from pathlib import Path

tmpdir = Path('${TMPDIR_PARALLEL}')

# Merge train
train_dfs = []
for p in sorted(tmpdir.glob('shard_*_train.parquet')):
    try:
        df = pd.read_parquet(p)
        train_dfs.append(df)
        print(f'  {p.name}: {len(df)} rows')
    except Exception as e:
        print(f'  {p.name}: SKIP ({e})')

if train_dfs:
    merged = pd.concat(train_dfs, ignore_index=True)
    merged.to_parquet('${OUTPUT_DIR}/train.parquet', index=False)
    print(f'  → {len(merged)} rows → ${OUTPUT_DIR}/train.parquet')
else:
    print('  WARNING: No train data!')
    sys.exit(1)

# Merge val
val_dfs = []
for p in sorted(tmpdir.glob('shard_*_val.parquet')):
    try:
        df = pd.read_parquet(p)
        val_dfs.append(df)
        print(f'  {p.name}: {len(df)} rows')
    except Exception as e:
        print(f'  {p.name}: SKIP ({e})')

if val_dfs:
    merged = pd.concat(val_dfs, ignore_index=True)
    merged.to_parquet('${OUTPUT_DIR}/val.parquet', index=False)
    print(f'  → {len(merged)} rows → ${OUTPUT_DIR}/val.parquet')
else:
    print('  WARNING: No val data!')
"

# ---- Print stats ----
echo ""
echo "=== Generation Complete ==="
python3 -c "
import pandas as pd
for path in ['${OUTPUT_DIR}/train.parquet', '${OUTPUT_DIR}/val.parquet']:
    df = pd.read_parquet(path)
    domains = set()
    for _, row in df.iterrows():
        domains.add(row['extra_info']['domain'])
    print(f'{path}: {len(df)} rows, domains={sorted(domains)}')
    if len(df) > 0:
        ei = df.iloc[0]['extra_info']
        print(f'  sample: domain={ei.get(\"domain\")}, scenario={ei.get(\"scenario_type\")}')
"

# Cleanup temp files (keep logs for debugging)
rm -f "${TMPDIR_PARALLEL}"/shard_*_train.parquet "${TMPDIR_PARALLEL}"/shard_*_val.parquet
echo ""
echo "Logs: ${TMPDIR_PARALLEL}/"
