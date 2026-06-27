#!/bin/bash
# GRPO training entry point — flexible multi-GPU via gpu_config.sh.
#
# Usage:
#   bash scripts/train_grpo.sh                           # all detected GPUs
#   bash scripts/train_grpo.sh --gpus 0,1,2,3            # specific GPUs
#   bash scripts/train_grpo.sh --total-steps 100         # override steps
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train_grpo.sh  # env override
#   GPU_COUNT=4 bash scripts/train_grpo.sh               # limit GPU count
#
# Env var overrides:
#   OVAL_TRAIN_FILE, OVAL_VAL_FILE     -- data paths
#   OVAL_TOTAL_STEPS                   -- training steps
#   OVAL_ROLLOUT_N                     -- rollouts per group
#   OVAL_RESPONSE_LENGTH               -- max response tokens
#   OVAL_GPU_MEM_UTIL                  -- override GPU memory utilization

set -euo pipefail

# ---- vLLM orphan cleanup trap ----
# verl/vLLM rollout can leave zombie EngineCore processes.
# This trap runs on any exit (normal, error, or signal) to clean them up.
_cleanup_vllm_orphans() {
    local exit_code=$?
    VLLM_ORPHANS=$(ps -eo pid,comm --no-headers 2>/dev/null | awk '/VLLM::EngineCore/{print $1}' || true)
    if [ -n "$VLLM_ORPHANS" ]; then
        echo "[cleanup] Killing orphaned VLLM::EngineCore: $VLLM_ORPHANS" >&2
        for pid in $VLLM_ORPHANS; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
    exit $exit_code
}
trap _cleanup_vllm_orphans EXIT INT TERM

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- environment ----
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HYDRA_FULL_ERROR=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DEDUP_LOGS=1
export LOGURU_LEVEL=INFO
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export TMPDIR="${TMPDIR:-/tmp/ssgrpo_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ssgrpo_ray}"
export LIVEMCP_RAY_TMPDIR="${LIVEMCP_RAY_TMPDIR:-${RAY_TMPDIR}}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" outputs

# ---- cleanup stale Ray ----
if ray status &>/dev/null 2>&1; then
    echo "[cleanup] Stopping stale Ray cluster..."
    ray stop --force 2>/dev/null || true
    sleep 2
fi

# ---- GPU detection (flexible, via gpu_config.sh) ----
# Accept --gpus <ids> argument, pass remaining args through
GPU_ARG=""
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPU_ARG="$2"; shift 2 ;;
        --gpus=*) GPU_ARG="${1#*=}"; shift ;;
        *) REMAINING_ARGS+=("$1"); shift ;;
    esac
done
set -- "${REMAINING_ARGS[@]}"

if [ -n "${GPU_ARG}" ]; then
    . scripts/gpu_config.sh "${GPU_ARG}"
else
    . scripts/gpu_config.sh
fi

# Allow env override of mem_util
GPU_MEM_UTIL="${OVAL_GPU_MEM_UTIL:-${GPU_MEM_UTIL}}"

# ---- per-tier defaults (overrideable via env) ----
if [ "${GPU_TIER}" = "L20" ]; then
    PROMPT_LENGTH="${OVAL_PROMPT_LENGTH:-12384}"
    RESPONSE_LENGTH="${OVAL_RESPONSE_LENGTH:-16384}"
    MAX_NUM_SEQS="${OVAL_MAX_NUM_SEQS:-64}"
    MICRO_BATCH="${OVAL_MICRO_BATCH:-2}"
    TRAIN_BATCH_SIZE="${OVAL_TRAIN_BATCH_SIZE:-32}"
    MINI_BATCH_SIZE="${OVAL_MINI_BATCH_SIZE:-8}"
    ROLLOUT_N="${OVAL_ROLLOUT_N:-9}"
elif [ "${GPU_TIER}" = "A10" ]; then
    PROMPT_LENGTH="${OVAL_PROMPT_LENGTH:-10240}"
    RESPONSE_LENGTH="${OVAL_RESPONSE_LENGTH:-4096}"
    MAX_NUM_SEQS="${OVAL_MAX_NUM_SEQS:-8}"
    MICRO_BATCH="${OVAL_MICRO_BATCH:-1}"
    TRAIN_BATCH_SIZE="${OVAL_TRAIN_BATCH_SIZE:-8}"
    MINI_BATCH_SIZE="${OVAL_MINI_BATCH_SIZE:-8}"
    ROLLOUT_N="${OVAL_ROLLOUT_N:-8}"
elif [ "${GPU_TIER}" = "A100" ] || [ "${GPU_TIER}" = "Hopper" ]; then
    PROMPT_LENGTH="${OVAL_PROMPT_LENGTH:-16384}"
    RESPONSE_LENGTH="${OVAL_RESPONSE_LENGTH:-16384}"
    MAX_NUM_SEQS="${OVAL_MAX_NUM_SEQS:-128}"
    MICRO_BATCH="${OVAL_MICRO_BATCH:-4}"
    TRAIN_BATCH_SIZE="${OVAL_TRAIN_BATCH_SIZE:-64}"
    MINI_BATCH_SIZE="${OVAL_MINI_BATCH_SIZE:-16}"
    ROLLOUT_N="${OVAL_ROLLOUT_N:-16}"
else
    PROMPT_LENGTH="${OVAL_PROMPT_LENGTH:-10240}"
    RESPONSE_LENGTH="${OVAL_RESPONSE_LENGTH:-2048}"
    MAX_NUM_SEQS="${OVAL_MAX_NUM_SEQS:-8}"
    MICRO_BATCH="${OVAL_MICRO_BATCH:-1}"
    TRAIN_BATCH_SIZE="${OVAL_TRAIN_BATCH_SIZE:-8}"
    MINI_BATCH_SIZE="${OVAL_MINI_BATCH_SIZE:-8}"
    ROLLOUT_N="${OVAL_ROLLOUT_N:-4}"
fi

# ---- algorithm parameters (hardware-independent) ----
MODEL_PATH="${OVAL_MODEL_PATH:-models/Qwen3-4B}"
REWARD_FN_PATH="src/reward/oval_reward_fn.py"
AGENT_LOOP="livemcp_oval"

# env var overrides (OVAL_* prefix)
TRAIN_FILE="${OVAL_TRAIN_FILE:-data/train.parquet}"
VAL_FILE="${OVAL_VAL_FILE:-data/val.parquet}"
TOTAL_STEPS="${OVAL_TOTAL_STEPS:-100}"

LR="${OVAL_LR:-1e-6}"
LR_WARMUP_RATIO="${OVAL_LR_WARMUP_RATIO:-0.1}"
KL_COEF="${OVAL_KL_COEF:-0.01}"
PPO_EPOCHS="${OVAL_PPO_EPOCHS:-1}"
GRAD_CLIP="${OVAL_GRAD_CLIP:-1.0}"
TEMPERATURE="${OVAL_TEMPERATURE:-0.7}"
TOP_P="${OVAL_TOP_P:-0.95}"
ROLLOUT_TP="${OVAL_ROLLOUT_TP:-1}"
LOG_PROB_MICRO_BATCH="${OVAL_LOG_PROB_MICRO_BATCH:-1}"
VAL_BATCH_SIZE="${OVAL_VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"

# FSDP offload: defaults from gpu_config.sh, env overrides
ACTOR_PARAM_OFFLOAD="${OVAL_ACTOR_PARAM_OFFLOAD:-${PARAM_OFFLOAD}}"
REF_PARAM_OFFLOAD="${OVAL_REF_PARAM_OFFLOAD:-${PARAM_OFFLOAD}}"

# Phase 1 default: R_task + C_safety only (no F_gamma / P_process)
export OVAL_I_SHAPE=0
export OVAL_I_PROCESS=0

echo "============================================"
echo "OVAL-MCP GRPO Training"
echo "============================================"
echo "MODEL:     ${MODEL_PATH}"
echo "GPU:       ${GPU_COUNT}x ${GPU_TIER} (${GPU_MODEL})"
echo "GPU_IDS:   ${GPU_IDS}"
echo "TRAIN:     ${TRAIN_FILE}"
echo "VAL:       ${VAL_FILE}"
echo "REWARD:    ${REWARD_FN_PATH}"
echo "AGENT:     ${AGENT_LOOP}"
echo "STEPS:     ${TOTAL_STEPS}"
echo "ROLLOUT_N: ${ROLLOUT_N}"
echo "BATCH:     ${TRAIN_BATCH_SIZE} (mini=${MINI_BATCH_SIZE}, micro=${MICRO_BATCH})"
echo "RESPONSE:  ${RESPONSE_LENGTH}"
echo "PROMPT:    ${PROMPT_LENGTH}"
echo "MEM_UTIL:  ${GPU_MEM_UTIL}"
echo "FSDP:      actor_offload=${ACTOR_PARAM_OFFLOAD} ref_offload=${REF_PARAM_OFFLOAD}"
echo "============================================"

# ---- register estimator ----
export PYTHONPATH=".:${PYTHONPATH:-}"

# ---- validate data ----
echo ""
echo "=== Validating data ==="
python3 -c "
import sys, pandas as pd
for path in ['${TRAIN_FILE}', '${VAL_FILE}']:
    df = pd.read_parquet(path)
    domains = set()
    for _, row in df.iterrows():
        domains.add(row['extra_info']['domain'])
    print(f'  {path}: {len(df)} rows, domains={sorted(domains)}')
    if len(df) > 0:
        ei = df.iloc[0]['extra_info']
        print(f'    sample: domain={ei.get(\"domain\")}, scenario={ei.get(\"scenario_type\")}')
"
echo ""

# ---- launch training ----
CONDA_PYTHON="${CONDA_PYTHON:-python3}"
exec "${CONDA_PYTHON}" "scripts/train_runner.py" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.max_prompt_length="${PROMPT_LENGTH}" \
    data.max_response_length="${RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.reward_fn_key=data_source \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.grad_clip="${GRAD_CLIP}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${LR_WARMUP_RATIO}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}" \
    actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$((PROMPT_LENGTH + RESPONSE_LENGTH))" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH}" \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.default_agent_loop="${AGENT_LOOP}" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/agent_loop.yaml \
    actor_rollout_ref.rollout.agent.num_workers="${GPU_COUNT}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
    algorithm.adv_estimator=livemcp_grpo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    trainer.project_name=mcp_grpo \
    trainer.experiment_name=train \
    trainer.logger='["console"]' \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${GPU_COUNT}" \
    trainer.save_freq=50 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    reward_model.enable=False \
    2>&1 | tee "outputs/train_grpo.log"

echo ""
echo "=== Training Complete ==="
echo "Check outputs/train_grpo.log for results"
