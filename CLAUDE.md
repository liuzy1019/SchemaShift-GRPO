# CLAUDE.md — LiveMCP-GRPO

Design doc: `docs/OVAL-MCP.md`. Read it before making changes.

## Environment

```bash
conda activate arl          # Python 3.11, PyTorch 2.8, CUDA 12.8
nvidia-smi                  # Confirm GPU availability
```

FlashInfer JIT must be disabled:

```bash
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
```

## Pipeline Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Generation | ✅ | PROVE state-machine, complete 2–5 step oracle + stratified split |
| OVAL Agent Loop | ✅ | Single-call protocol + initial-state hash + final-state evidence |
| OVAL Reward | ✅ | Ordered coverage + task-aware safety |
| GRPO Estimator | ✅ | Saturation skip + 2D stratified advantage |
| GPU Auto-Adaptation | ✅ | Multi-tier (L20/A100/A10/Hopper/T4) |
| Full Training Run | ⏳ | Pending data generation |

### Verified Pipeline

```
live MCP servers (10 domains)
→ PROVE Teacher (LLM-in-the-loop, Qwen3-32B)
→ Real MCP execution → oracle trace
→ Jaccard dedup (0.70, position-aware)
→ Parquet serialization (success_criteria as JSON string)
→ oval_reward_fn.py (R_task + C_safety, optional F_gamma/P_process)
→ verl GRPO training
```

## Training Route

| Route | Agent Loop | Reward Fn | Entry | Status |
|-------|-----------|-----------|-------|--------|
| OVAL GRPO | `livemcp_oval` | `oval_reward_fn.py` | `bash scripts/train_grpo.sh` | ✅ Primary |

`scripts/train_grpo.py` is the Hydra entry point; `src/training/run_grpo.py` is the official training entry.
Config managed by `src/training/trainer_config.py` (PyTorch Lightning style), with GPU tier defaults and `OVAL_*` env var overrides.

### Current Hardware

- Teacher model: Qwen3-32B (vLLM TP=4, GPU 4–7, 4×L20 44GB)
- Policy model: Qwen3-4B (`models/Qwen/Qwen3-4B`)
- Default environment: 8×L20 44GB

## Constraints

- Training scripts **must not hardcode** GPU count, batch size, micro batch, TP size.
- All project paths use **repo-root-relative** paths — no absolute machine paths.
- Training hyperparams injectable via CLI args, environment variables (`OVAL_*` prefix), or Hydra override.
- `data.max_prompt_length` ≥ 10240.
- Ray temp dir: short path (`/tmp/ssgrpo_ray`) to avoid AF_UNIX socket path > 107 bytes.
- SFT cold-start code has been removed. Only GRPO route exists.
- `success_criteria` value field is mixed-type (str/float/int), serialized as JSON string in parquet.
- `OracleCall(action="clarification")` must be preserved in parquet; reward side sets `allowed_terminal=["ask_clarification"]`.

## Known Design Limitations

| Issue | Explanation |
|-------|-------------|
| Illegal tool JSON → no AuditEvent | Model output format errors only produce error observation, no audit event. Fix requires cross-module type extension. |
| Reward uses last observation as final state | Exact value verification may miss during consecutive tool_calls (low probability, has seen_ids fallback). |
| Perturbation only in teacher phase | PROVE design: perturbation for teacher robustness testing, clean training environment. |

## Common Commands

```bash
# Data generation
bash scripts/generate_data.sh --model models/Qwen/Qwen3-32B --count 500 --val-count 100
bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --domain calendar --count 200

# GRPO training
bash scripts/train_grpo.sh
bash scripts/train_grpo.sh --gpus 0,1,2,3 --total-steps 300
bash scripts/train_grpo.sh --wandb --wandb-project oval-mcp-grpo

# Validation
python -m pytest tests/
python -m compileall src scripts tests
git diff --check
```

## Git

```text
Remote: https://github.com/liuzy1019/LiveMCP-GRPO
Branch: main
Author: liuzy1019 <liuzy1019@buaa.edu.cn>
```

Conventional Commits: `<type>: <subject>`

| Type | Use |
|------|-----|
| feat | New feature / experiment / estimator |
| fix | Bug fix |
| docs | Documentation |
| refactor | Behavior-preserving refactor |
| test | Tests |
| chore | Config / build / deps |
| perf | Performance |

Do not push without verification. Test before commit.
