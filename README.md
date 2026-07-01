# LiveMCP-GRPO

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![veRL 0.6.1](https://img.shields.io/badge/veRL-0.6.1-orange.svg)](https://github.com/volcengine/verl)

> **State-Machine Data Synthesis × Constrained GRPO for Multi-Step MCP Tool-Use Agents**
>
> PROVE-style LLM-in-the-loop teacher + live MCP execution + event-sourced reward + 2D stratified advantage.
> Trains agents on **10 domains, 188 tools** across banking, calendar, email, filesystem, CRM, payments, shopping, food delivery, issue tracking, and team chat.

---

## 🎯 Problem & Motivation

Multi-step tool-use agents face a **data-quality deadlock** — pure LLM synthesis produces hallucinatory traces, while pure template generation lacks realistic variability. When combined with GRPO training, outcome-only binary rewards cause **group saturation** and **collapsing reasoning depth**.

We solve this with two integrated innovations:

1. **PROVE State-Machine Synthesis** — LLM-in-the-loop teacher with full domain context (tools + live state + execution history), producing verified oracle traces through real MCP servers.
2. **Constrained GRPO with Event-Sourced Reward** — `J = R_task + I_shape·λ_shape·F_gamma + I_process·λ_process·P_process − λ_safe·C_safety`, paired with 2D stratified advantage (`perturbation_level × scenario_type`) and LATA (Length-Aware Turn-Advantage).

---

## 💡 Method

### Data Generation: PROVE State-Machine

```
Per domain, per persona:
  1. Deterministic state seeding (state_seeder.py, 10 domains)
  2. LLM-in-the-loop (task_planner.py):
     teacher sees full domain context → plans next action (tool_call / clarification / terminal)
  3. Real MCP execution (executor.py) → oracle trace
  4. State delta → success_criteria (state_equals / state_exists / file_exists / …)
  5. Perturbation injection:
     · distractor tools (30%) — inject 3–8 irrelevant tools
     · missing function (20%) — hide a required tool
     · irrelevance query (5%) — request report_error for unrelated query
  6. Position-aware Jaccard dedup (0.70 threshold)
  7. Convert to verl-compatible parquet (prompt + reward_model + scenario_type)
```

Default difficulty distribution: **complete 60% / missing 20% / minimal 20%**.

### GRPO Training: Constrained Advantage + Event-Sourced Reward

```
parquet → vLLM rollout → LiveMCPOvalLoop (real MCP execution + audit events)
  → Reward: J = R_task + I_shape·λ_shape·F_gamma + I_process·λ_process·P_process − λ_safe·C_safety
  → 2D StratAdv (perturbation_level × scenario_type) + LATA + saturation skip
  → FSDP gradient update
```

**Key design decisions:**
- **Event-sourced audit** — every tool call/result/safety violation produces an `AuditEvent`, not just final outcome.
- **LATA** — replaces linear `advantage / L` normalization with `advantage / √L`, preserving marginal incentives for long reasoning chains.
- **Saturation skip** — when a perturbation-level × scenario-type group reaches all-0 or all-1 rewards, skip its gradient to prevent deadlock.
- **Adaptive λ** — shape/process reward coefficients update based on training dynamics.

---

## 🏗️ Project Structure

```
📦 livemcp-grpo/
├── ⚙️ configs/                     # Hydra YAML configs
│   ├── live_mcp/                   # 10 domain configs + suite_mvp.yaml
│   ├── agent_loop.yaml             # Agent loop registration
│   └── ds_zero2.json               # DeepSpeed ZeRO-2
│
├── 💻 src/
│   ├── 🏗️ live_mcp/                 # MCP environment + data synthesis
│   │   ├── task_planner.py         #   PROVE state-machine (LLM-in-the-loop)
│   │   ├── orchestrator.py         #   Task orchestration + perturbation + yield guard
│   │   ├── state_seeder.py         #   Deterministic state seeding (10 domains)
│   │   ├── dedup.py                #   Position-aware Jaccard dedup (0.70)
│   │   ├── api.py                  #   LiveMCPBranch unified API
│   │   ├── executor.py             #   Tool executor
│   │   ├── manager.py              #   MCP server lifecycle
│   │   ├── oracle.py               #   Oracle program builder
│   │   ├── reward.py               #   Reward computation (live_mcp side)
│   │   ├── schema_registry.py      #   Schema registry & query
│   │   ├── state_seeder.py         #   10-domain seed state generation
│   │   ├── transport.py            #   Subprocess stdio transport
│   │   ├── types.py                #   Core types (LiveTask, OracleCall, …)
│   │   └── servers/ × 10          #   MCP subprocess servers
│   │
│   ├── 🔄 agent_loop/              # verl Agent Loop
│   │   ├── livemcp_oval_loop.py    #   LiveMCPOvalLoop ("livemcp_oval")
│   │   └── oval_mcp_worker.py      #   Session + audit wrapper
│   │
│   ├── 🎯 oval_mcp/                # Reward + constrained GRPO algorithm
│   │   ├── rewards/
│   │   │   ├── task_reward.py      #   R_task: task completion reward
│   │   │   ├── f_gamma.py          #   F_gamma: efficiency shaping
│   │   │   ├── p_process.py        #   P_process: process reasonableness
│   │   │   └── scalar_return.py    #   ScalarReturn aggregation
│   │   ├── verifier/
│   │   │   ├── events.py           #   AuditEvent / EventLog / TrajectoryEventLog
│   │   │   └── safety.py           #   C_safety: safety constraint verification
│   │   ├── envs/
│   │   │   ├── domain_adapter.py   #   DomainAdapter
│   │   │   └── audit_wrapper.py    #   AuditWrapper
│   │   └── training/
│   │       ├── lambda_state.py     #   Adaptive λ update
│   │       ├── lata.py             #   LATA: Length-Aware Turn-Advantage
│   │       └── saturation.py       #   Saturation group skip
│   │
│   ├── 🎁 reward/                  # verl reward entry
│   │   ├── action_parser.py        #   Action parser
│   │   └── oval_reward_fn.py       #   compute_score() entry point
│   │
│   ├── 🏋️ training/                # verl training components
│   │   ├── run_grpo.py             #   Official training entry
│   │   ├── trainer_config.py       #   PyTorch Lightning-style config
│   │   ├── livemcp_grpo_estimator.py #  2D StratAdv + LATA estimator
│   │   ├── livemcp_advantage.py    #   Advantage factory
│   │   ├── livemcp_hyperparams.py  #   Hyperparameters
│   │   ├── livemcp_task_runner.py  #   TaskRunner
│   │   ├── hooks.py                #   Training hooks
│   │   ├── length_check.py         #   Prompt length pre-check
│   │   ├── register_estimator.py   #   Estimator registration + λ update
│   │   └── advantage_core.py       #   Core advantage computation
│   │
│   └── utils.py                    # Utilities
│
├── 📜 scripts/
│   ├── generate_data.py            # Data generation CLI (PROVE state-machine)
│   ├── generate_data.sh            # Unified launcher (auto-detect GPU topology)
│   ├── train_grpo.py               # GRPO training Python entry (Hydra)
│   ├── train_grpo.sh               # GRPO training Shell entry
│   ├── gpu_config.sh               # GPU topology auto-detection
│   ├── inspect_prompts.py          # Prompt inspection utility
│   └── test_runner.py              # Production smoke test runner
│
├── 🧪 tests/
│   ├── test_all_domains.py         # Full 10-domain tool coverage test
│   ├── test_reward_contract.py     # Reward function contract test
│   └── test_training_data_contract.py # Training data schema test
│
├── 📚 docs/
│   └── OVAL-MCP.md                 # Authoritative design document
│
├── 🗂️ data/                        # Training data + experiment records
│   └── README.md                   # Data directory specification
│
├── 📖 reference/                   # Reference papers (PROVE et al.)
│
├── 🧩 verl/                        # verl framework (vendored, editable install)
│
├── 📄 pyproject.toml               # Project metadata & dependencies
├── 📄 requirements.txt             # pip dependencies
└── 📄 CLAUDE.md                    # Entry point + pipeline status
```

---

## 🚀 Quick Start

### 1. Environment

```bash
conda create -n arl python=3.11 -y
conda activate arl

# Install verl from vendored source
pip install -e ./verl

# Install project
pip install -e .
pip install -e ".[train,rl]"
```

### 2. Data Generation

```bash
# Unified launcher (auto-detect model size + GPU topology)
bash scripts/generate_data.sh --model models/Qwen/Qwen3-32B --count 500 --val-count 100

# Single domain quick test
bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --domain calendar --count 200

# Direct Python CLI (advanced)
python scripts/generate_data.py \
    --model Qwen3-32B \
    --api-base http://localhost:8000/v1 \
    --count 500 --val-count 100
```

### 3. GRPO Training

```bash
# Default training
bash scripts/train_grpo.sh

# Custom GPU and steps
bash scripts/train_grpo.sh --gpus 0,1,2,3 --total-steps 300

# With WandB logging
bash scripts/train_grpo.sh --wandb --wandb-project oval-mcp-grpo
```

### 4. Validate

```bash
# Full domain tool coverage
python tests/test_all_domains.py

# Lightweight syntax check
python -m compileall src scripts tests
```

> **Hardware**: 8×L20 44GB (default). GPU tier auto-detection via `scripts/gpu_config.sh` — adapts batch size, TP size, and parallelism automatically.
> **Teacher model**: Qwen3-32B (vLLM TP=4, GPU 4-7). **Policy model**: Qwen3-4B (GPU 0-3).

---

## 🌟 Technical Highlights

### 1. PROVE State-Machine Synthesis (Data Quality)
A verified, LLM-in-the-loop data generation pipeline that produces **execution-grounded oracle traces**. Unlike pure LLM synthesis, every tool call in the trace is executed against a real MCP server and verified before inclusion.

**Six invariants** enforced at generation time:
- Tool calls in prompt ≡ tool calls in oracle trace
- All domain tools present (visible + hidden)
- Chain length ≤ 5 (three-layer hardware cap)
- Every `tool_result` matches an executed `tool_call`
- `missing_function` tasks have empty oracle
- No teacher trace leakage into prompt

### 2. Event-Sourced Reward (Signal Richness)
Instead of binary outcome-only reward, the system captures a full `TrajectoryEventLog` — every tool call, result, safety check, and schema validation produces an auditable event. Reward decomposition:
- **R_task** — task completion coverage (state criteria + operation assertions)
- **F_gamma** — efficiency shaping (penalizes unnecessarily long chains)
- **P_process** — process reasonableness (redundancy, placeholder, reasoning quality)
- **C_safety** — safety constraint violations (schema validity, execution success)

### 3. 2D Stratified Advantage + LATA (Training Stability)
- **2D StratAdv** — groups trajectories by `perturbation_level × scenario_type`, computes per-group advantage to prevent cross-group signal contamination.
- **LATA** — `advantage / √L` replaces `advantage / L`, preserving marginal incentives for longer reasoning.
- **Saturation Skip** — detects all-0/all-1 groups and skips their gradients.
- **Adaptive λ** — shape/process coefficients update based on reward statistics during training.

### 4. Multi-Tier GPU Auto-Adaptation (Engineering)
Single script (`scripts/gpu_config.sh`) detects GPU model (L20/A100/A10/Hopper/T4), VRAM, and count, then selects optimal batch sizes, TP sizes, and parallelism strategies — no manual tuning needed.

---

## 📚 Documentation

| Document | Content |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | Entry point: pipeline status, constraints, common commands |
| [docs/OVAL-MCP.md](docs/OVAL-MCP.md) | Authoritative design: reward function, constrained GRPO, event system |
| [data/README.md](data/README.md) | Data directory spec, parquet schema, experiment record format |
| [configs/README.md](configs/README.md) | Config file reference, environment variable overrides |

---

## 🛠️ Tech Stack

- **Training Framework**: [veRL](https://github.com/volcengine/verl) 0.6.1 (FSDP + vLLM V1)
- **Teacher Model**: Qwen3-32B (vLLM)
- **Policy Model**: Qwen3-4B
- **Benchmark**: 10-domain Live MCP (188 tools)
- **Inference Engine**: vLLM 0.11.0
- **RL Algorithm**: GRPO with 2D stratified advantage + LATA
- **Attention**: FlashAttention-2, FlashInfer

---

## 📜 Pipeline Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Generation Pipeline | ✅ | Complete 2–5 step oracle + stratified split |
| OVAL Agent Loop | ✅ | Single-call protocol + initial-state hash + final-state evidence |
| OVAL Reward | ✅ | Ordered coverage + task-aware safety |
| GRPO Estimator | ✅ | Saturation skip + 2D stratified advantage |
| GPU Auto-Adaptation | ✅ | Multi-tier (L20/A100/A10/Hopper/T4) |
| Full Training Run | ⏳ | Pending data generation |

---

## 🔧 Design Constraints

- Training scripts must not hardcode GPU count, batch size, micro batch, or TP size.
- All project paths use repo-root-relative paths — no absolute machine paths.
- Training hyperparams injectable via CLI args, environment variables (`OVAL_*` prefix), or Hydra override.
- `data.max_prompt_length` ≥ 10240.
- Ray temp dir uses short path (`/tmp/ssgrpo_ray`) to avoid AF_UNIX socket path overflow.
- Illegal tool JSON does not produce `AuditEvent` (model receives error observation instead).

---

## 🙏 Acknowledgements

- [veRL](https://github.com/volcengine/verl) — open-source RL training framework
- [Qwen](https://github.com/QwenLM/Qwen) — strong base policy models
- PROVE paper — state-machine data synthesis methodology

---

## 📄 License

MIT License.
