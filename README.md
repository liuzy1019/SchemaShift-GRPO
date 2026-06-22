# SchemaShift-GRPO

> MCP-Tools-Aware Tool-Use RL — Schema Robustness via Online Rollout + Verifiable Reward

通过 Schema Perturbation + Distractor Injection + Shaped Reward + Stratified Advantage，让模型在陌生 schema、多工具干扰下学会稳定的 tool-use 决策。

---

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 数据准备 | ✅ 完成 | Toucan inspection + EpisodeSeed 构建 + SFT 样本导出 (9146条) |
| SFT Cold-Start (4B) | 🔄 待重跑 | Qwen3-4B, 8×L20 44GB, DeepSpeed ZeRO-2, effective_batch=128 |
| SFT 生成验证 | ⬜ 待 SFT 完成 | 需重新验证 |
| RL 闭环 | 🔄 smoke 调试中 | ReplayMCPExecutor / MCPToolEnvironment / TrajectoryVerifier 已实现，verl GRPO smoke 待重跑 |
| Live MCP MVP | ✅ smoke 可运行 | calendar/shopping subprocess stdio server + LiveTask 生成 + offline agent loop + 五组件 reward |
| 评测 | ⬜ 待实现 | Self MCP Robustness Set + BFCL V3 |

**下一步**：在 L20 上跑 Qwen3-4B SFT cold-start，然后重跑 verl GRPO smoke。Live MCP 当前只作为显式 smoke/offline 后端，不替换默认 replay 训练链路。

---

## 项目结构

```
schemashift-grpo/
├── mcp_tools_rl_project_plan.md   # 权威方案文档
├── CLAUDE.md                       # AI 工作入口
├── AGENTS.md                       # AI 协作约定
├── src/
│   ├── envs/                       # schema_perturber + api_mapper
│   ├── data/                       # episode_seed_builder + sft_step_exporter + distractor_sampler
│   ├── reward/                     # action_parser + component_reward
│   ├── eval/                       # matching
│   ├── training/                   # schemashift_advantage + grpo_estimator + register + length_check
│   ├── live_mcp/                   # live MCP MVP: subprocess stdio servers + offline rollout/reward
│   └── agent_loop/                 # 默认 GRPO 多轮 agent loop 待接入
├── scripts/                        # 训练/数据/环境脚本
├── configs/                        # DeepSpeed 配置
├── data/                           # 训练/评测数据 (gitignored)
├── tests/                          # 单元测试 (240+ passed)
├── verl/                           # verl 源码 (editable)
├── requirements.txt
└── pyproject.toml
```

---

## 环境搭建

```bash
# 已有 conda env: arl
conda activate arl
python -m pip install -e ./verl
python -m pip install -e .
python scripts/check_dependency_conflicts.py
```

**硬件要求**：8×L20 (44GB)；A10 (23GB) 可用于 SFT（batch=1 + ZeRO-2）但不建议跑 GRPO。

---

## 测试

```bash
pytest tests/  # 240+ passed
```

## GRPO Smoke

```bash
# SFT cold-start（8×L20）
torchrun --nproc_per_node=8 scripts/sft_cold_start.py \
    --config configs/sft_cold_start_4b.yaml

# 数据准备
python scripts/prepare_grpo_data.py \
    --episode_seeds data/toucan/episode_seeds.jsonl \
    --output data/grpo_train.parquet \
    --val_output data/grpo_val.parquet

# Smoke test
bash scripts/run_grpo_smoke.sh --config configs/grpo_smoke.yaml
```

脚本默认以项目根目录为锚点，`data/...`、`outputs/...`、`src/...` 均使用相对路径。常用超参可以通过命令行注入，未知参数会继续透传为 Hydra overrides：

```bash
bash scripts/run_grpo_smoke.sh \
    --config configs/grpo_smoke.yaml \
    --n-gpus 4 \
    --cuda-visible-devices 0,1,2,3 \
    --total-steps 2 \
    --lr 1e-6 \
    --rollout-n 2 \
    --prompt-length 10240 \
    --response-length 1024 \
    --micro-batch 1
```

正式 GRPO 也支持 YAML，并且命令行参数会覆盖 YAML：

```bash
bash scripts/run_grpo.sh \
    --config configs/grpo_train.yaml \
    --lr 5e-7 \
    --rollout-n 4
```

---

## Live MCP Smoke

Live MCP 默认不接入 GRPO，只通过显式脚本运行：

```bash
python scripts/generate_live_mcp_tasks.py \
    --suite configs/live_mcp/suite_mvp.yaml \
    --output data/live_mcp/tasks/live_mcp_mvp.jsonl \
    --num-tasks 20 \
    --seed 42

python scripts/run_live_mcp_smoke.py \
    --suite configs/live_mcp/suite_mvp.yaml \
    --tasks data/live_mcp/tasks/live_mcp_mvp.jsonl \
    --server calendar \
    --num-tasks 10 \
    --seed 42
```

第一版 live MVP 覆盖 calendar/shopping 两个 stateful subprocess stdio server、session deterministic reset、grounded task sampling、oracle validation、trace 落盘和五组件 execution-aware reward。

---

## 文档

| 文档 | 内容 |
|------|------|
| [`mcp_tools_rl_project_plan.md`](./mcp_tools_rl_project_plan.md) | 权威方案：架构、数据、reward、评测、阶段计划 |
| [`docs/live_mcp_branch.md`](./docs/live_mcp_branch.md) | Live MCP 并行分支：API、CLI、边界 |
| [`CLAUDE.md`](./CLAUDE.md) | AI 工作入口：读取顺序、开发边界、验证命令 |
| [`AGENTS.md`](./AGENTS.md) | AI 协作约定：环境、红线、git、代码约束 |

---

## 许可

MIT License.
