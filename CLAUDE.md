# CLAUDE.md — SchemaShift-GRPO 工作入口

本项目的权威方案文档是 `docs/project_plan.md`。进入开发前先读：

1. `docs/project_plan.md`
2. `AGENTS.md`
3. `README.md`
4. 与当前任务相关的 `data/README.md` 或输出报告

若文档之间出现冲突，处理顺序是：

```text
先核对代码/报告事实
-> 更新 docs/project_plan.md
-> 更新 AGENTS.md / README.md / data README
-> 再改实现
```

## 当前状态

- **模型**：Qwen3-4B（`models/Qwen3-4B`，BF16 ~8GB）
- **默认环境**：8×L20 44GB（conda env: `arl`）
- **SFT cold-start**：待在 L20 上重新训练（输出到 `outputs/sft_cold_start_4b/final`）
- **GRPO 训练**：smoke test 待 SFT 完成后验证

## 当前开发边界

下一阶段只推进最小 RL smoke 验证：

```text
SFT cold-start (Qwen3-4B)
-> verl GRPO smoke
-> MCPToolsAgentLoop
```

不要切换 RL 框架，不要把 BFCL/ACEBench 当主训练数据，不要从零自建大规模训练语料。

## 必守约束

- 正式 RL 基于 `verl`；`TRL` 只用于 SFT cold-start。
- 训练主源是 Toucan 派生的 replayable `EpisodeSeed`。
- 自建的是 schema perturbation、distractor、verifier、environment 和 eval set。
- 训练脚本不得写死 GPU 数、batch size、micro batch、TP size。
- 项目代码和脚本中的项目文件路径必须以项目根目录为锚点使用相对路径；不要写死 `/data/...`、`/mnt/...` 等机器绝对路径。
- 训练超参必须支持通过脚本命令行参数、环境变量或 Hydra override 注入。
- `data.max_prompt_length` 不得低于 `10240`。
- Ray 临时目录必须使用短路径（默认 `/tmp/ssgrpo_ray`），避免 AF_UNIX socket path 超过 107 bytes。
- `_parse_bfcl_native_args` 必须保持 bounded linear parser。
- 不确定的事实先核验或停下来对齐，不把假设写进实现。

## 常用命令

```bash
# SFT cold-start（8×L20 44GB）
torchrun --nproc_per_node=8 scripts/sft_cold_start.py --config configs/sft_cold_start_4b.yaml

# GRPO smoke test
bash scripts/run_grpo_smoke.sh --config configs/grpo_smoke.yaml

# 正式 GRPO 训练
WANDB_ENTITY=liuzyyy-beihang-university bash scripts/run_grpo.sh --config configs/grpo_train.yaml

# 单元测试
python -m pytest tests/

# 轻量检查
python -m compileall src scripts tests
git diff --check
```

## 验证

文档改动至少跑：

```bash
git diff --check
```

代码改动优先跑：

```bash
python -m pytest tests/
```

只能轻量检查时至少跑：

```bash
python -m compileall src scripts tests
git diff --check
```
