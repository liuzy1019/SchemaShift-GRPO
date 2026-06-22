# data/

本目录下的数据文件不入库（详见根目录 `.gitignore`），clone 后请按以下步骤复现。

## 目录结构

```
data/
├── sft/                        # SFT cold-start 训练数据
│   ├── sft_train.jsonl         # 9146 条样本 (40MB)
│   └── export_stats.json       # 导出统计
├── grpo_train.parquet          # verl GRPO 训练数据（由 prepare_grpo_data.py 生成）
├── grpo_val.parquet            # verl GRPO 验证数据（由 prepare_grpo_data.py 生成）
├── live_mcp/                   # Live MCP 生成任务与 trace（由 live smoke 脚本生成）
├── toucan/                     # Toucan 主数据源
│   └── FALLBACK_DECISION.md    # 数据策略决定
└── toolace/                    # ToolACE 备用数据 (已处理)
```

> 注：大文件（jsonl、json 数据）通过 `.gitignore` 排除，仅保留说明文件。

## 数据复现

```bash
# 1. ToolACE 数据
python scripts/download_toolace.py
python scripts/convert_toolace.py

# 2. Toucan 数据 (需要 HF 镜像)
python scripts/download_toucan.py
python scripts/inspect_toucan.py

# 3. SFT 样本导出 (依赖 episode seeds)
# 由 src/data/sft_step_exporter.py 从 episode_seed 导出

# 4. GRPO parquet 导出
python scripts/prepare_grpo_data.py \
  --episode_seeds data/toucan/episode_seeds.jsonl \
  --output data/grpo_train.parquet \
  --val_output data/grpo_val.parquet

# 5. Live MCP 任务与 smoke trace（显式 live 后端）
python scripts/generate_live_mcp_tasks.py \
  --suite configs/live_mcp/suite_mvp.yaml \
  --output data/live_mcp/tasks/live_mcp_mvp.jsonl \
  --num-tasks 20 \
  --seed 42
```

## 设计原则

- **训练-评测分离**：训练主源 Toucan（ToolACE 仅用于 verifier 单测和 ablation），评测用 BFCL V3 + ACEBench
- **Oracle-Preserving**：Schema 扰动后 ground truth 通过 name_map/enum_map 可还原
- **SFT 仅对齐格式**：SFT 样本从 episode_seed 可见上下文导出，不暴露 oracle_trace
