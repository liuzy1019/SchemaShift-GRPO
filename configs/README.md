# configs/

训练和环境配置文件。所有 YAML 参数均带行内注释说明用途和约束。

## 训练路线与配置对应关系

| 路线 | 配置文件 | 启动脚本 | 状态 |
|------|----------|----------|------|
| OVAL GRPO | `grpo_direct.yaml`（+ 环境变量） | `bash scripts/train_grpo.sh` | ✅ 主路线 |
| Direct GRPO | `grpo_direct.yaml` | `python scripts/train_grpo.py --config configs/grpo_direct.yaml` | Legacy |
| Cold GRPO | `grpo_cold.yaml` | `python scripts/train_grpo.py --config configs/grpo_cold.yaml` | Legacy |

## 文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `grpo_direct.yaml` | GRPO 训练（OVAL MCP + StratAdv） | ✅ |
| `grpo_cold.yaml` | SFT 冷启动 → GRPO | ✅ |
| `grpo_smoke.yaml` | GRPO smoke test（2 步验证链路） | ✅ |
| `agent_loop.yaml` | Agent loop 注册 | ✅ |
| `ds_zero2.json` | DeepSpeed ZeRO-2（JSON 格式） | ✅ |
| `live_mcp/` | 10 domain 子进程配置（suite_mvp.yaml 等） | ✅ |

## 正式训练核心参数

配置文件：[grpo_direct.yaml](grpo_direct.yaml)

| YAML 路径 | 当前值 | 说明 |
|-----------|--------|------|
| `algorithm.livemcp.beta` | 0.25 | StratAdv 分层权重 |
| `data.max_prompt_length` | 10240 | 最大 prompt 长度 |
| `data.max_response_length` | 4096 | 最大 response 长度 |
| `rollout.group_size` | 1 | 数据侧每组条数 |
| `rollout.max_turns` | 5 | 交互式最大轮次 |
| `trainer.total_training_steps` | 300 | 训练步数 |
| `actor.ppo_micro_batch_size_per_gpu` | 3 | 每卡 micro batch |

可通过环境变量覆盖：`N_GPUS`、`BETA`、`TOTAL_STEPS`、`SAVE_FREQ`、`TEST_FREQ`

正式入口也支持 `--config PATH` 与位置参数形式的 Hydra overrides，例如：

```bash
bash scripts/run_grpo.sh --config configs/grpo_direct.yaml trainer.total_training_steps=10
```

## 环境变量覆盖

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `OVAL_I_SHAPE` | 启用 F_gamma shaping | 0 |
| `OVAL_I_PROCESS` | 启用 P_process scoring | 1 |
| `OVAL_LAMBDA_SHAPE` | λ_shape 权重 | 0.5 |
| `OVAL_LAMBDA_PROCESS` | λ_process 权重 | 0.3 |
| `OVAL_GAMMA` | F_gamma 衰减因子 | 1.0 |
| `OVAL_DOMAINS` | Oval loop domain 列表 | 全部 10 个 |
| `OVAL_SUITE_PATH` | Suite 配置路径 | configs/live_mcp/suite_mvp.yaml |

## 注意

- `ds_zero2.json` 保持 JSON 格式（DeepSpeed 不支持 YAML）
- SFT cold-start 相关配置已从主训练路线移除
- 所有路径使用项目根目录相对路径，禁止写死机器绝对路径
- 本目录只描述配置事实；正式训练是否完成以 checkpoints、训练日志和 GPU 环境复验结果为准
