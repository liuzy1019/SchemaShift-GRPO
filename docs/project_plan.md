# MCP Tools RL Project Plan

> SchemaShift-GRPO 权威方案文档。若本文与 `README.md`、`AGENTS.md` 冲突，以本文为准；修改工程路线时先更新本文，再改实现。

## 1. 项目目标

SchemaShift-GRPO 面向 MCP-style tool-use RL。模型不学习 MCP JSON-RPC 协议本身，而是在给定任务、当前工具 schema、可见历史动作和工具返回后，选择下一步动作：

```text
tool_call | final_answer | ask_clarification | report_error
```

核心问题是：模型在 schema 变化、相似工具干扰、多步工具返回上下文中，能否稳定选择正确工具、参数和值，并在应该停止时输出最终答案。

## 2. 已确认路线

### 2.1 训练框架

- 正式 RL 继续基于 `verl`。
- `TRL` 只用于 SFT cold-start，不作为正式 agent RL 框架。
- 不切换到 OpenRLHF，不自研完整 RL 框架；除非后续出现可复现实验证明 `verl` 路线无法满足需求。

当前代码事实：

- `./verl` 作为 editable 源码依赖。
- `src/training/schemashift_grpo_estimator.py` 已实现 `schemashift_grpo` estimator。
- `src/training/register_estimator.py` 已处理 `non_tensor_batch` metadata 透传。
- `src/training/schemashift_advantage.py` 已实现按 `(perturbation_level, scenario_type)` 分层的 advantage 逻辑。

### 2.2 数据策略

训练主源：

```text
Toucan raw data
  -> EpisodeSeed
  -> schema perturbation / distractor injection / conditioned context
  -> online rollout
  -> replay executor / verifier
  -> GRPO
```

关键边界：

- RL 训练单元是 replayable `episode_seed`，不是静态 `prompt + answer`。
- 不从零自建大规模轨迹语料作为主路线。
- 自建的是环境变体、schema perturbation、distractor injection、verifier 和 robustness eval。
- ToolACE 不参与主训练循环；保留为 verifier 单测、schema perturbation 测试、argument key/value 测试和 ablation。
- BFCL、ACEBench、MCP-Bench 类公开 benchmark 不作为主训练源，避免污染公开评测。

### 2.3 Cold-start

SFT cold-start 目标只限于：

- action format；
- JSON / tag parseability；
- schema-following；
- tool name 与参数的初始对齐；
- conditioned context 下继续决策的初始能力。

当前模型：**Qwen3-4B**（从 Qwen2.5-7B-Instruct 切换，显存更充裕）。

SFT 训练配置：
- 模型：`models/Qwen3-4B`（BF16 ~8GB）
- YAML：`configs/sft_cold_start_4b.yaml`
- 输出：`outputs/sft_cold_start_4b/final`
- 硬件：8×L20 44GB + DeepSpeed ZeRO-2
- effective_batch=128（per_device=4 × grad_accum=4 × 8GPU）
- 3 epochs → ~204 total steps

## 3. 当前实现状态

| 模块 | 状态 | 文件 |
|---|---|---|
| Schema perturbation | 已实现 | `src/envs/schema_perturber.py` |
| API mapping | 已实现 | `src/envs/api_mapper.py` |
| Episode schema | 已实现 | `src/data/episode_schema.py` |
| Episode seed builder | 已实现 | `src/data/episode_seed_builder.py` |
| SFT step exporter | 已实现 | `src/data/sft_step_exporter.py` |
| Distractor sampler | 已实现 | `src/data/distractor_sampler.py` |
| Conditioned builder | 已实现 | `src/data/conditioned_builder.py` |
| Action parser | 已实现 | `src/reward/action_parser.py` |
| Component reward | 已实现 | `src/reward/component_reward.py` |
| Matching helpers | 已实现 | `src/eval/matching.py` |
| SchemaShift GRPO estimator | 已实现 | `src/training/schemashift_grpo_estimator.py` |
| Stratified advantage | 已实现 | `src/training/schemashift_advantage.py` |
| Estimator registration | 已实现 | `src/training/register_estimator.py` |
| Length precheck | 已实现 | `src/training/length_check.py` |
| ReplayMCPExecutor | 已实现 | `src/envs/replay_mcp_executor.py` |
| MCPToolEnvironment | 已实现 | `src/envs/mcp_tool_environment.py` |
| TrajectoryVerifier | 已实现 | `src/reward/trajectory_verifier.py` |
| SchemaShift reward_fn | 已实现 | `src/reward/schemashift_reward_fn.py` |
| SchemaShiftTaskRunner | 已实现 | `src/training/schemashift_task_runner.py` |
| GRPO smoke 脚本 | 已实现 | `scripts/run_grpo_smoke.sh` + `scripts/train_grpo.py` |
| GRPO YAML configs | 已实现 | `configs/grpo_smoke.yaml` + `configs/grpo_train.yaml` |
| 数据准备脚本 | 已实现 | `scripts/prepare_grpo_data.py` |
| Live MCP branch | 已实现为并行分支 | `src/live_mcp/` + `scripts/generate_live_mcp_tasks.py` + `scripts/run_live_mcp_smoke.py` |
| Default GRPO MCPToolsAgentLoop | 未接入 | `src/agent_loop/`（默认 replay GRPO 多轮 rollout） |
| Live MCP offline AgentLoop | 已实现为并行分支 | `src/live_mcp/agent_loop.py` |
| strict replay smoke | 待验证 | 需要重跑 |
| self MCP robustness eval | 未开始 | 待新增 |

## 4. Reward 与 advantage

Tool-call partial reward：

```text
0.10 format_valid
+ 0.15 schema_valid
+ 0.30 tool_selection
+ 0.20 argument_keys
+ 0.25 argument_values
```

Correctness floor：

```text
effective_reward =
  1.0 + 0.05 * partial_reward    if exact_success = 1
  0.3 * partial_reward           if exact_success = 0
```

StratAdv：

```text
A = strat_z + 0.25 * global_z
stratum = (perturbation_level, scenario_type)
```

Metadata 必须随 rollout 进入 `non_tensor_batch`，至少包括：

```text
episode_id
group_id
perturbation_level
scenario_type
action_type
tool_name
```

## 5. 评测策略

评测分三层，不混用训练与公开 benchmark。

### 5.1 Self MCP Robustness Set

项目核心评测，必须覆盖：

- original schema vs perturbed schema；
- distractor@k；
- `call_only` / `call_then_final` / `call_then_call` / `no_tool`；
- single-tool / parallel-tool；
- seen server / unseen server；
- enum remapping；
- conditioned tool output 后继续调用或停止。

### 5.2 BFCL

公开 function-calling 横向评测。优先接 BFCL V3；后续可扩展 BFCL V4。BFCL adapter 应转成项目统一 action format，不反向修改核心 reward 去适配 benchmark。

### 5.3 ACEBench / MCP 类 benchmark

ACEBench 可作为 agent/tool-use 补充评测。MCP-Bench、MCP-Universe、MCP-AgentBench 等当前仓库尚无 adapter 或 runner，只作为后续扩展候选。

## 6. 下一阶段开发顺序

第一阶段只验证最小 RL smoke，不追求 reward 正值：

```text
strict replay smoke
-> verl GRPO smoke
-> MCPToolsAgentLoop
```

验收标准：

- step 1 完成；
- loss finite；
- 无 traceback；
- reward 能回传；
- `perturbation_level`、`scenario_type`、`group_id` 能进入 `schemashift_grpo`。
- Ray 使用短临时目录（默认 `/tmp/ssgrpo_ray`），避免 plasma store AF_UNIX socket path 超过 107 bytes。

并行新增 Live MCP MVP，但默认训练后端仍为 replay：

```text
Live MCP framework skeleton
-> calendar / shopping subprocess stdio servers
-> grounded LiveTask synthesis + oracle validation
-> offline MCPToolsAgentLoop smoke + execution-aware reward
-> later SchemaShiftTaskRunner integration
```

Live MCP 第一版边界：

- 保留 `ReplayMCPExecutor`、`MCPToolEnvironment`、`schemashift_reward_fn.py` 的现有 next-action 路线。
- 新增 live 路线只通过 `src.live_mcp.api.LiveMCPBranch` 或新 smoke 脚本显式启用。
- smoke 默认必须走 subprocess stdio transport；in-process transport 只用于单测或显式 debug。
- 第一版实现 calendar / shopping 两个 stateful server，覆盖 live execution、session isolation、deterministic reset、dependency graph、grounded query、oracle validation、trace recording 和五组件 reward。
- 不在第一版默认启用 live GRPO，不接真实公网 API，不依赖外部 LLM teacher。
- `configs/live_mcp/suite_mvp.yaml` 默认 `environment.backend: replay`，避免误接默认训练链路。

## 7. 工程约束

- 训练脚本不得写死 GPU 数、batch size、micro batch、TP size；必须按 `N_GPUS` 或实际资源自适应。
- 项目文件路径必须以项目根目录为锚点使用相对路径；禁止把 `/data/...`、`/mnt/...` 等机器路径写入默认配置。
- 训练超参必须支持脚本命令行参数、环境变量或 Hydra override 注入。
- `data.max_prompt_length` 不得低于 `10240`。
- `_parse_bfcl_native_args` 如涉及修改，必须保持 bounded linear parser，不得回退到 outer loop 内重复 `find` 的 O(N²) 写法。
- `flashinfer` JIT 默认禁用：`VLLM_USE_FLASHINFER_SAMPLER=0`，`VLLM_ATTENTION_BACKEND=FLASH_ATTN`。
- 大改动前先更新本文档，再进入实现。

## 8. 待核验事实

- `docs/` 目录当前不存在；后续若新增外部接入、评测 runbook 或架构图，再创建对应目录与文档。
- Qwen3-4B SFT 尚未完成，需要在 L20 机器上重新跑。
