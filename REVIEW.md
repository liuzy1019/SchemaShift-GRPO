# LiveMCP-GRPO 项目审查报告

> 最后更新：2026-06-27（第十四轮：PROVE 对齐审计 + 全项目验证）
> 当前状态：PROVE 管线已对齐，待生成训练数据

---

## 1. PROVE 参数对齐（全部通过 ✅）

| 参数 | PROVE | 当前实现 | 文件 |
|------|-------|---------|------|
| Jaccard dedup | 0.70 | 0.70 | `src/live_mcp/dedup.py` |
| 难度分层 | complete 60% / missing 20% / minimal 20% | 一致 | `scripts/generate_data.py` L100 |
| Enum stripping | 30% | `random() < 0.30` | `task_planner.py` L144 |
| Irrelevance | 5% | `irrelevance_ratio=0.05` | `scripts/generate_data.py` L64 |
| Reward weights | w_val=0.5, w_cov=0.5, w_eff=0.15, w_name=0.2, w_arg=0.1 | 一致 | `task_reward.py` L18-26 |
| λ_shape / λ_process | 0.5 / 0.3 | 一致 | `oval_reward_fn.py` L42-43 |
| 扰动总概率 | 0.15–0.30 per tool call | ~0.20（每类 0.10） | `task_planner.py` L408-480 |
| 扰动类型 × 4 | intermittent, paginated, incomplete, partial_batch | ✅ 全部实现 | `task_planner.py` |
| Domain 扰动适配 | 按 domain 分组 | ✅ 7组映射×10 domain | `task_planner.py` |
| Two-phase 生成 | Phase1 选工具名 + Phase2 真实执行 | ✅ 默认模式 | `orchestrator.py` |
| Replay validation | 可执行性验证 | ✅ | `orchestrator.py` L170 |

唯一偏离（硬件限制）：
- Domain: 10 vs PROVE 20 — 精选 Table 2 核心子集
- Teacher: Qwen3-8B vs Gemma-4-31B-it — A10 24GB 限制，8B 已验证可用

---

## 2. 项目方案架构

```
PROVE pipeline (default):
  LLM Teacher (Qwen3-8B) → plan_task() → [tool_seq only] → execute_plan()
       ↑                                              ↓
   LLMClient                                 真实 MCP 执行 + infer_args
  (local/vLLM)                               oracle trace → replay_validate
                                                     ↓
                                             parquet → GRPO training

E2E pipeline (legacy, --teacher e2e):
  LLM Teacher → orchestrator → replay validate → Jaccard dedup(0.70) → parquet → GRPO
```

---

## 3. 第十四轮修复记录（2026-06-27）

| # | 问题 | 文件 | 修复 |
|---|------|------|------|
| 1 | 扰动只实现 2/4 种类型 | `task_planner.py` | 补全 `incomplete_intermediate` + `partial_batch_failure` + domain 分组 |
| 2 | 扰动无 domain 适配 | `task_planner.py` | 添加 7 组 domain 映射表 |
| 3 | `suite_mvp.yaml` 权重与代码不一致 | `suite_mvp.yaml` | 更新为 PROVE 对齐值 |
| 4 | 文档引用 8+ 个不存在文件 | `REVIEW.md` `CLAUDE.md` | 全部修正 |

---

## 4. 模块清单

### 数据生成
| 模块 | 文件 | 状态 |
|------|------|------|
| PROVE Teacher | `src/live_mcp/task_planner.py` | ✅ |
| LLM Client | `src/live_mcp/llm_client.py` | ✅ local/vLLM 双模式 |
| Orchestrator | `src/live_mcp/orchestrator.py` | ✅ |
| Dedup | `src/live_mcp/dedup.py` | ✅ Jaccard 0.70 |
| 10 Domain servers | `src/live_mcp/servers/{domain}/` | ✅ |
| 数据生成入口 | `scripts/generate_data.py` | ✅ |

### 训练
| 模块 | 文件 | 状态 |
|------|------|------|
| GRPO 训练入口 | `scripts/train_grpo.sh` | ✅ GPU 自适应 |
| OVAL rollout | `src/training/run_grpo.py` | ✅ |
| GRPO Estimator | `src/training/livemcp_grpo_estimator.py` | ✅ 饱和检测+降级 |

### 奖励
| 模块 | 文件 | 状态 |
|------|------|------|
| Reward fn | `src/reward/oval_reward_fn.py` | ✅ R_task + C_safety + F_gamma + P_process |
| Task Reward | `src/oval_mcp/rewards/task_reward.py` | ✅ |
| Safety Verifier | `src/oval_mcp/verifier/safety.py` | ✅ |
| Lambda State | `src/oval_mcp/training/lambda_state.py` | ✅ 自适应 λ |
| Audit Wrapper | `src/oval_mcp/envs/audit_wrapper.py` | ✅ |

### 测试
| 测试集 | 文件 |
|--------|------|
| 10 Domain 集成 | `tests/test_live_mcp_10_domains.py` |
| OVAL 组件 | `tests/test_oval_mcp_components.py` |
| OVAL 场景 | `tests/test_oval_mcp_scenarios.py` |
| GRPO Estimator | `tests/test_livemcp_grpo_estimator.py` |
| Smoke Import | `tests/test_smoke_grpo_imports.py` |

---

## 6. 第十五轮验证（2026-06-27）：全管线端到端测试

### 测试结果

| 阶段 | 命令 | 结果 |
|------|------|------|
| 编译检查 | `python3 -m compileall src/ scripts/ tests/` | ✅ 全部通过 |
| Import 烟雾 | `pytest tests/test_smoke_grpo_imports.py` | ✅ 33 passed |
| OVAL 组件 | `pytest tests/test_oval_mcp_components.py` | ✅ |
| OVAL 场景 | `pytest tests/test_oval_mcp_scenarios.py` | ✅ |
| 10 Domain 集成 | `pytest tests/test_live_mcp_10_domains.py` | ✅ |
| GRPO Estimator | `pytest tests/test_livemcp_grpo_estimator.py` | ✅ |
| **8B 数据生成** | `CUDA_VISIBLE_DEVICES=1 python3 scripts/generate_data.py --count 2 --val-count 1 --domain calendar --model models/Qwen3-8B` | **✅ 2/2 train + 1/1 val，100% yield** |

### 产出的 Parquet 结构

| 字段 | 值 |
|------|-----|
| `data_source` | `live_mcp_state_machine` |
| `extra_info.domain` | `calendar` |
| `extra_info.required_tools` | 如 `['add_attendee', 'list_events']` |
| `scenario_type` | `task_planner` |
| `perturbation_level` | `complete` / `minimal` |

### 已知限制 & 运维问题

| 问题 | 根因 | 解决方案 | 严重度 |
|------|------|---------|--------|
| 模型路径必须相对于项目根目录 | HuggingFace pipeline 不接受绝对路径 | 使用相对路径 `models/Qwen3-8B` | 🟡 |
| GPU0 常被其他进程占满 | 共享机器资源竞争 | `--gpus 1,2,3,...` 或 `GPU_FREE_ONLY=1` 自动跳过繁忙卡 | 🟢 已解决 |
| train/val 各自独立加载一次模型 | `generate_tasks_llm` 每次新建 LLMClient | 重构为单次加载复用 pipeline（优化项，8B 加载仅 ~4s） | 🟢 |
| **32B 本地 transformers 加载失败** | 4×A10 (92GB) 不够 Qwen3-32B BF16 + activations → CPU offload | **32B 必须用 vLLM 部署**，`--api-base http://...` 模式 | 🔴 |
| 数据并行生成时 model 加载两次（train+val） | `generate_data.py` 分两次调用，各自加载 | 低优先级优化，不影响正确性 | 🟢 |
### 8B vs 32B 数据生成耗时对比

| 模型 | GPU | 模型加载 | 单条 train | 产出 |
|------|-----|---------|-----------|------|
| Qwen3-8B | 1×A10 | ~4s | ~8s | ✅ 2/2 |
| Qwen3-32B (local) | 4×A10 | ~12s → CPU offload | >5min → 超时 | ❌ 0/2 |
| Qwen3-32B (vLLM) | 4×A10 TP4 | — | ~3s (估计) | ⏳ 未测试 |

结论：32B 数据生成必须走 vLLM 部署路线，本地 `device_map="auto"` 对 >8B 模型不可靠。

---

## 5. 待办

| 优先级 | 操作 | 说明 | 状态 |
|--------|------|------|------|
| P0 | 生成正式训练数据 | `python scripts/generate_data.py --count 500 --model models/Qwen3-8B` | ⏳ |
| P0 | GRPO 正式训练 | `bash scripts/train_grpo.sh`，被数据生成阻塞 | ⏳ |
| P3 | Distractor 频率实验 | 验证 20%+20% vs 40% | ❌ |
| P3 | L20 超参 sweep | RESPONSE_LENGTH=16384, ROLLOUT_N=9 | ❌ |

### 已知低优先遗留

- `audit_step` / `audit_step_with_state` ~100 行重复 — 重构成本高，功能正确
- `SaturationDiagnostics` 未用 — 保留供诊断
- `register_estimator.py` verl monkey-patch — 已有 fail-fast 保护

---

## 6. 多卡架构

### 配置入口

| 脚本 | GPU 控制方式 | 默认行为 |
|------|------------|---------|
| `scripts/gpu_config.sh` | `source scripts/gpu_config.sh [gpu_ids]` | 检测所有 GPU |
| `scripts/train_grpo.sh` | `--gpus 0,1,2,3` 或 `CUDA_VISIBLE_DEVICES` 或 `GPU_COUNT=4` | 所有 GPU，模型路径透传 |
| `scripts/generate_data_parallel.sh` | 同上，额外自动跳过繁忙 GPU（`GPU_FREE_ONLY=1`） | 所有空闲 GPU |

### 训练侧（DDP/FSDP）

verl 原生支持 FSDP。`train_grpo.sh` 将 `GPU_COUNT` 动态注入：
- `trainer.n_gpus_per_node=${GPU_COUNT}` — verl 分布式训练
- `actor_rollout_ref.rollout.agent.num_workers=${GPU_COUNT}` — agent workers
- `actor_rollout_ref.actor.strategy=fsdp` — FSDP 分片
- vLLM rollout 的 `tensor_model_parallel_size` 可独立配置

GPU tier 自动检测 → 注入 per-tier 超参（batch_size, response_length 等），所有参数支持 `OVAL_*` 环境变量覆盖。

### 推理侧（多卡加速）

| 模式 | 机制 | 适用场景 | 脚本 |
|------|------|---------|------|
| **单卡 local** | `LLMClient(mode="local", device=N)` | 单卡推理 | `generate_data.py --device N` |
| **多卡数据并行** | 每 GPU 独立模型副本，分片生成 | 8B 模型 × N 卡 | `scripts/generate_data_parallel.sh` |
| **vLLM TP** | `actor_rollout_ref.rollout.tensor_model_parallel_size` | 32B+ 大模型推理 | `train_grpo.sh`（vLLM rollout） |
| **vLLM API** | `--api-base http://...` + `mode="openai"` | 外部 vLLM 服务 | `generate_data.py --api-base ...` |

### 繁忙 GPU 自动跳过

`gpu_config.sh` 支持 `GPU_FREE_ONLY=1`，自动跳过显存占用 >20% 的 GPU。

```bash
# 自动用所有空闲 GPU
bash scripts/generate_data_parallel.sh --count 500 --model models/Qwen3-8B

# 指定 GPU + 强制跳过繁忙卡
GPU_FREE_ONLY=1 bash scripts/generate_data_parallel.sh --gpus 0,1,2,3
```

### 已验证的 7-GPU 并行生成

```
[gpu_config] Skipping GPU 0: 19757/23028 MiB (85%)
[gpu_config] 7x NVIDIA A10 (22GB) → tier=A10, ids=1,2,3,4,5,6,7
→ 7 shards × 2 train + 1 val = 14 train + 7 val rows → merged → train.parquet + val.parquet
耗时：~60s（包括模型加载+推理+合并）
