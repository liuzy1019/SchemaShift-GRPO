# LiveMCP-GRPO

> State-Machine Data Synthesis + Constrained GRPO for Multi-Step MCP Tool Use

PROVE-style state-machine teacher + live MCP execution + event-sourced reward + stratified advantage, training models on 10 domains / 188 tools.

**当前方案：**
- **数据生成**：PROVE state-machine (LLM-in-the-loop, Qwen3-32B-Instruct) → `scripts/generate_data.py` → parquet
- **Policy 模型**：Qwen3-4B
- **奖励**：`J = R_task + I_shape·λ_shape·F_gamma + I_process·λ_process·P_process − λ_safe·C_safety`
- **Advantage**：2D 分层 (perturbation_level × scenario_type) + LATA + 饱和跳过
- **硬件**：8×L20 44GB

---

## 项目结构

```
├── scripts/
│   ├── generate_data.py             # 数据生成 CLI (PROVE state-machine)
│   ├── generate_data.sh             # 统一生成脚本 (自动检测模型大小+GPU拓扑)
│   └── train_grpo.sh                # GRPO 训练入口 (GPU 自适应)
│
├── src/
│   ├── live_mcp/                    # MCP 环境 + 数据生成
│   │   ├── task_planner.py          #   PROVE state-machine (LLM-in-the-loop)
│   │   ├── orchestrator.py          #   任务编排 + 扰动 + 缺量守卫
│   │   ├── api.py                   #   统一 API
│   │   ├── llm_client.py            #   LLM 客户端 (vLLM / local)
│   │   ├── dedup.py                 #   Jaccard 去重 (0.70, 位置感知)
│   │   ├── state_seeder.py          #   确定性状态播种 (10 domains)
│   │   └── servers/ × 10           #   MCP 子进程服务器
│   │
│   ├── agent_loop/                  # verl Agent Loop
│   │   ├── livemcp_oval_loop.py     #   注册名 "livemcp_oval"
│   │   └── oval_mcp_worker.py       #   session + audit wrapper
│   │
│   ├── oval_mcp/                    # 奖励 + 约束 GRPO 算法
│   │   ├── rewards/                 #   R_task + F_gamma + P_process
│   │   ├── verifier/                #   C_safety + AuditEvent
│   │   ├── envs/                    #   DomainAdapter + AuditWrapper
│   │   └── training/                #   LambdaState + LATA + Saturation
│   │
│   ├── reward/                      # verl 奖励入口
│   │   └── oval_reward_fn.py        #   compute_score()
│   │
│   └── training/                    # verl 训练组件
│       ├── livemcp_grpo_estimator.py  # 2D StratAdv + LATA
│       └── register_estimator.py    #   estimator 注册 + λ 更新
│
├── configs/
│   ├── agent_loop.yaml              # Agent loop 注册
│   └── live_mcp/ × 10              # 10 domain 子进程配置
│
├── tests/
├── data/                            # 训练数据 (gitignored)
└── outputs/                         # 输出 (gitignored)
```

---

## 快速开始

### 1. 启动 vLLM Teacher

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen/Qwen3-32B-Instruct \
  --served-model-name Qwen3-32B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

### 2. 生成训练数据

```bash
# 全量生成（500 train + 100 val，所有 domain）
python scripts/generate_data.py \
  --model Qwen3-32B-Instruct \
  --api-base http://localhost:8000/v1 \
  --count 500 --val-count 100

# 单 domain 快速测试
python scripts/generate_data.py \
  --model Qwen3-32B-Instruct \
  --api-base http://localhost:8000/v1 \
  --domain calendar --count 5 --val-count 2
```

### 3. 启动 GRPO 训练

```bash
bash scripts/train_grpo.sh
```

---

## 管线说明

### 数据生成 (PROVE state-machine)

```
Per domain:
  1. Auto-discover tool dependency chains (DFS, length 2-5)
  2. LLM-in-the-loop: query generation → tool execution → continuation decisions
     (teacher sees live state, decides next action each turn)
  3. PROVE robustness: intermittent errors, paginated responses, partial batch failures
  4. Replay-validate each conversation against live MCP
  5. Derive success_criteria from state delta (state_equals/state_exists)
  6. Position-aware Jaccard dedup (0.70)
  7. Convert to verl-compatible parquet

Default difficulty mix: complete 60% / missing 20% / minimal 20%
```

### GRPO 训练

```
parquet → vLLM rollout → LiveMCPOvalLoop (真实执行 + 审计)
  → Reward: R_task + F_gamma + P_process − λ_safe·C_safety
  → StratAdv (2D stratified advantage) + LATA + 饱和跳过
  → FSDP gradient update
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [REVIEW.md](REVIEW.md) | 审查报告 + 修复记录 + 对抗式审查结果 |
| [CLAUDE.md](CLAUDE.md) | 工作入口 + 管线状态 |
| [AGENTS.md](AGENTS.md) | AI 协作约定 + 环境事实 |
| [data/README.md](data/README.md) | 数据目录规范 + 实验记录规范 |

---

## 许可

MIT License.
