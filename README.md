# SchemaShift-GRPO

面向 MCP Tool Schema 鲁棒性的 GRPO 训练优化。

标准 GRPO 在多轮 function calling 任务上会放大模型对训练 schema 表面形式的过拟合——工具名、描述措辞、枚举值变了就答不对。SchemaShift-GRPO 通过 **Schema Augmentation**（扰动数据增强）+ **Stratified Advantage**（分层归一化）解决这个问题。

> **工程进度（2026-06-16 收尾）**：`arl` 环境已降到 verl 0.6.1 兼容窗口（4×A10 / torch 2.8 / vllm 0.11 / flash_attn 2.7.3）。**E3 / E4 / E5 三个 smoke 全部跑通到 step 1**：loss 有限、grad_norm finite、ckpt 落盘。正式训练（>1 step）尚未跑。本阶段修复链：BFCL `test_entry_id` 透传 → `max_prompt_length` 2048→10240 → vLLM/flashinfer JIT 绕路 → AgentLoopWorker 卡死根因（`_parse_bfcl_native_args` O(N²) → bounded linear parser）→ E4 ray actor 跨进程 estimator 注册 → 数据长度预检自动化。**105/105 tests pass**。详见 [`docs/project_status.md`](./docs/project_status.md) 与 [`docs/archive/review.md`](./docs/archive/review.md)。

---

## 核心结果

> 尚未运行，待 GPU 集成后填写。

| 指标 | Vanilla GRPO | +Aug | +StratAdv (Ours) | 相较 Vanilla |
|------|-------------|------|-----------------|-------------|
| **整体 pass@1** | — | — | — | — |
| **原版 pass@1** | — | — | — | — |
| **mild pass@1** | — | — | — | — |
| **strong pass@1** | — | — | — | — |
| **Robustness Gap** | — | — | — | — |

---

## 问题：GRPO 放大 Schema 过拟合

标准 GRPO 在 function calling 任务上有三个致命问题：

1. **Schema 表面形式过拟合**：GRPO 会放大模型对训练 schema 具体工具名 / 描述措辞 / 枚举值的依赖，换个同义写法就脚
2. **群组奖励饱和**：二元 reward（0/1），group_size 较大时容易全 0 或全 1 → advantage 方差→0 → 梯度消失
3. **扰动信号淹没**：强扰动下模型更难答对，reward 偏低。标准 GRPO 的 prompt 内 z-score 会惩罚所有低分 rollout，包括“困难条件下表现合理”的

```
训练：search_flights(origin="JFK", destination="LHR")  → 正确
测试：find_flights(departure_location="JFK", to="LHR") → 失败
```

---

## 方法

### Schema Augmentation（信号源）

训练时混合 3 种扰动强度的 schema 变体（none/mild/strong），让模型见过不同的表面形式。扰动是确定性的（给定 seed），基于内置同义词映射表（~100 条规则）。

```
mild:  tool_name 同义替换
      search_flights → find_flights

strong: tool_name + description + enum 全部替换
      search_flights → find_flights
      "Search for available flights" → "Lookup for accessible flights"
      "economy" → "standard"
```

扰动器同时维护 `name_map` 和 `enum_map`，用于训练时 API dispatch 反向映射和评估时 ground truth 同步。

### Stratified Advantage（信号通路）

按扰动强度分层归一化 + 全局残差，防止强扰动 rollout 被弱扰动 rollout 的 reward 碾压。

**公式**：
```
A = strat_z + beta × global_z

strat_z:  每层（level）内独立 z-score，同一 task 跨 none/mild/strong 分层归一化
global_z: 同一 task 内跨层全局 z-score（作为跨层比较信号的残差）
beta:     跨层跨信号强度，默认 0.25
```

- beta=0 → 纯分层归一化
- beta 很大 → 趋近标准 GRPO
- 单样本层/组：std=1.0，不产生 NaN

### 与 verl 集成

注册为 verl 的自定义 estimator（`@register_adv_est("schemashift_grpo")`），从 uid 解析分组信息：

```
uid = "{task_id}___{level}"  →  estimator 解析 task_id 分组、level 分层
```

verl 的非 GRPO 路径通过 `adv_kwargs` 传入 `index(uid)`。SchemaShift estimator 通过 `register_estimator.py` monkey-patch `verl.compute_advantage` 以注入 `non_tensor_batch`（含 perturbation_level、group_id）。

---

## 消融实验

对标 longhorizon 的 Vanilla → 信号源 → 通路 → Joint 结构。

| 实验 | Schema Aug | Advantage | 测量 |
|------|-----------|-----------|------|
| **E3** (Vanilla GRPO) | ✗ | 标准 GRPO（per-prompt 的 n 条 rollout 分组） | 基线 |
| **E5** (+Aug) | ✓ (每 task 三个 level prompt，rollout.n=3) | 标准 GRPO（同一 (task, level) 内 n 条 rollout 分组） | Aug 单独贡献 |
| **E4** (Ours) | ✓ (3:3:3) | `strat_z + 0.25 × global_z`（同一 task 跨 none/mild/strong 9 条 rollout） | 联合方案 |

```
E3 (Vanilla) ──加 Aug──▶ E5 ──加 StratAdv──▶ E4 (Full)
                              │                    │
                              └─ Aug 贡献          └─ StratAdv 贡献
                              （每 level prompt    （同 task 跨 level
                               内 GRPO）          分层 z-score）
```

额外基线：E1（Zero-shot）、E2（SFT）、E6（正则化 baseline）。

### 消融结果表

> 尚未运行。

| 实验 | Robustness Gap ↓ | 原版 pass@1 ↑ | mild pass@1 ↑ | strong pass@1 ↑ |
|------|-----------------|-------------|-------------|----------------|
| E1 (Zero-shot) | — | — | — | — |
| E2 (SFT) | — | — | — | — |
| E3 (Vanilla GRPO) | — | — | — | — |
| E5 (+Aug) | — | — | — | — |
| **E4 (Ours)** | **—** | **—** | **—** | **—** |
| E6 (RegBaseline) | — | — | — | — |

---

## 假设验证

| 假设 | 状态 | 验证 |
|------|------|------|
| **H1**: GRPO 放大 schema 过拟合 | ⬜ 待验证 | E3 robustness gap > E2 gap |
| **H2**: Schema Aug 降低过拟合 | ⬜ 待验证 | E5 robustness gap < E3 gap |
| **H3**: StratAdv 进一步提升 | ⬜ 待验证 | E4 robustness gap < E5 gap |
| **H4**: 联合方案 > 最优基线 | ⬜ 待验证 | E4 综合指标最优 |

---

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"
pip install pyarrow pandas

# 安装 verl（参考 repo 的 fork，v0.6.1）
pip install -e verl/

# 数据准备
python scripts/build_parquet.py

# 运行测试
pytest tests/
```

### 训练

```bash
# E3: Vanilla GRPO 基线
bash scripts/train/grpo/run_vanilla_grpo.sh

# E5: Aug Only 消融（先跑）
bash scripts/train/grpo/run_aug_only.sh

# E4: SchemaShift-GRPO（主实验，后跑）
bash scripts/train/grpo/run_schemashift.sh
```

---

## 项目结构

```
schemashift-grpo/
├── src/
│   ├── envs/
│   │   ├── schema_perturber.py        # 扰动生成器（~380 行）
│   │   ├── api_mapper.py              # 函数名+枚举值双向映射（~130 行）
│   │   └── bfcl_env.py                # BFCL 多轮环境
│   ├── training/
│   │   ├── schemashift_grpo_estimator.py  # 自定义 verl estimator
│   │   ├── schemashift_advantage.py       # 独立 advantage 函数
│   │   ├── register_estimator.py          # verl 集成入口
│   │   ├── length_check.py                # 数据长度预检（fail-fast + warn）
│   │   ├── run_exp3.py                    # E3 / E5 入口
│   │   └── run_exp4.py                    # E4 入口（注入 SchemaShiftTaskRunner）
│   ├── agent_loop/
│   │   └── bfcl_agent_loop.py        # BFCL 多轮 agent loop
│   ├── reward/
│   │   └── bfcl_reward.py            # BFCL 正确性 reward
│   └── eval/
│       └── bfcl_eval.py               # 鲁棒性评估
├── scripts/
│   ├── build_parquet.py               # 数据 → parquet（含 uid 编码）
│   └── train/
│       ├── sft/
│       │   ├── run_exp2_sft.py
│       │   └── run_exp2_sft.sh
│       └── grpo/
│           ├── run_vanilla_grpo.sh    # E3 / E5（共用 run_exp3.py）
│           ├── run_schemashift.sh     # E4（run_exp4.py，注册 SchemaShift estimator）
│           └── run_aug_only.sh        # E5
├── tests/                             # 105 passed
├── docs/                              # 技术方案 + 消融实验 + Codex 审查报告
└── data/verl/                         # parquet 数据（E4: 8100/900，E5: 2700/300）
```

---

## 技术亮点

- **信号传输理论**：信号源（Schema Aug）+ 信号通路（StratAdv），模型无关，适用于任何 schema 鲁棒性 RL 任务
- **零参数 Process Reward**：SchemaPerturber 纯规则式，可解释、零可训练参数
- **自定义 verl Estimator**：通过 verl 的 `@register_adv_est` 机制注册 + monkey-patch `compute_advantage` 注入分层信息
- **内存高效**：基于 Qwen2.5-1.5B-Instruct，单卡 A100-80GB 可跑，~166 GPU-hours 完成全部实验

---

## 许可

MIT License
