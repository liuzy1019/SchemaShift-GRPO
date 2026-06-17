# SchemaShift-GRPO

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![verl 0.6.1](https://img.shields.io/badge/verl-0.6.1-orange.svg)](https://github.com/volcengine/verl)
[![Tests 105 passed](https://img.shields.io/badge/tests-105%20passed-brightgreen.svg)](./tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#-许可)

> **面向 MCP Tool Schema 鲁棒性的 GRPO 训练优化**
> 标准 GRPO 在多轮 function calling 任务上会放大模型对训练 schema 表面形式的过拟合 —— 工具名、描述措辞、枚举值变了就答不对。**SchemaShift-GRPO** 通过 *Schema Augmentation*（信号源）+ *Stratified Advantage*（信号通路）的联合设计，专门针对 schema 鲁棒性场景构建抗过拟合训练通路。

> **🛠️ 工程进度（2026-06-16 收尾）**：`arl` 环境已降到 verl 0.6.1 兼容窗口（4×A10 / torch 2.8 / vllm 0.11 / flash_attn 2.7.3）。**E3 / E4 / E5 三个 smoke 全部跑通到 step 1**：loss 有限、grad_norm finite、ckpt 落盘。正式训练（>1 step）尚未开始。本阶段修复链：BFCL `test_entry_id` 透传 → `max_prompt_length` 2048→10240 → vLLM/flashinfer JIT 绕路 → AgentLoopWorker 卡死根因（`_parse_bfcl_native_args` O(N²) → bounded linear parser）→ E4 ray actor 跨进程 estimator 注册 → 数据长度预检自动化。**105/105 tests pass**。详见 [`docs/project_status.md`](./docs/project_status.md) 与 [`docs/archive/review.md`](./docs/archive/review.md)。

---

## 🔥 核心结果

> 实验尚未运行。GPU 集成完成后回填本节。

| 指标 | Vanilla GRPO | +Aug | +StratAdv (Ours) | 相较 Vanilla |
|------|-------------|------|-----------------|-------------|
| **整体 pass@1** | — | — | — | — |
| **原版 pass@1** | — | — | — | — |
| **mild pass@1** | — | — | — | — |
| **strong pass@1** | — | — | — | — |
| **Robustness Gap** | — | — | — | — |

---

## 🎯 问题：GRPO 放大 Schema 过拟合

标准 GRPO 在 function calling 任务上有三个致命问题：

1. **Schema 表面形式过拟合**：GRPO 会放大模型对训练 schema 具体工具名 / 描述措辞 / 枚举值的依赖，换个同义写法就脚
2. **群组奖励饱和**：二元 reward（0/1），group_size 较大时容易全 0 或全 1 → advantage 方差→0 → 梯度消失
3. **扰动信号淹没**：强扰动下模型更难答对，reward 偏低。标准 GRPO 的 prompt 内 z-score 会惩罚所有低分 rollout，包括"困难条件下表现合理"的

```
训练：search_flights(origin="JFK", destination="LHR")  → 正确
测试：find_flights(departure_location="JFK", to="LHR") → 失败
```

---

## 💡 方法

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

## 🌟 技术亮点

### 1. 信号传输理论（算法贡献）
将 schema 鲁棒性 RL 的失效模式分解为**信号源**与**信号通路**：
- **信号源**（Schema Aug）：~100 条规则的纯函数扰动器，在 prompt 层制造 none/mild/strong 三档表面形式变体。
- **信号通路**（Stratified Advantage）：`strat_z + 0.25 × global_z`，按扰动强度分层归一化，防止强扰动 rollout 被弱扰动 rollout 的 reward 碾压。
- **单独失效**：仅 Aug 而无分层归一化时，强扰动样本被淹没；仅分层而无 Aug 时无表面形式多样性。**联合方案才能解锁鲁棒性提升**（待实验验证）。

> 该分解是**模型无关的**，可推广至任何 schema-perturbation-driven 的 RL 任务（MCP / OpenAPI / 工具调用）。

### 2. 零参数 Process Reward
`SchemaPerturber` 纯规则式扰动器：
- **可解释**：每条扰动来自显式的同义词词典（`api_mapper.py` 维护 `name_map` / `enum_map`）。
- **零可训练参数**：不需要额外的 reward model 训练，不引入 reward hacking 风险。
- **确定性**：给定 `(task_id, level, seed)` 完全可复现。

### 3. 自定义 verl Estimator
- 通过 verl `@register_adv_est("schemashift_grpo")` 注册，从 uid (`task_id___level`) 解析分组与分层。
- `register_estimator.py` monkey-patch `verl.compute_advantage` 注入 `non_tensor_batch`（含 perturbation_level、group_id），打通 verl 主路径。
- Ray actor 跨进程注册：`run_exp4.py` 在 actor `__init__` 中重新注册 estimator，避免 `Trainer.fit` 之后的 estimator 丢失。

### 4. 工程纪律
- **数据长度预检**（`length_check.py`）：训练前扫描 parquet，超过 `max_prompt_length` 的样本 fail-fast，避免 verl 内部静默截断。
- **Bounded linear BFCL parser**：把 `_parse_bfcl_native_args` 从 O(N²) 降到 O(N)，解决 AgentLoopWorker 在长函数列表上卡死的根因。
- **Smoke 全链路验证**：E3 / E4 / E5 三入口跑到 step 1，loss / grad_norm / ckpt 落盘均经过断言。

---

## 🏗️ 项目结构

```raw
📦 schemashift-grpo/
├── 💻 src/                                  # 核心源码
│   ├── 🌍 envs/                             # BFCL 环境与 schema 扰动
│   │   ├── 🐍 schema_perturber.py           # 扰动生成器（~380 行）
│   │   ├── 🐍 api_mapper.py                 # 函数名 + 枚举值双向映射（~130 行）
│   │   └── 🐍 bfcl_env.py                   # BFCL 多轮环境
│   ├── 🎓 training/                         # GRPO estimator 与训练入口
│   │   ├── 🐍 schemashift_grpo_estimator.py # 自定义 verl estimator
│   │   ├── 🐍 schemashift_advantage.py      # 独立 advantage 函数
│   │   ├── 🐍 register_estimator.py         # verl 集成入口（monkey-patch）
│   │   ├── 🐍 length_check.py               # 数据长度预检（fail-fast + warn）
│   │   ├── 🐍 run_exp3.py                   # E3 / E5 入口
│   │   └── 🐍 run_exp4.py                   # E4 入口（注入 SchemaShiftTaskRunner）
│   ├── 🤖 agent_loop/
│   │   └── 🐍 bfcl_agent_loop.py            # BFCL 多轮 agent loop
│   ├── 🏆 reward/
│   │   └── 🐍 bfcl_reward.py                # BFCL 正确性 reward
│   └── 📊 eval/
│       ├── 🐍 bfcl_eval.py                  # 鲁棒性评估器（none/mild/strong）
│       └── 🐍 matching.py                   # ground truth 匹配
├── ⚙️ configs/                              # 实验 YAML 配置
│   ├── 📜 exp2_sft.yaml / exp2_sft_smoke.yaml
│   ├── 📜 exp3_vanilla_grpo.yaml            # E3 基线
│   ├── 📜 exp4_schemashift.yaml             # E4 主实验（StratAdv + Aug）
│   ├── 📜 exp5_aug_only.yaml                # E5 消融（仅 Aug）
│   └── 📜 agent_loop.yaml                   # agent loop 共享配置
├── 📜 scripts/
│   ├── 🚀 train/grpo/                       # GRPO 训练启动脚本
│   │   ├── 📜 run_vanilla_grpo.sh           # E3 基线
│   │   ├── 📜 run_schemashift.sh            # E4 主实验
│   │   └── 📜 run_aug_only.sh               # E5 消融
│   ├── 🎓 train/sft/                        # SFT 预热脚本
│   │   ├── 📜 run_exp2_sft.sh
│   │   └── 🐍 run_exp2_sft.py
│   ├── 📈 eval/                             # 评测启动脚本
│   │   └── 🐍 eval_zero_shot.py
│   ├── 🐍 build_parquet.py                  # 数据 → parquet（含 uid 编码）
│   ├── 🐍 generate_perturbations.py         # schema 扰动批量生成
│   ├── 🐍 check_environment.py              # 环境自检
│   ├── 🐍 plot_results.py                   # 结果可视化
│   ├── 🐍 download_data.py                  # 数据下载
│   └── 📜 setup.sh                          # 一键环境搭建
├── 📚 docs/                                 # 项目文档
│   ├── 📝 project_status.md                 # 工程进度总账
│   ├── 📝 technical_report.md               # 技术方案与论文式说明
│   ├── 📝 ablation_plan.md                  # 消融实验计划
│   ├── 🖼️ figures/                          # 配图（实验跑完后回填真实数据）
│   └── 📁 archive/                          # 历史 review、环境快照等归档
├── 🧪 tests/                                # 单元 / 集成测试（105 passed）
├── 📁 data/                                 # BFCL 原始数据 + parquet 输出
├── 📁 experiments/                          # 训练 ckpt 与评测输出（git 不追踪内容）
├── 📜 requirements.txt
└── 🛠️ pyproject.toml
```

---

## 🚀 快速开始

### 1. 环境搭建

```bash
# 安装项目依赖（含 dev extras）
pip install -e ".[dev]"
pip install pyarrow pandas

# 安装 verl（参考 fork，v0.6.1）
pip install -e verl/

# 环境自检
python scripts/check_environment.py
```

> **硬件**：4×A10（24GB） / 单卡 A100-80GB 均可。
> **栈**：torch 2.8 / vllm 0.11 / flash_attn 2.7.3 / verl 0.6.1。

### 2. 数据准备

```bash
# BFCL multi-turn → parquet（含 uid: task_id___level 编码）
python scripts/build_parquet.py
```

### 3. 训练

```bash
# E3: Vanilla GRPO 基线
bash scripts/train/grpo/run_vanilla_grpo.sh

# E5: Aug Only 消融（先跑）
bash scripts/train/grpo/run_aug_only.sh

# E4: SchemaShift-GRPO 主实验（后跑）
bash scripts/train/grpo/run_schemashift.sh
```

### 4. 测试

```bash
pytest tests/                # 105 passed
```

---

## 📚 文档

| 📄 文档 | 📝 内容 |
|---------|---------|
| [`docs/project_status.md`](./docs/project_status.md) | **工程进度总账**：每日修复链、smoke 验证记录、未完事项 |
| [`docs/technical_report.md`](./docs/technical_report.md) | **技术方案**：动机、方法、消融设计、假设与验证路径 |
| [`docs/ablation_plan.md`](./docs/ablation_plan.md) | **消融实验计划**：E1–E6 排期、资源预算、产出物清单 |
| [`docs/archive/review.md`](./docs/archive/review.md) | 历史 review：版本契约、设计回顾 |
| [`CLAUDE.md`](./CLAUDE.md) | Agent 协作约定（项目规范、红线、命名规则） |

---

## 🛠️ 技术栈

- **训练框架**：[veRL](https://github.com/volcengine/verl) 0.6.1（FSDP + vLLM rollout）
- **策略模型**：Qwen2.5-1.5B-Instruct
- **评测基准**：[BFCL V3 multi-turn](https://gorilla.cs.berkeley.edu/leaderboard.html)（function calling）
- **推理引擎**：vLLM 0.11
- **注意力**：FlashAttention 2.7.3
- **硬件**：4×A10（24GB）或 1×A100（80GB）

---

## 🙏 致谢

- [veRL](https://github.com/volcengine/verl) —— 开源 RL 训练框架
- [BFCL / Gorilla](https://gorilla.cs.berkeley.edu/leaderboard.html) —— 多轮 function calling 评测基准
- [Qwen](https://github.com/QwenLM/Qwen) —— 强大的开源基座模型
- [agentic-grpo-longhorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon) —— 项目结构与 README 风格的参考来源

---

## 📜 许可

MIT License. 个人研究用途，不附带任何明示或暗示的担保。

---

> **为什么重要**：当前主流 RLHF / RLAIF 工作聚焦单轮问答或代码生成，但**多轮、多工具、schema 易变**的 function calling 场景里，vanilla GRPO 会因群组饱和与 schema 表面形式过拟合而失效。SchemaShift-GRPO 把"鲁棒性"问题分解为可工程化的"信号源 + 信号通路"，提供一条无需额外 reward model、可解释、轻量的鲁棒训练路径，对 MCP / OpenAPI / 工具调用类 RL 任务具备直接迁移价值。
