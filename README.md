# SchemaShift-GRPO

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![verl 0.6.1](https://img.shields.io/badge/verl-0.6.1-orange.svg)](https://github.com/volcengine/verl)
[![Tests 105 passed](https://img.shields.io/badge/tests-105%20passed-brightgreen.svg)](./tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#-许可)

> **面向 MCP / function calling Schema 鲁棒性的 GRPO 训练**
> 训练时对 BFCL V3 multi-turn 的 schema 做同义扰动（none/mild/strong），并在 verl 里换掉默认 GRPO advantage 为**按扰动强度分层的 z-score**。目标是让策略在换了工具名 / 描述措辞 / 枚举值的 schema 上仍能答对。

> **📦 状态**：代码与脚本就绪，单元 / 集成测试 105/105 通过。**正式训练与评测尚未开始，结果待跑出后回填**。工程过程细节见 [`docs/project_status.md`](./docs/project_status.md)。

---

## 🎯 问题

标准 GRPO 在 BFCL multi-turn 这类 function calling 任务上有两个已知问题：

1. **Schema 表面形式过拟合**：训练集里是 `search_flights(origin, destination)`，换成 `find_flights(departure_location, to)` 就答不对。
2. **群组奖励饱和**：二元 reward（0/1），group 内容易全 0 或全 1 → advantage 方差→ 0 → 梯度消失。

```
训练：search_flights(origin="JFK", destination="LHR")  → 正确
测试：find_flights(departure_location="JFK", to="LHR") → 失败
```

---

## 💡 方法

### Schema Augmentation

训练时按任务、按 seed 生成 3 个扰动强度的变体：

```
none:   原始 schema
mild:   tool_name 同义替换
        search_flights → find_flights
strong: tool_name + description + enum 全部替换
        search_flights → find_flights
        "Search for available flights" → "Lookup for accessible flights"
        "economy" → "standard"
```

扰动器是纯规则的（`schema_perturber.py`），`api_mapper.py` 同时维护 `name_map` / `enum_map`，用于训练时 API dispatch 反向映射和评测时 ground truth 同步。同一 `(task_id, level, seed)` 给定后完全可复现。

### Stratified Advantage

把 verl 默认的 GRPO advantage 换为按扰动强度分层的 z-score 加上跨层残差：

```
A = strat_z + beta × global_z

strat_z:  同一 task 在同扰动层（none/mild/strong）内 z-score
global_z: 同一 task 跨层 z-score
beta:     默认 0.25
```

- beta=0 → 纯分层归一化
- beta 趋于 ∞ → 趋近标准 GRPO
- 单样本层 / 组：std 回退 1.0，不产生 NaN

公式与边界在 [`tests/test_advantage.py`](./tests/test_advantage.py) 中有 10 个 case 覆盖。

### 与 verl 集成

注册为 verl 的自定义 estimator：

```python
@register_adv_est("schemashift_grpo")
```

- uid 编码为 `"{task_id}___{level}"`，estimator 从 uid 解析出 task 分组与 level 分层。
- `register_estimator.py` monkey-patch `verl.compute_advantage` ，向 `non_tensor_batch` 注入 `perturbation_level` / `group_id`。
- `run_exp4.py` 在 Ray actor `__init__` 重新注册 estimator，避免跨进程丢失。

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
│   └── 📝 ablation_plan.md                  # 消融实验计划
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

## 🔥 核心结果

> ⏳ 实验尚未运行，本节为待回填占位。
> 跑完 E3 / E4 / E5 后按下表结构填入真实数据。

### 主指标

| 指标 | E3 Vanilla GRPO | E5 +Aug | E4 +StratAdv (Ours) | Δ vs E3 |
|------|-----------------|---------|---------------------|---------|
| `pass@1(none)`         | TBD | TBD | TBD | TBD |
| `pass@1(mild)`         | TBD | TBD | TBD | TBD |
| `pass@1(strong)`       | TBD | TBD | TBD | TBD |
| `robust_avg`           | TBD | TBD | TBD | TBD |
| `robustness_gap` (↓)   | TBD | TBD | TBD | TBD |

> 指标定义见 [`docs/ablation_plan.md`](./docs/ablation_plan.md) §4。

### 训练曲线

> 待 E3 / E4 / E5 正式训练完成后回填。

### 假设验证

| # | 假设 | 比较 | 状态 |
|---|------|------|------|
| H1 | GRPO 放大 schema 过拟合 | E3 vs E2 | ⏳ 待验证 |
| H2 | Schema Aug 降低 robustness gap | E5 vs E3 | ⏳ 待验证 |
| H3 | StratAdv 进一步提升 | E4 vs E5 | ⏳ 待验证 |
| H4 | 联合方案 > 最优基线 | E4 vs max(E3, E6) | ⏳ 待验证 |

> 假设来源与判据见 [`docs/ablation_plan.md`](./docs/ablation_plan.md) §3。

---

## 📚 文档

| 📄 文档 | 📝 内容 |
|---------|---------|
| [`docs/project_status.md`](./docs/project_status.md) | 工程进度总账：问题、方法、修复链、smoke 验证记录 |
| [`docs/technical_report.md`](./docs/technical_report.md) | 技术方案：BFCL 兼容修复、扰动器规则、StratAdv 公式与边界 |
| [`docs/ablation_plan.md`](./docs/ablation_plan.md) | 消融实验矩阵：E1–E6 设计、消融链、reward / eval 约定 |

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

- [veRL](https://github.com/volcengine/verl) —— RL 训练框架
- [BFCL / Gorilla](https://gorilla.cs.berkeley.edu/leaderboard.html) —— 多轮 function calling 评测基准
- [Qwen](https://github.com/QwenLM/Qwen) —— 基座模型

---

## 📜 许可

MIT License.
