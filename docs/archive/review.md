# SchemaShift-GRPO Review（2026-06-16 收尾）

> 本轮目标：把"E3 进入 rollout 但被 BFCL bug 阻塞"推进到"E3 / E4 / E5 三个 step 1 smoke 全通过、105/105 测试 pass、长度/依赖/解析三道工程兜底就位"。结果达成。**正式训练（>1 step）尚未跑**。
> 上一轮 review.md（早晨版）已被本文件替代，历史决策见 git log。

---

## 1. 本轮 smoke 结果（4×A10）

按用户判据"step 1 完成 + loss finite + 无 traceback"为 pass。三个全过：

| 实验 | 状态 | step 1 timing | max_mem | 关键 marker |
|---|---|---|---|---|
| **E3** vanilla GRPO | ✅ PASS | **55.17 s** | **22.41 GB** | grad_norm finite, ckpt saved |
| **E5** aug only | ✅ PASS | **145.28 s** | **18.03 GB** | grad_norm finite, ckpt saved |
| **E4** SchemaShift-GRPO | ✅ PASS | ~13 min（含 final val rollout） | ~21 GB（offload 后） | `schemashift_grpo monkey-patch batch=36, has_perturbation_level=True` + `[schemashift_grpo] tasks=4, levels=3` |

跑 smoke 的标准命令在 `CLAUDE.md`。

---

## 2. 本轮发现并修复的项

### 2.1 `_parse_bfcl_native_args` O(N²) → bounded linear（重大）

**症状**：E3 smoke 在 step 1 rollout 阶段 cluster-wide 卡死 15+ 分钟，GPU util 跌到 0%，进程不退。

**定位**：py-spy 5 轮采样（间隔 2s）一致显示 3 个 AgentLoopWorker 的 AsyncIO Thread 全部停在 `_parse_bfcl_native_args:180`，状态 `active+gil`——纯 Python CPU 循环。

**根因**：原实现含 `args_part.find("=", i)` × outer while 的 O(N²) 路径。第 1 step 模型尚未 RL 化，会产生超长 garbled 函数调用（args_part 几万字符），单次解析需要数十分钟。

**修复**：重写为 bounded linear parser（`src/agent_loop/bfcl_agent_loop.py:_parse_bfcl_native_args`）：
- 输入硬上限 `_PARSE_MAX_INPUT_LEN=8192`
- 段数上限 `_PARSE_MAX_ARGS=64`
- key 上限 `_PARSE_MAX_KEY_LEN=64`
- 字面量回填上限 `_PARSE_MAX_LITERAL_LEN=4096`
- 单遍线性扫描，所有 while 强制 `i += 1`，永远 O(N)
- 解析失败返回 `("", {})` 或 `(name, {})`，调用方走既有 `ToolCallResult(success=False)` fallback

**实测耗时**（micro bench）：
- 正常调用 24-70B：30-180 μs
- 超长 all-A 20K：0.00 ms（直接降级）
- = bomb 6K：1.51 ms（修复前 >15 分钟）
- quoted long 7K：0.93 ms
- nested 4K 层：1.29 ms

**回归测试**：`tests/test_runtime_regressions.py` 新增 4 个 case（超长输入毫秒级 / 12 类畸形输入不抛异常 / 段数超限优雅降级 / 嵌套结构内 `,` 不误切段）。

### 2.2 E4 estimator 未注册到 ray actor 进程（重大）

**症状**：E4 smoke 跑到 actor 内 `Unknown advantage estimator: schemashift_grpo`。

**根因**：`run_exp4.py` 在主进程注册 estimator + monkey-patch `compute_advantage`，但 verl trainer 跑在独立 ray actor 进程，主进程的 dict / monkey-patch 不会跨进程传递。

**修复**：`src/training/run_exp4.py` 引入 `SchemaShiftTaskRunner(TaskRunner)` 子类，在 `run()` 入口重新执行注册，通过 `verl.trainer.main_ppo.run_ppo(config, task_runner_class=...)` hook 注入。这是 verl 官方扩展点，无侵入。

### 2.3 数据长度自动校验

**问题**：之前 yaml 里 `max_prompt_length=2048` 没人验过，dataloader 在第 1 个 batch 直接 raise，浪费整个 init 周期。

**修复**：新增 `src/training/length_check.py`：
- `check_split_length`：跑 chat_template + tokenize，统计 max/p99/p95，超 limit → RuntimeError，p99 > limit×0.95 → loguru warning
- `assert_e4_group_integrity`：E4 专属 3:3:3 group 检查（从 run_exp4 抽出来复用）
- `maybe_run_length_check`：入口包装，默认开启，`SCHEMASHIFT_SKIP_LENGTH_CHECK=1` 跳过
- `run_exp3.py` / `run_exp4.py` 入口 `main()` 第一行调用

**实测校准**（Qwen2.5-1.5B tokenizer，900/100 行）：
```
train: max=8809  p99=8732  p95=8713  p50=6820  limit=10240  overflow=0  near_limit=0
val:   max=8809  p99=8732  p95=8709  p50=6333  limit=10240  overflow=0  near_limit=0
```
yaml 注释里写的 "p99≈8800" 跟实测 p99=8732 完全对齐。

### 2.4 依赖契约 CI 兜底

**问题**：之前 vllm/torch/trl 误升会静默裂掉，没有 import 阶段就能看出来的兜底。

**修复**：新增 `tests/test_smoke_grpo_imports.py`，16 个测试覆盖 vllm ≤ 0.11.0 上限、flash_attn import、trl ValueHead 符号、tensordict、flashinfer JIT 启动、verl import、关键 monkey-patch 点存在性。

---

## 3. 累积工程兜底（含早晨修的）

| 文件 | 改动 | 类型 |
|---|---|---|
| `configs/exp3_vanilla_grpo.yaml` | `max_prompt_length` 2048 → 10240 | 早晨 |
| `configs/exp4_schemashift.yaml` | 同上 | 早晨 |
| `configs/exp5_aug_only.yaml` | 同上 | 早晨 |
| `scripts/train/grpo/run_*.sh` 三个 | `train_batch_size = lcm(mini_batch, group_size)` 自适应、`agent.num_workers=${N_GPUS}`、默认 `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | 早晨 |
| `src/agent_loop/bfcl_agent_loop.py` | (1) 透传 `test_entry_id = f"sshift_{request_id}"` 到 BFCL；(2) `_parse_bfcl_native_args` 重写为 bounded linear parser | 早晨 + 收尾 |
| `src/training/run_exp4.py` | 引入 `SchemaShiftTaskRunner` 跨 ray actor 重注册；`_assert_group_integrity_after_filter` 重构为调 `length_check.assert_e4_group_integrity` | 收尾 |
| `src/training/run_exp3.py` | 入口接入 `length_check.maybe_run_length_check` | 收尾 |
| `src/training/length_check.py` | 新增（177 行） | 收尾 |
| `tests/test_runtime_regressions.py` | 新增 4 个 parser 灾难性输入测试 | 收尾 |
| `tests/test_length_check.py` | 新增 9 个测试 | 收尾 |
| `tests/test_smoke_grpo_imports.py` | 新增 16 个测试 | 收尾 |

---

## 4. 测试

| 文件 | tests | 备注 |
|---|---|---|
| test_advantage | 10 | 公式 + 边界 |
| test_api_mapper | 9 | 函数名 + 枚举值 |
| test_schema_perturber | 13 | 扰动 / 确定性 / 安全 |
| test_schemashift_grpo_estimator | 11 | 注册 + 分层 advantage + uid fallback |
| test_runtime_regressions | 19 | parquet / BFCL 解析（含 4 个新 parser regression） / reward / eval |
| test_enum_mapping | 14 | enum 映射 |
| test_register_estimator_integration | 4 | monkey-patch 真实注入 |
| test_smoke_grpo_imports | 16 | 依赖契约 |
| test_length_check | 9 | 数据长度预检 |
| **合计** | **105** | **105/105 pass** |

---

## 5. 已确认通过的链路（事实）

| 环节 | 状态 | 证据 |
|---|---|---|
| arl 环境 import 链 | ✅ | sanity 7 项全绿 |
| verl 入口 hydra config | ✅ | TaskRunner dump config 完整 |
| 数据加载（10240 max_prompt）| ✅ | 三个 split 全部 0 overflow |
| FSDP 4 卡 init | ✅ | `[Gloo] Rank 0~3 connected` |
| vLLM 0.11 启动（flash_attn 后端）| ✅ | `Initializing a V1 LLM engine ... max_seq_len=14336` |
| CUDA graph capture | ✅ | `Capturing CUDA graphs (decode, FULL): 100%` |
| BFCL tool 调用 | ✅ | `test_entry_id` 透传，无 missing arg |
| BFCL 解析器 | ✅ | bounded linear，毫秒级，cluster GIL 卡死未再复现 |
| GRPO step 1（E3/E5）| ✅ | grad_norm finite, ckpt saved |
| SchemaShift estimator E4 | ✅ | tasks=4, levels=3，monkey-patch batch=36 has_perturbation_level=True |

---

## 6. 仍待办（按优先级）

**P0 / 全量训练前必做**
- 跑 ≥ 100 step 验证收敛（E3/E4/E5 各一次），观测 loss 曲线 + reward 是否抬升
- 跑一次 `SCHEMASHIFT_PRECHECK=1 python -m src.training.run_exp4 ...` 完整跑 group 完整性

**P1 / 后续**
- BFCL 官方 state-based eval 集成（当前仅 response-based AST）
- E1（zero-shot）+ E6（regularization baseline）补脚本
- `_parse_bfcl_native_args` 抽到 `src/envs/bfcl_parser.py`，解除 5 个文件对 `agent_loop` 的反向 import。**ROI 仅在要接第二个环境时显现，目前可不做**

**清理项（红线，须用户确认）**
- 删除根目录散落临时文件：`dev_summary.txt`(空) / `e4_quick.txt`(临时输出) / `file_path`(空) / `env_check_output.txt`(check_environment.py 的一次输出) / `test_core.py`(早期 smoke，已被 tests/ 覆盖)
- 清理 `outputs/2026-06-16/`（27 个 hydra run 子目录）和 `wandb/run-*`（16 个 smoke run），保留最近 1-2 个用于追溯即可

---

## 7. 历史决策保留

- 7 轮 Codex 对抗式审查（R1-R7，R7 P0=0），关键转折点 R6 改"monkey-patch verl"为"编码 uid + 注册自定义 estimator"
- 2026-06-15 完整代码 review：所有 P0 已修
- 2026-06-16 早晨：环境降级 + flashinfer 绕路 + max_prompt_length / num_workers / test_entry_id 修复
- 2026-06-16 收尾（本轮）：parser 灾难性慢路径修复 + E4 跨 actor 重注册 + 数据长度预检 + 依赖契约 CI + 三 smoke 全 pass

详见 git log。
