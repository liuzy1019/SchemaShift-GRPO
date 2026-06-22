# Toucan Fallback Decision

> Phase 2 产出 — 2026-06-19
> 基于 Phase 1 inspection report 做出的数据策略决定

## 结论：Toucan 作为主数据源，不需要 fallback

### 依据

| 指标 | 全量值 | 排除 irrelevant 后 | 阈值 | 判定 |
|------|--------|-------------------|------|------|
| parseable_rate | 100% | 100% | > 90% | ✅ |
| verifier_ready_rate | 47.88% | **72.2%** | > 40% | ✅ |
| tool_observation_pairing_rate | 82.2% | **100%** (paired subset) | > 80% | ✅ |
| arguments_parseable_rate | 65.6% | **98.9%** (有 tool_call 的) | > 60% | ✅ |
| short_episode_rate (≤3 turns) | 60.04% | — | > 30% | ✅ |
| unique_tools | 845 | — | > 100 | ✅ |
| unique_mcp_servers | 365 | — | > 50 | ✅ |

### 关键发现

1. **irrelevant 子集 (1682/5000 = 33.6%)** 是天然的 `no_tool` 训练数据
   - 有工具 schema 但不需要调用工具
   - 直接转为 `episode_type=no_tool`

2. **排除 irrelevant 后 verifier_ready_rate 达到 72.2%**
   - 远超 40% 的 fallback 阈值
   - 不需要用 ToolACE 补充

3. **并行调用支持良好 (17.8%)**
   - 足够训练 parallel_tool_call 场景

4. **数据多样性极高**
   - 845 unique tools / 365 unique MCP servers
   - 覆盖 weather, finance, media, programming, language 等多领域

### ToolACE 的角色

ToolACE **不作为 fallback**，但保留以下用途：
- exact reward 单元测试的 ground truth
- schema perturbation 验证
- argument key/value verifier 测试
- 基线对比实验

### 数据使用策略

```text
Toucan (主数据源):
  - single-turn-original + single-turn-diversify → call_only / call_then_final
  - multi-turn → call_then_call / call_then_final
  - irrelevant → no_tool

ToolACE (辅助):
  - 仅用于 verifier 单元测试和基线对比
  - 不参与主训练循环
```
