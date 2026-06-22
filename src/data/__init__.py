"""MCP-RL Full 数据构建模块。

包含：
  - episode_schema: EpisodeSeed 正式 schema 定义（Phase 2 核心产出）
  - episode_seed_builder: 从 Toucan 数据构建 EpisodeSeed
  - distractor_sampler: 干扰工具采样（含同强度扰动）
  - conditioned_builder: 从真实数据加载 conditioned decision 和 no-tool 样本

旧的静态 parquet mixed_dataset_builder 已删除。新方案的数据入口应构建
episode_seed，并由 ReplayMCPExecutor/MCPToolEnvironment 在 rollout 时消费。
"""
