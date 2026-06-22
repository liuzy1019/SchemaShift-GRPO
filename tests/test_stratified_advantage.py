"""
MCP-RL Full 2 维 stratum advantage 测试。

覆盖：
  - 基本 2 维分层
  - fallback 逻辑（stratum < 3 → scenario → global）
  - 与旧 1 维分层的兼容性
  - 边界情况
"""
import torch
import pytest

from src.training.schemashift_advantage import (
    compute_stratified_advantage,
    compute_schemashift_advantages,
)


class TestStratifiedAdvantage:
    """2 维 stratum advantage 测试。"""

    def test_basic_2d_stratum(self):
        """基本 2 维分层：(none, single_step) vs (mild, distractor)。"""
        rewards = torch.tensor([0.9, 0.8, 0.7, 0.5, 0.4, 0.3])
        levels = ["none"] * 3 + ["mild"] * 3
        scenarios = ["single_step"] * 3 + ["distractor"] * 3

        adv = compute_stratified_advantage(rewards, levels, scenarios, beta=0.0)

        assert adv.shape == (6,)
        assert torch.all(torch.isfinite(adv))
        # 每个 stratum 内最高 reward 应得正 advantage
        assert adv[0] > 0  # none/single_step 最高
        assert adv[3] > 0  # mild/distractor 最高

    def test_fallback_single_sample_to_scenario(self):
        """单样本 stratum 回退到 scenario-level。"""
        rewards = torch.tensor([0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2])
        levels = ["none", "none", "none", "mild", "mild", "mild", "strong"]
        scenarios = ["single_step"] * 3 + ["single_step"] * 3 + ["single_step"]
        # (strong, single_step) 只有 1 个样本，应回退到 scenario=single_step（7 个样本）

        adv = compute_stratified_advantage(rewards, levels, scenarios, beta=0.0)

        assert adv.shape == (7,)
        assert torch.all(torch.isfinite(adv))
        # strong 的唯一样本 reward=0.2，低于 scenario 均值，应得负 advantage
        assert adv[6] < 0

    def test_fallback_two_samples(self):
        """2 样本 stratum：只减均值不除 std。"""
        rewards = torch.tensor([0.8, 0.2])
        levels = ["none", "none"]
        scenarios = ["distractor", "distractor"]

        adv = compute_stratified_advantage(rewards, levels, scenarios, beta=0.0, min_stratum_size=3)

        # 2 样本：只减均值 → (0.8-0.5, 0.2-0.5) = (0.3, -0.3)
        assert adv[0] > 0
        assert adv[1] < 0
        assert torch.isclose(adv[0], -adv[1], atol=1e-5)

    def test_fallback_single_to_global(self):
        """单样本且 scenario 也只有 1 样本：回退到 global。"""
        rewards = torch.tensor([0.9, 0.8, 0.7, 0.1])
        levels = ["none", "none", "none", "strong"]
        scenarios = ["single_step", "single_step", "single_step", "output_error"]
        # (strong, output_error) 只有 1 样本，且 scenario=output_error 也只有 1 样本

        adv = compute_stratified_advantage(rewards, levels, scenarios, beta=0.0)

        assert adv.shape == (4,)
        assert torch.all(torch.isfinite(adv))
        # reward=0.1 低于全局均值，应得负 advantage
        assert adv[3] < 0

    def test_beta_effect(self):
        """beta > 0 时加入全局残差。"""
        rewards = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        levels = ["none"] * 3 + ["strong"] * 3
        scenarios = ["single_step"] * 6

        adv_no_beta = compute_stratified_advantage(rewards, levels, scenarios, beta=0.0)
        adv_with_beta = compute_stratified_advantage(rewards, levels, scenarios, beta=0.25)

        # beta=0 时 strong 层内最高(0.3)得正 advantage
        assert adv_no_beta[3] > 0
        # beta=0.25 时 strong 层内最高(0.3)的 advantage 应降低（因为全局低于均值）
        assert adv_with_beta[3] < adv_no_beta[3]

    def test_mixed_scenarios(self):
        """混合场景类型。"""
        rewards = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        levels = ["none"] * 3 + ["mild"] * 3 + ["strong"] * 3
        scenarios = ["single_step", "distractor", "conditioned_tool_call"] * 3

        adv = compute_stratified_advantage(rewards, levels, scenarios, beta=0.25)

        assert adv.shape == (9,)
        assert torch.all(torch.isfinite(adv))

    def test_length_mismatch_raises(self):
        """长度不匹配应报错。"""
        rewards = torch.tensor([1.0, 0.5])
        levels = ["none"]
        scenarios = ["single_step", "single_step"]

        with pytest.raises(ValueError):
            compute_stratified_advantage(rewards, levels, scenarios)

    def test_backward_compat_with_1d(self):
        """当所有 scenario_type 相同时，应与旧 1 维分层结果一致。"""
        rewards = torch.tensor([0.9, 0.8, 0.7, 0.5, 0.4, 0.3])
        levels = ["none"] * 3 + ["mild"] * 3
        scenarios = ["single_step"] * 6

        adv_2d = compute_stratified_advantage(rewards, levels, scenarios, beta=0.25)
        adv_1d = compute_schemashift_advantages(rewards, levels, beta=0.25)

        # 当 scenario 全相同时，2 维退化为 1 维，结果应一致
        assert torch.allclose(adv_2d, adv_1d, atol=1e-5)

    def test_all_same_reward(self):
        """所有 reward 相同时 advantage 应全为 0。"""
        rewards = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        levels = ["none"] * 3 + ["mild"] * 3
        scenarios = ["single_step"] * 3 + ["distractor"] * 3

        adv = compute_stratified_advantage(rewards, levels, scenarios)

        assert torch.allclose(adv, torch.zeros(6), atol=1e-5)
