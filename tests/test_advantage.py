"""
LiveMCP advantage 单元测试。
"""
import torch
import pytest
from src.training.livemcp_advantage import (
    compute_livemcp_advantages,
    compute_standard_grpo_advantages,
)


class TestComputeLiveMCPAdvantages:
    """compute_livemcp_advantages 功能测试。"""

    def test_basic_3x3(self):
        """3:3:3 分组的基本测试。"""
        rewards = torch.tensor(
            [0.9, 0.8, 0.7,    # none
             0.5, 0.4, 0.3,    # mild
             0.2, 0.1, 0.0]    # strong
        )
        levels = ["none"] * 3 + ["mild"] * 3 + ["strong"] * 3

        adv = compute_livemcp_advantages(rewards, levels)

        assert adv.shape == (9,), f"输出形状应为 (9,)，实际为 {adv.shape}"
        # 每个层级内应该有正负 advantage
        assert adv[:3].std() > 0, "none 组应有标准差"
        assert adv[3:6].std() > 0, "mild 组应有标准差"
        assert adv[6:9].std() > 0, "strong 组应有标准差"

    def test_protects_strong_correct(self):
        """strong 层内的正确者不应被全局碾压。"""
        rewards = torch.tensor(
            [0.9, 0.1,     # none: 一好一差
             0.5, 0.1,     # mild: 一好一差
             0.3, 0.1]     # strong: 一好一差
        )
        levels = ["none"] * 2 + ["mild"] * 2 + ["strong"] * 2

        adv = compute_livemcp_advantages(rewards, levels, beta=0.0)

        # beta=0.0 表示只看层内比较
        # strong 层内 reward=0.3 的应该拿到正 advantage
        assert adv[4] > 0, "strong-correct 应有正 advantage"

    def test_same_level_differentiates(self):
        """同一层内应能区分好坏。"""
        rewards = torch.tensor([0.6, 0.5, 0.4])
        levels = ["none"] * 3

        adv = compute_livemcp_advantages(rewards, levels, beta=0.0)
        assert adv[0] > adv[1] > adv[2]

    def test_cross_level_correction(self):
        """跨层校正应降低强扰动中正确者的 advantage（相对值）。"""
        rewards = torch.tensor(
            [0.9, 0.1,     # none
             0.3, 0.1]     # strong
        )
        levels = ["none"] * 2 + ["strong"] * 2

        adv_level_only = compute_livemcp_advantages(
            rewards, levels, beta=0.0
        )
        adv_full = compute_livemcp_advantages(
            rewards, levels, beta=0.25
        )

        # strong-correct (index 2, reward=0.3) 在 beta=0.0 时为正
        # 加跨层校正后应该降低（因为 0.3 低于全局 mean）
        assert adv_level_only[2] > 0
        assert adv_full[2] < adv_level_only[2]

    def test_missing_level(self):
        """缺少某个扰动强度时不应崩溃。"""
        rewards = torch.tensor([0.9, 0.8, 0.7, 0.6])
        levels = ["none"] * 4  # 只有 none

        adv = compute_livemcp_advantages(rewards, levels)
        assert adv.shape == (4,)

    def test_length_mismatch(self):
        """长度不匹配时应报错。"""
        rewards = torch.tensor([1.0, 0.5])
        levels = ["none"]

        with pytest.raises(ValueError):
            compute_livemcp_advantages(rewards, levels)

    def test_beta_extremes(self):
        """beta=0 退化为纯层内归一化（只减层均值，无全局残差）。"""
        rewards = torch.tensor([0.9, 0.8, 0.5, 0.4, 0.2, 0.1])
        levels = (["none"] * 2 + ["mild"] * 2 + ["strong"] * 2)

        adv_grpo = compute_standard_grpo_advantages(rewards)
        adv_schema = compute_livemcp_advantages(
            rewards, levels, beta=0.0
        )

        # beta=0.0 时纯分层内，不同于标准 GRPO
        assert not torch.allclose(adv_grpo, adv_schema)

    def test_mixed_batch(self):
        """混合 batch — 验证分层 advantage 的数值正确性。"""
        rewards = torch.tensor(
            [0.9, 0.8, 0.7,    # none
             0.5, 0.4, 0.3,    # mild
             0.2, 0.1, 0.0]    # strong
        )
        levels = ["none"] * 3 + ["mild"] * 3 + ["strong"] * 3

        adv = compute_livemcp_advantages(rewards, levels)

        # 所有值应为有限数
        assert torch.all(torch.isfinite(adv))
        assert not torch.any(torch.isnan(adv))

        # 层内最高 reward 应得正 advantage（分层归一化）
        # none 层: 0.9,0.8,0.7 → 0.9 最高 → adv > 0
        assert adv[0] > 0, "none 层最高 reward 应得正 advantage"
        # strong 层: 0.2,0.1,0.0 → 0.0 最低 → adv < 0
        assert adv[8] < 0, "strong 层最低 reward 应得负 advantage"

        # 层内均值应接近 0（分层归一化后每层均值为 0，加 beta*global_z 后略有偏移）
        none_mean = adv[:3].mean()
        mild_mean = adv[3:6].mean()
        strong_mean = adv[6:9].mean()
        # 全局残差 beta=0.25, global_z 范围有限, 层均值不应偏离 0 太远
        assert abs(none_mean.item()) < 0.8, f"none 层均值偏离过大: {none_mean}"
        assert abs(strong_mean.item()) < 0.8, f"strong 层均值偏离过大: {strong_mean}"


class TestStandardGRPOAdvantages:
    """标准 GRPO advantage 测试（基线对照）。"""

    def test_basic(self):
        """基本功能。"""
        rewards = torch.tensor([1.0, 0.5, 0.0])
        adv = compute_standard_grpo_advantages(rewards)

        assert adv.shape == (3,)
        assert adv[0] > 0  # 最佳应正
        assert adv[2] < 0  # 最差应负

    def test_all_same(self):
        """所有 reward 相同。"""
        rewards = torch.tensor([0.5, 0.5, 0.5])
        adv = compute_standard_grpo_advantages(rewards)

        # std=0 时，所有 advantage=0
        assert torch.all(adv == 0)
