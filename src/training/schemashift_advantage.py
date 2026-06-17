"""
SchemaShift Advantage 计算。

核心 GRPO 算法改进：按 Schema 扰动强度分层归一化 + 跨层一致性校正。
"""

import torch
from loguru import logger

from src.envs.schema_perturber import PerturbationLevel, TRAINING_LEVELS


def compute_schemashift_advantages(
    rewards: torch.Tensor,
    perturbation_levels: list[PerturbationLevel],
    beta: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算 SchemaShift advantage：A = strat_z + beta * global_z。

    分层归一化：每层内独立 z-score。
    全局残差：beta * global_z，确保跨层梯度流动。

    Args:
        rewards: [N] 每个 rollout 的 scalar reward。
        perturbation_levels: [N] 每个 rollout 对应的扰动强度。
        beta: 全局残差强度。0 = 纯分层，0.25 = 默认。
        eps: 数值稳定常数。

    Returns:
        [N] 每个 rollout 的最终 advantage。

    Raises:
        ValueError: rewards 和 perturbation_levels 长度不匹配。

    Example:
        >>> rewards = torch.tensor([0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        >>> levels = ["none", "none", "none", "mild", "mild", "mild", "strong", "strong", "strong"]
        >>> adv = compute_schemashift_advantages(rewards, levels)
    """
    N = len(rewards)
    if len(perturbation_levels) != N:
        raise ValueError(
            f"rewards ({N}) 和 perturbation_levels ({len(perturbation_levels)}) "
            f"长度不匹配"
        )

    device = rewards.device
    advantages = torch.zeros(N, device=device)

    # ── Step 1: 按扰动强度分层归一化 ──
    levels_present = set(perturbation_levels)
    unsupported = levels_present - set(TRAINING_LEVELS)
    if unsupported:
        logger.warning(
            f"扰动强度包含非训练级别: {unsupported}，"
            f"将不回退到层内归一化（仅 global_z 生效）"
        )
    logger.debug(
        f"分层归一化: 扰动强度分布={levels_present}"
    )

    for level in TRAINING_LEVELS:
        if level not in levels_present:
            continue

        # 找到该强度的 rollout 索引
        mask = torch.tensor(
            [pl == level for pl in perturbation_levels],
            device=device,
            dtype=torch.bool,
        )
        idxs = torch.where(mask)[0]
        num_samples = len(idxs)

        if num_samples < 2:
            logger.warning(
                f"扰动强度 '{level}' 只有 {num_samples} 个样本，"
                f"回退到全局归一化"
            )
            continue

        level_rewards = rewards[idxs]
        level_mean = level_rewards.mean()
        level_std = level_rewards.std(unbiased=False).clamp(min=eps)

        level_advantages = (level_rewards - level_mean) / level_std
        advantages[idxs] = level_advantages

        logger.debug(
            f"  层 '{level}': {num_samples} 个样本, "
            f"mean={level_mean:.4f}, std={level_std:.4f}, "
            f"adv=[{level_advantages[0]:.4f}, ...]"
        )

    # ── Step 2: 全局残差校正 ──
    global_mean = rewards.mean()
    global_std = rewards.std(unbiased=False).clamp(min=eps)
    global_z = (rewards - global_mean) / global_std

    # ── Step 3: 融合（加法形式）──
    # A = strat_z + beta * global_z
    # beta=0 → 纯分层；beta=0.25 → 加全局残差
    final_advantages = advantages + beta * global_z

    logger.debug(
        f"Advantage 融合: beta={beta}, "
        f"adv_range=[{final_advantages.min():.4f}, {final_advantages.max():.4f}]"
    )

    return final_advantages


def compute_standard_grpo_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """标准 GRPO advantage 计算（基线对照用）。

    Args:
        rewards: [N] 每个 rollout 的 scalar reward。
        eps: 数值稳定常数。

    Returns:
        [N] advantage 向量。
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False).clamp(min=eps)
    advantages = (rewards - mean) / std
    return advantages
