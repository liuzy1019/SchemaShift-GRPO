"""
SchemaShift-GRPO 自定义 advantage estimator。

注册为 verl 的 "schemashift_grpo" estimator。
优先从 non_tensor_batch（由 _get_gen_batch 保留，register_estimator.py 转发）读取
perturbation_level 和 group_id，fallback 到从 uid 解析。

uid 格式: {task_id}___{level}（由 build_parquet.py 生成）
"""

from collections import defaultdict

import numpy as np
import torch
from loguru import logger

try:
    from verl.trainer.ppo.core_algos import register_adv_est
except ImportError as e:
    raise ImportError(
        "verl 未安装或版本不兼容，schemashift_grpo estimator 无法注册。"
        f"原始错误: {e}"
    )


def _parse_uid(uid: str) -> tuple[str, str]:
    """从 uid 解析 (task_id, perturbation_level)。"""
    if "___" in uid:
        parts = uid.rsplit("___", 1)  # rsplit: level 在最后一段
        return parts[0], parts[1]
    return uid, "none"


def _diagnose_batch(index, non_tensor_batch, task_ids, levels, scores):
    """记录诊断信息，帮助排查数据流问题。"""
    n = len(scores)
    nunique_task = len(set(task_ids))
    nunique_level = len(set(levels))
    nb_fields = set(non_tensor_batch.keys()) if non_tensor_batch else set()
    logger.info(
        f"[schemashift_grpo] batch={n}, tasks={nunique_task}, "
        f"levels={nunique_level}, nb_fields={nb_fields}, "
        f"score_range=[{scores.min().item():.4f}, {scores.max().item():.4f}]"
    )


@register_adv_est("schemashift_grpo")
def compute_schemashift_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A = strat_z + beta * global_z。

    与 verl 集成：
    - verl 的 ray_trainer 通过 adv_kwargs 传入 token_level_rewards, response_mask, index(uid), config
    - non_tensor_batch 通过 register_estimator.py 的 monkey-patch 注入
    - 含 perturbation_level（分层）和 group_id（分组=task_id）
    - fallback: 从 uid 解析（uid 在训练时会被 verl 覆盖为随机 UUID，fallback 仅在 val 时有效）
    """
    beta = float(config.get("beta", 0.25) if config else 0.25)

    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]

    # ── 从 non_tensor_batch（优先）或 uid（fallback）解析分组和分层信息 ──
    non_tensor_batch = kwargs.get("non_tensor_batch")
    task_ids = []
    levels = []
    source = "uid"  # 标记数据来源，用于诊断

    if non_tensor_batch is not None:
        nb_groups = non_tensor_batch.get("group_id")
        nb_levels = non_tensor_batch.get("perturbation_level")
        if nb_groups is not None and nb_levels is not None:
            # 安全地将值转为 Python 列表（兼容 np.ndarray 向量/标量、list、单值 str）
            def _to_list(val):
                if isinstance(val, np.ndarray) and val.ndim > 0:
                    return val.tolist()
                elif isinstance(val, (list, tuple)):
                    return list(val)
                else:
                    # 单值（str, int, np.str_ 标量等）
                    return [str(val)]

            task_ids = _to_list(nb_groups)
            levels = _to_list(nb_levels)
            # 单值 → 广播到 batch 长度
            if len(task_ids) == 1 and bsz > 1:
                task_ids = task_ids * bsz
            if len(levels) == 1 and bsz > 1:
                levels = levels * bsz
            source = "non_tensor_batch"

    if not task_ids or not levels:
        # fallback: 从 uid 解析（训练时 uid 被覆盖为随机 UUID，分层信息丢失）
        logger.warning(
            "non_tensor_batch 中缺少 perturbation_level/group_id，"
            "回退到 uid 解析 — 训练阶段分层信息将丢失，"
            "advantage 退化（建议检查数据流水线）"
        )
        for uid in index:
            tid, level = _parse_uid(str(uid))
            task_ids.append(tid)
            levels.append(level)
        source = "uid (fallback)"

    with torch.no_grad():
        advantages = torch.zeros(bsz, device=scores.device)

        # 按 task_id 分组
        task2indices = defaultdict(list)
        for i, tid in enumerate(task_ids):
            task2indices[tid].append(i)

        for tid, indices in task2indices.items():
            idx_tensor = torch.tensor(indices, device=scores.device)
            group_scores = scores[idx_tensor]
            group_levels = [levels[i] for i in indices]
            n_group = len(group_scores)

            # 按 perturbation_level 分层
            level2local = defaultdict(list)
            for local_i, level in enumerate(group_levels):
                level2local[level].append(local_i)

            # Step 1: 层内归一化
            # strat_z = (r - μ_level) / σ_level  (norm_adv_by_std_in_grpo=True)
            # strat_z = (r - μ_level)              (norm_adv_by_std_in_grpo=False, Dr.GRPO 风格)
            strat_advs = torch.zeros(n_group, device=scores.device)
            for level, loc_indices in level2local.items():
                loc_tensor = torch.tensor(loc_indices, device=scores.device)
                level_scores = group_scores[loc_tensor]
                level_mean = level_scores.mean()
                if norm_adv_by_std_in_grpo and len(level_scores) >= 2:
                    level_std = level_scores.std(unbiased=False).clamp(min=epsilon)
                    strat_advs[loc_tensor] = (level_scores - level_mean) / level_std
                else:
                    # 单样本或 norm_adv_by_std_in_grpo=False: 只减均值
                    strat_advs[loc_tensor] = level_scores - level_mean

            # Step 2: 全局 z-score 用作残差
            group_mean = group_scores.mean()
            if norm_adv_by_std_in_grpo and n_group >= 2:
                group_std = group_scores.std(unbiased=False).clamp(min=epsilon)
                global_z = (group_scores - group_mean) / group_std
            else:
                global_z = group_scores - group_mean

            # Step 3: A = strat_z + beta * global_z
            advantages[idx_tensor] = strat_advs + beta * global_z

        # 首次执行时打诊断日志
        if not hasattr(compute_schemashift_grpo_advantage, '_diagnosed'):
            _diagnose_batch(index, non_tensor_batch, task_ids, levels, scores)
            compute_schemashift_grpo_advantage._diagnosed = True

        advantages = advantages.unsqueeze(-1) * response_mask

    return advantages, advantages
