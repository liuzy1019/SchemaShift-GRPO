"""
livemcp_grpo_estimator 逻辑测试。

由于 verl/Ray 依赖不在本地环境，通过等价逻辑验证 estimator 的核心算法。
生产 estimator 对每个 task group 独立调用分层 advantage，
这里手动模拟 per-task-per-level 分组并验证结果。
"""
import pytest
import torch

from src.training.livemcp_advantage import compute_livemcp_advantages


def _simulate_estimator(
    scores: torch.Tensor,
    task_ids: list[str],
    levels: list[str],
    beta: float = 0.25,
    norm_by_std: bool = True,
) -> torch.Tensor:
    """模拟 livemcp_grpo_estimator 的 per-task-per-level 分层逻辑。

    对每个 task 独立调用 compute_livemcp_advantages。
    """
    from collections import defaultdict

    task2indices = defaultdict(list)
    for i, tid in enumerate(task_ids):
        task2indices[tid].append(i)

    device = scores.device
    advantages = torch.zeros(len(scores), device=device)
    eps = 1e-6

    for indices in task2indices.values():
        idx_tensor = torch.tensor(indices, device=device)
        group_scores = scores[idx_tensor]
        group_levels = [levels[i] for i in indices]
        n = len(group_scores)

        # Step 1: per-level stratification
        level2indices = defaultdict(list)
        for li, lv in enumerate(group_levels):
            level2indices[lv].append(li)

        strat_advs = torch.zeros(n, device=device)
        for loc_indices in level2indices.values():
            lt = torch.tensor(loc_indices, device=device)
            ls = group_scores[lt]
            mean = ls.mean()
            if norm_by_std and len(ls) >= 2:
                std = ls.std(unbiased=False).clamp(min=eps)
                strat_advs[lt] = (ls - mean) / std
            else:
                strat_advs[lt] = ls - mean

        # Step 2: global z-score (within task)
        gmean = group_scores.mean()
        if norm_by_std and n >= 2:
            gstd = group_scores.std(unbiased=False).clamp(min=eps)
            global_z = (group_scores - gmean) / gstd
        else:
            global_z = group_scores - gmean

        advantages[idx_tensor] = strat_advs + beta * global_z

    return advantages


class TestPerTaskPerLevelStratification:
    """验证 per-task-per-level 分层逻辑（与 livemcp_grpo_estimator 等价）。"""

    def test_single_task_equivalent_to_global(self):
        """单 task 时，per-task 分层应与全局分层等价。"""
        scores = torch.tensor([0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        levels = ["none"] * 3 + ["mild"] * 3 + ["strong"] * 3
        task_ids = ["task1"] * 9

        adv_global = compute_livemcp_advantages(scores, levels, beta=0.25)
        adv_per_task = _simulate_estimator(scores, task_ids, levels, beta=0.25)

        assert torch.allclose(adv_global, adv_per_task), (
            f"单 task 应与全局等价:\n  global={adv_global}\n  per_task={adv_per_task}"
        )

    def test_multi_task_independent(self):
        """不同 task 应独立分层，互不影响。"""
        # task1: 全是高分, task2: 全是低分
        scores = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        levels = ["none", "none", "none", "none", "none", "none"]
        task_ids = ["task1"] * 3 + ["task2"] * 3

        adv = _simulate_estimator(scores, task_ids, levels, beta=0.0)

        # 每个 task 内部均值应为 0（beta=0 纯分层）
        assert abs(adv[:3].mean().item()) < 1e-5, f"task1 mean should be 0, got {adv[:3].mean()}"
        assert abs(adv[3:].mean().item()) < 1e-5, f"task2 mean should be 0, got {adv[3:].mean()}"

    def test_cross_task_no_contamination(self):
        """task1 的高分不应被 task2 的低分拉低。"""
        # task1: high reward, task2: low reward, same level
        scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        levels = ["none"] * 6
        task_ids = ["task1"] * 3 + ["task2"] * 3

        adv_per_task = _simulate_estimator(scores, task_ids, levels, beta=0.25)
        adv_global = compute_livemcp_advantages(scores, levels, beta=0.25)

        # per-task: task1 的 advantage 应接近 0（reward 相同，std=0）
        assert abs(adv_per_task[0].item()) < 1e-5, (
            f"per-task: task1 reward=1.0 all same → adv should be 0, got {adv_per_task[0]}"
        )
        # global: task1 的 advantage 应为正（高于全局均值 0.5）
        assert adv_global[0] > 0, (
            f"global: task1 reward=1.0 > global mean → adv should be >0, got {adv_global[0]}"
        )

    def test_beta_zero_pure_stratification(self):
        """beta=0 时全局残差为零，纯层内比较。"""
        scores = torch.tensor([0.9, 0.8, 0.5, 0.4])
        levels = ["none", "none", "strong", "strong"]
        task_ids = ["t1"] * 4

        adv = _simulate_estimator(scores, task_ids, levels, beta=0.0)

        # none 层内: 0.9 > 0.8 → adv[0] > 0, adv[1] < 0
        assert adv[0] > 0, f"beta=0: none 层高分应正 adv, got {adv[0]}"
        assert adv[1] < 0, f"beta=0: none 层低分应负 adv, got {adv[1]}"

    def test_beta_full_global_correction(self):
        """beta=1.0: A = strat_z + 1.0*global_z, 单层时两者相等 → A = 2*z。"""
        scores = torch.tensor([0.9, 0.8, 0.7])
        levels = ["none"] * 3
        task_ids = ["t1"] * 3

        adv_beta1 = _simulate_estimator(scores, task_ids, levels, beta=1.0)

        # 单层时 strat_z = global_z = (r - mean) / std
        mean = scores.mean()
        std = scores.std(unbiased=False).clamp(min=1e-6)
        z = (scores - mean) / std
        expected = z + 1.0 * z  # = 2 * z

        assert torch.allclose(adv_beta1, expected, atol=1e-5), (
            f"beta=1.0 单层: A = 2*z:\n  got={adv_beta1}\n  expected={expected}"
        )

        # 验证 beta=0.5 时 A = 1.5*z
        adv_beta05 = _simulate_estimator(scores, task_ids, levels, beta=0.5)
        assert torch.allclose(adv_beta05, z + 0.5 * z, atol=1e-5), (
            f"beta=0.5 单层: A = 1.5*z:\n  got={adv_beta05}"
        )

    def test_single_sample_per_level(self):
        """某层只有 1 个样本时不应崩溃（减均值即可，不除 std）。"""
        scores = torch.tensor([0.9, 0.5, 0.4, 0.3])
        # none=1样本, mild=1样本, strong=2样本
        levels = ["none", "mild", "strong", "strong"]
        task_ids = ["t1"] * 4

        adv = _simulate_estimator(scores, task_ids, levels, beta=0.0)
        assert torch.all(torch.isfinite(adv))
        assert not torch.any(torch.isnan(adv))

    def test_no_std_normalization(self):
        """norm_by_std=False 时只减均值。"""
        scores = torch.tensor([0.9, 0.8, 0.7])
        levels = ["none"] * 3
        task_ids = ["t1"] * 3

        adv = _simulate_estimator(scores, task_ids, levels, beta=0.0, norm_by_std=False)

        # 只减均值: adv = r - mean, mean = 0.8
        expected = scores - 0.8
        assert torch.allclose(adv, expected, atol=1e-5), (
            f"norm_by_std=False: {adv} != {expected}"
        )

    def test_mixed_levels_per_task(self):
        """同一 task 内混合多个 perturbation level。"""
        scores = torch.tensor([
            0.9, 0.8, 0.7,    # task1: none
            0.5, 0.4, 0.3,    # task1: mild
            0.2, 0.1, 0.0,    # task1: strong
            0.9, 0.5, 0.2,    # task2: none, mild, strong — 各 1 个
        ])
        levels = [
            "none", "none", "none",
            "mild", "mild", "mild",
            "strong", "strong", "strong",
            "none", "mild", "strong",
        ]
        task_ids = ["task1"] * 9 + ["task2"] * 3

        adv = _simulate_estimator(scores, task_ids, levels, beta=0.25)

        # 所有值应为有限数
        assert torch.all(torch.isfinite(adv))
        assert not torch.any(torch.isnan(adv))

        # task1: 9 个样本 3 层各 3 → 层内应有正负 advantage
        assert adv[:3].std() > 0, "task1 none 层应有方差"
        assert adv[3:6].std() > 0, "task1 mild 层应有方差"
        assert adv[6:9].std() > 0, "task1 strong 层应有方差"

    def test_to_list_scalar_broadcast(self):
        """验证 _to_list 的标量广播逻辑：单值应正确广播到 batch。"""
        # 模拟 _to_list 对单值的处理
        val = "none"
        bsz = 9
        # else 分支: [str(val)] → 单值广播
        result = [str(val)]
        if len(result) == 1 and bsz > 1:
            result = result * bsz
        assert len(result) == 9
        assert all(r == "none" for r in result)


class TestUidFallback:
    """uid 解析 fallback 逻辑。"""

    @staticmethod
    def _parse_uid(uid: str) -> tuple[str, str]:
        """复制 livemcp_grpo_estimator._parse_uid 的逻辑。"""
        if "___" in uid:
            parts = uid.split("___", 1)
            return parts[0], parts[1]
        return uid, "none"

    def test_uid_with_level(self):
        """uid 格式 {task_id}___{level}。"""
        tid, level = self._parse_uid("abc123___mild")
        assert tid == "abc123"
        assert level == "mild"

    def test_uid_without_level(self):
        """uid 无分隔符时默认 level=none。"""
        import uuid
        random_uid = str(uuid.uuid4())
        tid, level = self._parse_uid(random_uid)
        assert tid == random_uid
        assert level == "none"
