"""TrajectoryVerifier — 对完整 trajectory 计算 step-level + trajectory-level reward。

职责：
  1. 接收 MCPToolEnvironment.get_reward_info() 的输出
  2. 对每个 step 调用 ComponentReward 计算 step-level reward
  3. 计算 trajectory-level signals（no_extra_call, all_steps_exact）
  4. 输出 token_level_rewards tensor（与 response tokens 对齐）

与 verl 的集成点：
  - verl 的 reward function 调用 verifier.compute_rewards(reward_info, response_ids, response_mask)
  - 返回 token_level_rewards tensor，形状 [seq_len]
"""

from typing import Any, Optional

import torch

from src.reward.component_reward import ComponentReward, OracleAction, RewardResult, SampleMetadata


class TrajectoryVerifier:
    """Trajectory-level reward 计算器。

    Usage:
        verifier = TrajectoryVerifier()
        token_rewards = verifier.compute_rewards(reward_info, response_mask)
    """

    def __init__(
        self,
        reward_fn: Optional[ComponentReward] = None,
        # trajectory-level 权重
        trajectory_bonus: float = 0.1,
        truncation_penalty: float = -0.2,
        # reward 分配策略
        reward_placement: str = "last_token",  # "last_token" | "uniform" | "eos"
    ):
        self.reward_fn = reward_fn or ComponentReward()
        self.trajectory_bonus = trajectory_bonus
        self.truncation_penalty = truncation_penalty
        self.reward_placement = reward_placement

    def compute_rewards(
        self,
        reward_info: dict[str, Any],
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算 token-level rewards。

        Args:
            reward_info: MCPToolEnvironment.get_reward_info() 的输出。
            response_mask: [seq_len] 的 0/1 mask，标记 response tokens。

        Returns:
            token_level_rewards: [seq_len] tensor。
        """
        seq_len = response_mask.shape[0]
        token_rewards = torch.zeros(seq_len, dtype=torch.float32, device=response_mask.device)

        step_infos = reward_info.get("step_infos", [])
        if not step_infos:
            # 无 step 但仍需计算 trajectory-level reward（如 truncation_penalty）
            trajectory_reward = self._compute_trajectory_reward([], reward_info)
            if trajectory_reward != 0.0:
                return self._place_reward(trajectory_reward, response_mask)
            return token_rewards

        # 计算 step-level rewards
        step_rewards = []
        step_results = []
        for si in step_infos:
            if si.oracle_action is not None and si.parsed_action is not None:
                result = self.reward_fn.compute(
                    model_output=si.model_output,
                    oracle=si.oracle_action,
                    metadata=si.metadata,
                )
                step_rewards.append(result.total_reward)
                step_results.append(result)
            else:
                # 无 oracle 对照（超出 trace 的额外步骤）→ 0 reward
                step_rewards.append(0.0)
                step_results.append(None)

        # 计算 trajectory-level signals
        trajectory_reward = self._compute_trajectory_reward(
            step_results, reward_info
        )

        # 合并：step rewards 的均值 + trajectory bonus/penalty
        if step_rewards:
            total_reward = sum(step_rewards) / len(step_rewards) + trajectory_reward
        else:
            total_reward = trajectory_reward

        # 分配到 token level
        token_rewards = self._place_reward(total_reward, response_mask)

        return token_rewards

    def compute_batch_rewards(
        self,
        reward_infos: list[dict[str, Any]],
        response_masks: torch.Tensor,
    ) -> torch.Tensor:
        """批量计算 token-level rewards。

        Args:
            reward_infos: 每个样本的 reward_info 列表。
            response_masks: [batch_size, seq_len] 的 mask。

        Returns:
            token_level_rewards: [batch_size, seq_len] tensor。
        """
        batch_size, seq_len = response_masks.shape
        all_rewards = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=response_masks.device)

        for i, (info, mask) in enumerate(zip(reward_infos, response_masks)):
            all_rewards[i] = self.compute_rewards(info, mask)

        return all_rewards

    def _compute_trajectory_reward(
        self,
        step_results: list[Optional[RewardResult]],
        reward_info: dict[str, Any],
    ) -> float:
        """计算 trajectory-level reward。"""
        reward = 0.0

        # 截断惩罚
        if reward_info.get("truncated", False):
            reward += self.truncation_penalty

        # all_steps_exact bonus
        valid_results = [r for r in step_results if r is not None]
        all_exact = (
            bool(step_results)
            and len(valid_results) == len(step_results)
            and all(r.exact_success for r in valid_results)
        )
        if all_exact and step_results:
            reward += self.trajectory_bonus

        # no_extra_call：模型步数 <= oracle 步数
        model_steps = reward_info.get("total_steps", 0)
        oracle_steps = reward_info.get("oracle_total_steps", 0)
        if model_steps <= oracle_steps and not reward_info.get("truncated", False):
            reward += self.trajectory_bonus * 0.5

        return reward

    def _place_reward(
        self,
        total_reward: float,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """将 scalar reward 分配到 token level。"""
        seq_len = response_mask.shape[0]
        token_rewards = torch.zeros(seq_len, dtype=torch.float32, device=response_mask.device)

        # 找到 response tokens 的位置
        response_positions = response_mask.nonzero(as_tuple=True)[0]
        if len(response_positions) == 0:
            return token_rewards

        if self.reward_placement == "last_token":
            # 放在最后一个 response token 上
            last_pos = response_positions[-1].item()
            token_rewards[last_pos] = total_reward

        elif self.reward_placement == "uniform":
            # 均匀分配到所有 response tokens
            n_tokens = len(response_positions)
            per_token = total_reward / n_tokens
            token_rewards[response_positions] = per_token

        elif self.reward_placement == "eos":
            # 放在最后一个 response token（同 last_token）
            last_pos = response_positions[-1].item()
            token_rewards[last_pos] = total_reward

        else:
            # 默认 last_token
            last_pos = response_positions[-1].item()
            token_rewards[last_pos] = total_reward

        return token_rewards

    def get_diagnostics(
        self,
        reward_info: dict[str, Any],
    ) -> dict[str, Any]:
        """获取诊断信息（用于 logging）。"""
        step_infos = reward_info.get("step_infos", [])
        diagnostics = {
            "episode_id": reward_info.get("episode_id", ""),
            "total_steps": len(step_infos),
            "oracle_total_steps": reward_info.get("oracle_total_steps", 0),
            "truncated": reward_info.get("truncated", False),
            "all_matched": reward_info.get("all_matched", False),
            "perturbation_level": reward_info.get("perturbation_level", "none"),
            "scenario_type": reward_info.get("scenario_type", ""),
        }

        # 每步的 action type 和 match 状态
        step_details = []
        for si in step_infos:
            detail = {
                "step": si.step_idx,
                "model_action_type": si.parsed_action.action_type if si.parsed_action else "none",
                "oracle_action_type": si.oracle_step.action_type if si.oracle_step else "none",
                "matched": si.execution_result.matched if si.execution_result else False,
            }
            step_details.append(detail)
        diagnostics["steps"] = step_details

        return diagnostics
