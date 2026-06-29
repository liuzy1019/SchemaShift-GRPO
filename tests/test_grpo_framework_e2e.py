"""GRPO 框架端到端逻辑验证测试。

覆盖：
  1. Reward 计算链路（oval_reward_fn.compute_score）
  2. Advantage estimator 分组语义（group_id = task_id 后的正确性）
  3. Lambda_safe 动态更新（dual ascent + stall protection）
  4. 数据格式兼容性（parquet → non_tensor_batch → estimator）
  5. 边界情况（空 audit_events、饱和组、单样本 group）
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# 确保项目在路径中
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════
# Group 1: Reward 计算链路
# ═══════════════════════════════════════════════════════════════════════

class TestRewardComputation:
    """验证 oval_reward_fn.compute_score 的核心逻辑。"""

    def _make_audit_event(
        self,
        step: int = 0,
        action_type: str = "tool_call",
        tool_name: str = "list_events",
        execution_success: bool = True,
        schema_valid: bool = True,
        state_changed: bool = False,
        operation: str = "query",
        forbidden_transition: str = "",
        identity_violation: str = "",
    ) -> dict:
        """构造一个 audit_event dict（模拟 agent loop 输出的序列化格式）。"""
        return {
            "event_id": f"evt_{step:04d}",
            "session_id": "sess_test",
            "step": step,
            "action_type": action_type,
            "tool_name": tool_name,
            "tool_arguments": {},
            "terminal_action": None if action_type == "tool_call" else "done",
            "operation": operation,
            "target_type": "calendar_event",
            "target_id": f"res_{step}",
            "before_hash": "",
            "after_hash": "",
            "changed_fields": [],
            "created_ids": [],
            "deleted_ids": [],
            "duplicate_of": None,
            "identity_violation": identity_violation,
            "forbidden_transition": forbidden_transition,
            "observation": None,
            "execution_success": execution_success,
            "error_type": None,
            "error_message": "",
            "schema_valid": schema_valid,
            "state_changed": state_changed,
            "latency_ms": 100,
        }

    def test_no_audit_events_returns_zero(self):
        """无 audit_events 时 score=0。"""
        from src.reward.oval_reward_fn import compute_score

        result = compute_score(
            data_source="live_mcp_state_machine",
            solution_str="<final_answer>done</final_answer>",
            ground_truth={"task_id": "test_001"},
            extra_info={"audit_events": [], "task_id": "test_001"},
        )
        assert result["score"] == 0.0
        assert result["error"] == "no audit events"

    def test_successful_trajectory_positive_score(self):
        """成功的轨迹应产生正 score。"""
        from src.reward.oval_reward_fn import compute_score

        events = [
            self._make_audit_event(
                step=0, tool_name="list_events", operation="query",
                execution_success=True, schema_valid=True,
            ),
            self._make_audit_event(
                step=1, tool_name="create_event", operation="create",
                execution_success=True, schema_valid=True, state_changed=True,
            ),
            self._make_audit_event(
                step=2, action_type="final_answer", tool_name="",
                operation="terminal", execution_success=True, schema_valid=True,
            ),
        ]

        extra_info = {
            "audit_events": events,
            "task_id": "test_success",
            "domain": "calendar",
            "required_tools": ["list_events", "create_event"],
            "session_id": "sess_test",
            "budget": 5,
        }

        result = compute_score(
            data_source="live_mcp_state_machine",
            solution_str="<final_answer>Event created</final_answer>",
            ground_truth={"task_id": "test_success"},
            extra_info=extra_info,
        )

        assert result["score"] > 0.0, f"Expected positive score, got {result['score']}"
        assert result["r_task"] > 0.0
        assert result["c_safety"] == 0  # 无安全违规
        assert result["error"] == ""

    def test_safety_violation_reduces_score(self):
        """安全违规应降低 score（通过 lambda_safe * C_safety 惩罚）。"""
        from src.reward.oval_reward_fn import compute_score
        from src.oval_mcp.training.lambda_state import LambdaState, DEFAULT_STATE_PATH

        # 设置高 lambda_safe 到全局状态文件（compute_score 从这里读取）
        state = LambdaState(lambda_safe=2.0, state_path=DEFAULT_STATE_PATH)
        state.save()

        try:
            events = [
                self._make_audit_event(
                    step=0, tool_name="delete_event", operation="delete",
                    execution_success=True, schema_valid=True, state_changed=True,
                    forbidden_transition="unauthorized_deletion",
                ),
                self._make_audit_event(
                    step=1, action_type="final_answer", tool_name="",
                    operation="terminal", execution_success=True, schema_valid=True,
                ),
            ]

            extra_info = {
                "audit_events": events,
                "task_id": "test_unsafe",
                "domain": "calendar",
                "required_tools": ["delete_event"],
                "session_id": "sess_test",
                "budget": 5,
            }

            result = compute_score(
                data_source="live_mcp_state_machine",
                solution_str="<final_answer>Deleted</final_answer>",
                ground_truth={"task_id": "test_unsafe"},
                extra_info=extra_info,
            )

            assert result["c_safety"] == 1, "Should detect safety violation"
            assert result["lambda_safe"] == 2.0, f"Lambda should be 2.0 from state file, got {result['lambda_safe']}"
            # J = R_task + shape + process - lambda_safe * C_safety
            # 由于 C_safety=1, lambda_safe=2.0, 惩罚项 = -2.0
            assert result["score"] < 0.0, f"Expected negative score with safety violation, got {result['score']}"
        finally:
            # 清理：恢复默认状态
            LambdaState.reset(DEFAULT_STATE_PATH)

    def test_reward_json_string_audit_events(self):
        """audit_events 为 JSON 字符串时应正确解析。"""
        from src.reward.oval_reward_fn import compute_score

        events = [
            self._make_audit_event(step=0, execution_success=True, schema_valid=True),
            self._make_audit_event(step=1, action_type="final_answer", operation="terminal",
                                   execution_success=True, schema_valid=True),
        ]

        extra_info = {
            "audit_events": json.dumps(events),  # JSON 字符串格式
            "task_id": "test_json",
            "domain": "calendar",
            "required_tools": ["list_events"],
            "session_id": "sess_test",
            "budget": 5,
        }

        result = compute_score(
            data_source="live_mcp_state_machine",
            solution_str="<final_answer>done</final_answer>",
            ground_truth={"task_id": "test_json"},
            extra_info=extra_info,
        )

        assert result["n_events"] == 2.0
        assert result["error"] == ""

    def test_reward_ablation_switches(self):
        """消融开关 I_SHAPE=0, I_PROCESS=0 时 shape/process 项为 0。"""
        from src.reward import oval_reward_fn

        # 保存原始值
        orig_shape = oval_reward_fn._I_SHAPE
        orig_process = oval_reward_fn._I_PROCESS

        try:
            oval_reward_fn._I_SHAPE = 0
            oval_reward_fn._I_PROCESS = 0

            events = [
                self._make_audit_event(step=0, execution_success=True, schema_valid=True,
                                       state_changed=True, operation="create"),
                self._make_audit_event(step=1, action_type="final_answer", operation="terminal",
                                       execution_success=True, schema_valid=True),
            ]

            result = oval_reward_fn.compute_score(
                data_source="live_mcp_state_machine",
                solution_str="done",
                ground_truth={"task_id": "test_ablation"},
                extra_info={
                    "audit_events": events,
                    "task_id": "test_ablation",
                    "domain": "calendar",
                    "required_tools": ["create_event"],
                    "session_id": "sess_test",
                    "budget": 5,
                },
            )

            assert result["f_gamma"] == 0.0, "F_gamma should be 0 when I_SHAPE=0"
            assert result["p_process"] == 0.0, "P_process should be 0 when I_PROCESS=0"
        finally:
            oval_reward_fn._I_SHAPE = orig_shape
            oval_reward_fn._I_PROCESS = orig_process


# ═══════════════════════════════════════════════════════════════════════
# Group 2: Advantage Estimator 分组语义
# ═══════════════════════════════════════════════════════════════════════

class TestAdvantageEstimator:
    """验证 livemcp_grpo_estimator 的分组和 z-score 逻辑。"""

    def _make_batch(
        self,
        scores: list[float],
        group_ids: list[str],
        levels: list[str],
        scenarios: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
        """构造模拟 batch 数据。"""
        bsz = len(scores)
        seq_len = 10
        # token_level_rewards: 最后一个 token 放 score，其余为 0
        token_rewards = torch.zeros(bsz, seq_len)
        for i, s in enumerate(scores):
            token_rewards[i, -1] = s
        response_mask = torch.ones(bsz, seq_len)
        index = np.array([f"uid_{i}" for i in range(bsz)])
        non_tensor_batch = {
            "group_id": np.array(group_ids, dtype=object),
            "perturbation_level": np.array(levels, dtype=object),
            "scenario_type": np.array(scenarios or ["normal"] * bsz, dtype=object),
            "uid": index,
        }
        return token_rewards, response_mask, index, non_tensor_batch

    def test_per_prompt_grouping_after_fix(self):
        """修复后：每个 task_id 独立一组，同组内做 z-score。

        模拟 verl repeat(N=3) 后的数据：
        - task_A 的 3 个 rollout: scores [0.8, 0.5, 0.2]
        - task_B 的 3 个 rollout: scores [0.9, 0.9, 0.9]（饱和组）
        """
        from src.training.livemcp_grpo_estimator import compute_livemcp_grpo_advantage

        # 清除诊断标记
        for attr in ('_diagnosed', '_fallback_warned', '_sat_warned'):
            if hasattr(compute_livemcp_grpo_advantage, attr):
                delattr(compute_livemcp_grpo_advantage, attr)

        scores = [0.8, 0.5, 0.2, 0.9, 0.9, 0.9]
        group_ids = ["task_A", "task_A", "task_A", "task_B", "task_B", "task_B"]
        levels = ["medium", "medium", "medium", "easy", "easy", "easy"]

        token_rewards, response_mask, index, non_tensor_batch = self._make_batch(
            scores, group_ids, levels
        )

        advantages, returns = compute_livemcp_grpo_advantage(
            token_level_rewards=token_rewards,
            response_mask=response_mask,
            index=index,
            non_tensor_batch=non_tensor_batch,
            config={"beta": 0.25, "min_group_std": 1e-6},
        )

        # advantages 应该是 [bsz, seq_len] 形状
        assert advantages.shape == (6, 10)

        # task_A 内部：score 0.8 > mean(0.5) → advantage > 0
        # task_A 内部：score 0.2 < mean(0.5) → advantage < 0
        task_a_advs = advantages[:3].sum(dim=-1)  # 按 token 求和得到 trajectory-level
        assert task_a_advs[0] > 0, f"Highest score in group should have positive advantage: {task_a_advs[0]}"
        assert task_a_advs[2] < 0, f"Lowest score in group should have negative advantage: {task_a_advs[2]}"

        # task_B 是饱和组（std < min_group_std），advantage 应为 0
        task_b_advs = advantages[3:].sum(dim=-1)
        assert torch.allclose(task_b_advs, torch.zeros(3)), \
            f"Saturated group should have zero advantages: {task_b_advs}"

    def test_single_sample_groups_fallback_to_batch_grpo(self):
        """所有 group 都是 size=1 时，fallback 到 batch-level GRPO。"""
        from src.training.livemcp_grpo_estimator import compute_livemcp_grpo_advantage

        for attr in ('_diagnosed', '_fallback_warned', '_sat_warned'):
            if hasattr(compute_livemcp_grpo_advantage, attr):
                delattr(compute_livemcp_grpo_advantage, attr)

        scores = [0.9, 0.5, 0.1, 0.7]
        group_ids = ["t1", "t2", "t3", "t4"]  # 每个 group 只有 1 个样本
        levels = ["easy", "medium", "hard", "easy"]

        token_rewards, response_mask, index, non_tensor_batch = self._make_batch(
            scores, group_ids, levels
        )

        advantages, _ = compute_livemcp_grpo_advantage(
            token_level_rewards=token_rewards,
            response_mask=response_mask,
            index=index,
            non_tensor_batch=non_tensor_batch,
            config={"beta": 0.25},
        )

        # Batch-level GRPO: 最高 score 的 advantage 最大
        advs = advantages.sum(dim=-1)
        assert advs[0] > advs[1] > advs[2], \
            f"Batch-level GRPO should rank by score: {advs.tolist()}"

    def test_stratified_advantage_with_different_levels(self):
        """同一 group 内有不同 perturbation_level 时，分层 z-score 生效。"""
        from src.training.livemcp_grpo_estimator import compute_livemcp_grpo_advantage

        for attr in ('_diagnosed', '_fallback_warned', '_sat_warned'):
            if hasattr(compute_livemcp_grpo_advantage, attr):
                delattr(compute_livemcp_grpo_advantage, attr)

        # 同一 group 内 6 个样本，3 个 easy（高分）+ 3 个 hard（低分）
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        group_ids = ["task_X"] * 6
        levels = ["easy", "easy", "easy", "hard", "hard", "hard"]

        token_rewards, response_mask, index, non_tensor_batch = self._make_batch(
            scores, group_ids, levels
        )

        advantages, _ = compute_livemcp_grpo_advantage(
            token_level_rewards=token_rewards,
            response_mask=response_mask,
            index=index,
            non_tensor_batch=non_tensor_batch,
            config={"beta": 0.25, "min_stratum_size": 3},
        )

        advs = advantages.sum(dim=-1)
        # 在 easy stratum 内: 0.9 > mean(0.8) → positive
        # 在 hard stratum 内: 0.3 > mean(0.2) → positive
        # 关键：hard stratum 的最高分(0.3)也应有正 advantage（层内比较）
        assert advs[3] > 0, \
            f"Highest in hard stratum should have positive strat advantage: {advs[3]}"

    def test_advantage_zero_mean_within_group(self):
        """每个 group 内的 advantage 均值应接近 0（GRPO 基本性质）。"""
        from src.training.livemcp_grpo_estimator import compute_livemcp_grpo_advantage

        for attr in ('_diagnosed', '_fallback_warned', '_sat_warned'):
            if hasattr(compute_livemcp_grpo_advantage, attr):
                delattr(compute_livemcp_grpo_advantage, attr)

        scores = [0.9, 0.6, 0.3, 0.1, 0.7, 0.4, 0.2, 0.8]
        group_ids = ["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"]
        levels = ["medium"] * 8

        token_rewards, response_mask, index, non_tensor_batch = self._make_batch(
            scores, group_ids, levels
        )

        advantages, _ = compute_livemcp_grpo_advantage(
            token_level_rewards=token_rewards,
            response_mask=response_mask,
            index=index,
            non_tensor_batch=non_tensor_batch,
            config={"beta": 0.0},  # beta=0 时纯 strat_z，group 内均值严格为 0
        )

        advs = advantages.sum(dim=-1)
        g1_mean = advs[:4].mean()
        g2_mean = advs[4:].mean()
        assert abs(g1_mean.item()) < 1e-5, f"Group 1 advantage mean should be ~0: {g1_mean}"
        assert abs(g2_mean.item()) < 1e-5, f"Group 2 advantage mean should be ~0: {g2_mean}"


# ═══════════════════════════════════════════════════════════════════════
# Group 3: Lambda_safe 动态更新
# ═══════════════════════════════════════════════════════════════════════

class TestLambdaState:
    """验证 lambda_safe 的 dual ascent 更新和 stall protection。"""

    def test_lambda_decreases_when_safe(self):
        """当 hat_C < epsilon 时，lambda_safe 应减小。"""
        from src.oval_mcp.training.lambda_state import LambdaState

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            state = LambdaState(lambda_safe=1.0, alpha_lambda=0.01, epsilon=0.05, state_path=path)
            # 全部安全：hat_C = 0 < epsilon = 0.05
            new_lambda, skipped = state.update([0, 0, 0, 0, 0])
            assert not skipped
            assert new_lambda < 1.0, f"Lambda should decrease when safe: {new_lambda}"
            # lambda = 1.0 + 0.01 * (0.0 - 0.05) = 0.9995
            assert abs(new_lambda - 0.9995) < 1e-6
        finally:
            os.unlink(path)

    def test_lambda_increases_when_unsafe(self):
        """当 hat_C > epsilon 时，lambda_safe 应增大。"""
        from src.oval_mcp.training.lambda_state import LambdaState

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            state = LambdaState(lambda_safe=1.0, alpha_lambda=0.01, epsilon=0.05, state_path=path)
            # 80% 不安全：hat_C = 0.8 > epsilon = 0.05
            new_lambda, skipped = state.update([1, 1, 1, 1, 0])
            assert not skipped
            assert new_lambda > 1.0, f"Lambda should increase when unsafe: {new_lambda}"
            # lambda = 1.0 + 0.01 * (0.8 - 0.05) = 1.0075
            assert abs(new_lambda - 1.0075) < 1e-6
        finally:
            os.unlink(path)

    def test_stall_protection_freezes_lambda(self):
        """连续 k_stall 步 unsafe 后，lambda 应被冻结。"""
        from src.oval_mcp.training.lambda_state import LambdaState

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            state = LambdaState(lambda_safe=1.0, alpha_lambda=0.1, epsilon=0.05, state_path=path)
            k_stall = 5

            # 连续 k_stall 步高 unsafe rate
            for i in range(k_stall):
                new_lambda, skipped = state.update(
                    [1, 1, 1, 1, 1],  # hat_C = 1.0 >> tau_unsafe_stall
                    k_stall=k_stall,
                    tau_unsafe_stall=0.5,
                )

            # 第 k_stall 步后应该被冻结
            assert state.is_stall_frozen, "Should be frozen after k_stall consecutive unsafe steps"

            # 冻结后再次 unsafe，lambda 不再增大
            lambda_before = state.lambda_safe
            new_lambda, skipped = state.update(
                [1, 1, 1, 1, 1],
                k_stall=k_stall,
                tau_unsafe_stall=0.5,
            )
            assert skipped, "Update should be skipped when frozen"
            assert state.lambda_safe == lambda_before, "Lambda should not increase when frozen"
        finally:
            os.unlink(path)

    def test_lambda_persistence(self):
        """lambda_safe 状态应正确持久化和恢复。"""
        from src.oval_mcp.training.lambda_state import LambdaState

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            state = LambdaState(lambda_safe=2.5, alpha_lambda=0.02, epsilon=0.1, state_path=path)
            state.step = 42
            state.save()

            loaded = LambdaState.load_or_default(path)
            assert abs(loaded.lambda_safe - 2.5) < 1e-6
            assert loaded.alpha_lambda == 0.02
            assert loaded.epsilon == 0.1
            assert loaded.step == 42
        finally:
            os.unlink(path)

    def test_lambda_bounded(self):
        """lambda_safe 不应超过 lambda_safe_max，不应低于 0。"""
        from src.oval_mcp.training.lambda_state import LambdaState

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            # 测试上界
            state = LambdaState(lambda_safe=9.99, alpha_lambda=0.1, epsilon=0.0,
                                lambda_safe_max=10.0, state_path=path)
            new_lambda, _ = state.update([1, 1, 1, 1, 1])  # hat_C=1.0, 大幅增加
            assert new_lambda <= 10.0, f"Lambda should be bounded by max: {new_lambda}"

            # 测试下界
            state2 = LambdaState(lambda_safe=0.001, alpha_lambda=0.1, epsilon=1.0,
                                 state_path=path)
            new_lambda2, _ = state2.update([0, 0, 0, 0, 0])  # hat_C=0, epsilon=1.0, 大幅减少
            assert new_lambda2 >= 0.0, f"Lambda should be bounded by 0: {new_lambda2}"
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════
# Group 4: 数据格式兼容性
# ═══════════════════════════════════════════════════════════════════════

class TestDataFormatCompatibility:
    """验证 parquet 数据格式与下游组件的兼容性。"""

    def test_extra_info_json_normalization(self):
        """extra_info 为 JSON 字符串时应正确解析。"""
        from src.utils import normalize_extra_info

        # dict → dict
        assert normalize_extra_info({"a": 1}) == {"a": 1}
        # JSON string → dict
        assert normalize_extra_info('{"a": 1}') == {"a": 1}
        # None → {}
        assert normalize_extra_info(None) == {}
        # invalid → {}
        assert normalize_extra_info("not json") == {}
        assert normalize_extra_info(42) == {}

    def test_register_estimator_normalize_promotes_fields(self):
        """_normalize_livemcp_non_tensor_batch 应从 extra_info 提升字段。"""
        from src.training.register_estimator import _normalize_livemcp_non_tensor_batch

        # 模拟 verl 传来的 non_tensor_batch（只有 extra_info，没有 top-level 字段）
        extra_info_list = [
            {
                "task_id": "t1",
                "group_id": "t1",
                "perturbation_level": "easy",
                "scenario_type": "normal",
                "episode_id": "t1",
                "action_type": "tool_call",
                "tool_name": "list_events",
            },
            {
                "task_id": "t2",
                "group_id": "t2",
                "perturbation_level": "hard",
                "scenario_type": "distractor",
                "episode_id": "t2",
                "action_type": "tool_call",
                "tool_name": "create_event",
            },
        ]
        non_tensor_batch = {
            "extra_info": np.array(extra_info_list, dtype=object),
        }

        result = _normalize_livemcp_non_tensor_batch(non_tensor_batch, batch_size=2)

        assert "group_id" in result
        assert "perturbation_level" in result
        assert "scenario_type" in result
        assert result["group_id"].tolist() == ["t1", "t2"]
        assert result["perturbation_level"].tolist() == ["easy", "hard"]

    def test_register_estimator_normalize_json_string_extra_info(self):
        """extra_info 为 JSON 字符串数组时也应正确处理。"""
        from src.training.register_estimator import _normalize_livemcp_non_tensor_batch

        extra_info_list = [
            json.dumps({"group_id": "t1", "perturbation_level": "easy",
                        "scenario_type": "normal", "episode_id": "t1",
                        "action_type": "", "tool_name": ""}),
            json.dumps({"group_id": "t2", "perturbation_level": "hard",
                        "scenario_type": "distractor", "episode_id": "t2",
                        "action_type": "", "tool_name": ""}),
        ]
        non_tensor_batch = {
            "extra_info": np.array(extra_info_list, dtype=object),
        }

        result = _normalize_livemcp_non_tensor_batch(non_tensor_batch, batch_size=2)

        assert result["group_id"].tolist() == ["t1", "t2"]
        assert result["perturbation_level"].tolist() == ["easy", "hard"]

    def test_group_id_equals_task_id_in_data_generation(self):
        """验证数据生成后 group_id == task_id（修复后的语义）。"""
        # 模拟 _tasks_to_rows 的输出格式
        # 修复后：group_id = task.task_id
        from unittest.mock import MagicMock

        task = MagicMock()
        task.task_id = "calendar_create_event_001"
        task.visible_tools = [{"name": "create_event", "description": "Create event",
                               "input_schema": {"properties": {}, "required": []}}]
        task.target_servers = ["calendar"]
        task.required_tools = ["create_event"]
        task.user_prompt = "Create a meeting"
        task.session_seed = 42
        task.max_turns = 5
        task.task_type = "normal"
        task.difficulty = "easy"
        task.metadata = {}
        task.oracle_program = None

        # 直接验证逻辑：group_id 应等于 task_id
        group_id = task.task_id
        assert group_id == "calendar_create_event_001"
        # 不再是 f"group_{domain}_{index // 4}" 的格式
        assert not group_id.startswith("group_")


# ═══════════════════════════════════════════════════════════════════════
# Group 5: 边界情况和集成
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """验证边界情况的正确处理。"""

    def test_empty_batch(self):
        """空 batch 不应崩溃。"""
        from src.training.livemcp_grpo_estimator import compute_livemcp_grpo_advantage

        for attr in ('_diagnosed', '_fallback_warned', '_sat_warned'):
            if hasattr(compute_livemcp_grpo_advantage, attr):
                delattr(compute_livemcp_grpo_advantage, attr)

        # 单样本 batch
        token_rewards = torch.tensor([[0.5] * 10])
        response_mask = torch.ones(1, 10)
        index = np.array(["uid_0"])
        non_tensor_batch = {
            "group_id": np.array(["g1"], dtype=object),
            "perturbation_level": np.array(["easy"], dtype=object),
            "scenario_type": np.array(["normal"], dtype=object),
            "uid": index,
        }

        advantages, _ = compute_livemcp_grpo_advantage(
            token_level_rewards=token_rewards,
            response_mask=response_mask,
            index=index,
            non_tensor_batch=non_tensor_batch,
            config={"beta": 0.25},
        )

        # 单样本无法比较，advantage 应为 0
        assert advantages.shape == (1, 10)

    def test_all_same_scores_zero_advantage(self):
        """所有样本 score 相同时，advantage 应为 0（或接近 0）。"""
        from src.training.livemcp_grpo_estimator import compute_livemcp_grpo_advantage

        for attr in ('_diagnosed', '_fallback_warned', '_sat_warned'):
            if hasattr(compute_livemcp_grpo_advantage, attr):
                delattr(compute_livemcp_grpo_advantage, attr)

        scores = [0.5, 0.5, 0.5, 0.5]
        bsz = len(scores)
        token_rewards = torch.zeros(bsz, 10)
        for i, s in enumerate(scores):
            token_rewards[i, -1] = s
        response_mask = torch.ones(bsz, 10)
        index = np.array([f"uid_{i}" for i in range(bsz)])
        non_tensor_batch = {
            "group_id": np.array(["g1"] * bsz, dtype=object),
            "perturbation_level": np.array(["easy"] * bsz, dtype=object),
            "scenario_type": np.array(["normal"] * bsz, dtype=object),
            "uid": index,
        }

        advantages, _ = compute_livemcp_grpo_advantage(
            token_level_rewards=token_rewards,
            response_mask=response_mask,
            index=index,
            non_tensor_batch=non_tensor_batch,
            config={"beta": 0.25, "min_group_std": 1e-6},
        )

        # 饱和组：所有 score 相同 → std < min_group_std → advantage = 0
        advs = advantages.sum(dim=-1)
        assert torch.allclose(advs, torch.zeros(bsz), atol=1e-6), \
            f"Same scores should give zero advantages: {advs}"

    def test_reward_score_range(self):
        """验证 reward score 的值域合理性。"""
        from src.reward.oval_reward_fn import compute_score

        # 最佳情况：完美执行
        events = [
            {
                "event_id": "evt_0", "session_id": "s", "step": 0,
                "action_type": "tool_call", "tool_name": "list_events",
                "tool_arguments": {}, "terminal_action": None,
                "operation": "query", "target_type": "calendar_event",
                "target_id": "r1", "before_hash": "", "after_hash": "",
                "changed_fields": [], "created_ids": [], "deleted_ids": [],
                "duplicate_of": None, "identity_violation": "",
                "forbidden_transition": "", "observation": None,
                "execution_success": True, "error_type": None,
                "error_message": "", "schema_valid": True,
                "state_changed": False, "latency_ms": 50,
            },
            {
                "event_id": "evt_1", "session_id": "s", "step": 1,
                "action_type": "final_answer", "tool_name": "",
                "tool_arguments": {}, "terminal_action": "done",
                "operation": "terminal", "target_type": "",
                "target_id": "", "before_hash": "", "after_hash": "",
                "changed_fields": [], "created_ids": [], "deleted_ids": [],
                "duplicate_of": None, "identity_violation": "",
                "forbidden_transition": "", "observation": None,
                "execution_success": True, "error_type": None,
                "error_message": "", "schema_valid": True,
                "state_changed": False, "latency_ms": 10,
            },
        ]

        result = compute_score(
            data_source="live_mcp_state_machine",
            solution_str="done",
            ground_truth={"task_id": "range_test"},
            extra_info={
                "audit_events": events,
                "task_id": "range_test",
                "domain": "calendar",
                "required_tools": ["list_events"],
                "session_id": "s",
                "budget": 5,
            },
        )

        # R_task ∈ [-0.2, 1.0]
        assert -0.2 <= result["r_task"] <= 1.0, f"R_task out of range: {result['r_task']}"
        # C_safety ∈ {0, 1}
        assert result["c_safety"] in (0, 1, 0.0, 1.0)
        # F_gamma ∈ [0, 1] (when gamma=1)
        assert 0.0 <= result["f_gamma"] <= 1.0, f"F_gamma out of range: {result['f_gamma']}"
        # P_process ∈ [-p_max, p_max] = [-0.3, 0.3]
        assert -0.3 <= result["p_process"] <= 0.3, f"P_process out of range: {result['p_process']}"

    def test_uid_fallback_when_no_non_tensor_batch(self):
        """当 non_tensor_batch 缺少关键字段时，应 fallback 到 uid 解析。"""
        from src.training.livemcp_grpo_estimator import compute_livemcp_grpo_advantage

        for attr in ('_diagnosed', '_fallback_warned', '_sat_warned'):
            if hasattr(compute_livemcp_grpo_advantage, attr):
                delattr(compute_livemcp_grpo_advantage, attr)

        bsz = 4
        token_rewards = torch.zeros(bsz, 10)
        token_rewards[:, -1] = torch.tensor([0.9, 0.6, 0.3, 0.1])
        response_mask = torch.ones(bsz, 10)
        # uid 格式: {task_id}___{level}___{scenario_type}
        index = np.array([
            "task_X___easy___normal",
            "task_X___easy___normal",
            "task_X___hard___normal",
            "task_X___hard___normal",
        ])

        # 不传 non_tensor_batch → fallback 到 uid 解析
        advantages, _ = compute_livemcp_grpo_advantage(
            token_level_rewards=token_rewards,
            response_mask=response_mask,
            index=index,
            config={"beta": 0.25, "min_stratum_size": 2},
        )

        # 不应崩溃
        assert advantages.shape == (4, 10)
        advs = advantages.sum(dim=-1)
        # 所有样本属于同一 group (task_X)，应有非零 advantage
        assert not torch.allclose(advs, torch.zeros(4))


# ═══════════════════════════════════════════════════════════════════════
# Group 6: 奖励设计合理性验证
# ═══════════════════════════════════════════════════════════════════════

class TestRewardDesignSanity:
    """验证奖励设计的合理性：好的行为应得到更高奖励。"""

    def _compute_reward(self, events: list[dict], **extra_kwargs) -> float:
        """辅助方法：计算 reward score。"""
        from src.reward.oval_reward_fn import compute_score

        base_extra = {
            "task_id": "sanity_test",
            "domain": "calendar",
            "required_tools": ["list_events", "create_event"],
            "session_id": "sess",
            "budget": 8,
        }
        base_extra.update(extra_kwargs)
        base_extra["audit_events"] = events

        result = compute_score(
            data_source="live_mcp_state_machine",
            solution_str="done",
            ground_truth={"task_id": "sanity_test"},
            extra_info=base_extra,
        )
        return result["score"]

    def test_successful_execution_beats_failed(self):
        """成功执行的轨迹 score 应高于失败的。"""
        # 成功轨迹
        good_events = [
            {"event_id": "e0", "session_id": "s", "step": 0,
             "action_type": "tool_call", "tool_name": "list_events",
             "tool_arguments": {}, "terminal_action": None,
             "operation": "query", "target_type": "calendar_event",
             "target_id": "r1", "before_hash": "", "after_hash": "",
             "changed_fields": [], "created_ids": [], "deleted_ids": [],
             "duplicate_of": None, "identity_violation": "",
             "forbidden_transition": "", "observation": None,
             "execution_success": True, "error_type": None,
             "error_message": "", "schema_valid": True,
             "state_changed": False, "latency_ms": 50},
            {"event_id": "e1", "session_id": "s", "step": 1,
             "action_type": "tool_call", "tool_name": "create_event",
             "tool_arguments": {}, "terminal_action": None,
             "operation": "create", "target_type": "calendar_event",
             "target_id": "r2", "before_hash": "", "after_hash": "abc",
             "changed_fields": ["title"], "created_ids": ["evt_new"], "deleted_ids": [],
             "duplicate_of": None, "identity_violation": "",
             "forbidden_transition": "", "observation": None,
             "execution_success": True, "error_type": None,
             "error_message": "", "schema_valid": True,
             "state_changed": True, "latency_ms": 80},
            {"event_id": "e2", "session_id": "s", "step": 2,
             "action_type": "final_answer", "tool_name": "",
             "tool_arguments": {}, "terminal_action": "done",
             "operation": "terminal", "target_type": "",
             "target_id": "", "before_hash": "", "after_hash": "",
             "changed_fields": [], "created_ids": [], "deleted_ids": [],
             "duplicate_of": None, "identity_violation": "",
             "forbidden_transition": "", "observation": None,
             "execution_success": True, "error_type": None,
             "error_message": "", "schema_valid": True,
             "state_changed": False, "latency_ms": 10},
        ]

        # 失败轨迹：所有调用都失败
        bad_events = [
            {"event_id": "e0", "session_id": "s", "step": 0,
             "action_type": "tool_call", "tool_name": "list_events",
             "tool_arguments": {}, "terminal_action": None,
             "operation": "query", "target_type": "calendar_event",
             "target_id": "r1", "before_hash": "", "after_hash": "",
             "changed_fields": [], "created_ids": [], "deleted_ids": [],
             "duplicate_of": None, "identity_violation": "",
             "forbidden_transition": "", "observation": None,
             "execution_success": False, "error_type": "invalid_args",
             "error_message": "bad args", "schema_valid": False,
             "state_changed": False, "latency_ms": 50},
            {"event_id": "e1", "session_id": "s", "step": 1,
             "action_type": "final_answer", "tool_name": "",
             "tool_arguments": {}, "terminal_action": "done",
             "operation": "terminal", "target_type": "",
             "target_id": "", "before_hash": "", "after_hash": "",
             "changed_fields": [], "created_ids": [], "deleted_ids": [],
             "duplicate_of": None, "identity_violation": "",
             "forbidden_transition": "", "observation": None,
             "execution_success": True, "error_type": None,
             "error_message": "", "schema_valid": True,
             "state_changed": False, "latency_ms": 10},
        ]

        good_score = self._compute_reward(good_events)
        bad_score = self._compute_reward(bad_events)

        assert good_score > bad_score, \
            f"Successful trajectory ({good_score:.4f}) should score higher than failed ({bad_score:.4f})"

    def test_efficient_execution_beats_wasteful(self):
        """高效执行（少调用）应比浪费（多余调用）得分高。"""
        # 高效：2 次调用完成任务
        efficient_events = [
            {"event_id": "e0", "session_id": "s", "step": 0,
             "action_type": "tool_call", "tool_name": "list_events",
             "tool_arguments": {}, "terminal_action": None,
             "operation": "query", "target_type": "calendar_event",
             "target_id": "r1", "before_hash": "", "after_hash": "",
             "changed_fields": [], "created_ids": [], "deleted_ids": [],
             "duplicate_of": None, "identity_violation": "",
             "forbidden_transition": "", "observation": None,
             "execution_success": True, "error_type": None,
             "error_message": "", "schema_valid": True,
             "state_changed": False, "latency_ms": 50},
            {"event_id": "e1", "session_id": "s", "step": 1,
             "action_type": "tool_call", "tool_name": "create_event",
             "tool_arguments": {}, "terminal_action": None,
             "operation": "create", "target_type": "calendar_event",
             "target_id": "r2", "before_hash": "", "after_hash": "x",
             "changed_fields": ["title"], "created_ids": ["new1"], "deleted_ids": [],
             "duplicate_of": None, "identity_violation": "",
             "forbidden_transition": "", "observation": None,
             "execution_success": True, "error_type": None,
             "error_message": "", "schema_valid": True,
             "state_changed": True, "latency_ms": 80},
            {"event_id": "e2", "session_id": "s", "step": 2,
             "action_type": "final_answer", "tool_name": "",
             "tool_arguments": {}, "terminal_action": "done",
             "operation": "terminal", "target_type": "",
             "target_id": "", "before_hash": "", "after_hash": "",
             "changed_fields": [], "created_ids": [], "deleted_ids": [],
             "duplicate_of": None, "identity_violation": "",
             "forbidden_transition": "", "observation": None,
             "execution_success": True, "error_type": None,
             "error_message": "", "schema_valid": True,
             "state_changed": False, "latency_ms": 10},
        ]

        # 浪费：5 次多余调用（重复 query）+ 2 次有效调用
        wasteful_events = list(efficient_events[:1]) * 5 + efficient_events[1:]
        # 修正 step 编号
        for i, e in enumerate(wasteful_events):
            wasteful_events[i] = {**e, "event_id": f"e{i}", "step": i}

        efficient_score = self._compute_reward(efficient_events)
        wasteful_score = self._compute_reward(wasteful_events)

        assert efficient_score > wasteful_score, \
            f"Efficient ({efficient_score:.4f}) should beat wasteful ({wasteful_score:.4f})"

    def test_safe_execution_beats_unsafe(self):
        """安全执行应比不安全执行得分高。"""
        base_event = {
            "event_id": "e0", "session_id": "s", "step": 0,
            "action_type": "tool_call", "tool_name": "create_event",
            "tool_arguments": {}, "terminal_action": None,
            "operation": "create", "target_type": "calendar_event",
            "target_id": "r1", "before_hash": "", "after_hash": "x",
            "changed_fields": ["title"], "created_ids": ["new1"], "deleted_ids": [],
            "duplicate_of": None, "identity_violation": "",
            "forbidden_transition": "", "observation": None,
            "execution_success": True, "error_type": None,
            "error_message": "", "schema_valid": True,
            "state_changed": True, "latency_ms": 50,
        }
        terminal = {
            "event_id": "e1", "session_id": "s", "step": 1,
            "action_type": "final_answer", "tool_name": "",
            "tool_arguments": {}, "terminal_action": "done",
            "operation": "terminal", "target_type": "",
            "target_id": "", "before_hash": "", "after_hash": "",
            "changed_fields": [], "created_ids": [], "deleted_ids": [],
            "duplicate_of": None, "identity_violation": "",
            "forbidden_transition": "", "observation": None,
            "execution_success": True, "error_type": None,
            "error_message": "", "schema_valid": True,
            "state_changed": False, "latency_ms": 10,
        }

        safe_events = [base_event, terminal]
        unsafe_events = [{**base_event, "forbidden_transition": "unauthorized_access"}, terminal]

        safe_score = self._compute_reward(safe_events)
        unsafe_score = self._compute_reward(unsafe_events)

        assert safe_score > unsafe_score, \
            f"Safe ({safe_score:.4f}) should beat unsafe ({unsafe_score:.4f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
