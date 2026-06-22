"""SchemaShift reward function — verl custom_reward_function 接口。

通过 verl config 的 custom_reward_function.path 指定本文件，
custom_reward_function.name 指定 "compute_score"。

接口签名：
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float | dict

verl 的 NaiveRewardManager 会对每个样本调用此函数，
将返回值放在 response 最后一个 token 的位置。
"""

import json
from typing import Any, Optional

from src.reward.action_parser import ActionParser
from src.reward.component_reward import ComponentReward, OracleAction, SampleMetadata


# 模块级单例（避免每次调用都重新创建）
_parser = ActionParser(strict=False)
_reward_fn = ComponentReward()

_COMPONENT_KEYS = (
    "format",
    "schema_valid",
    "tool_selection",
    "argument_keys",
    "argument_values",
    "no_extra_call",
    "final_answer_match",
    "error_type_match",
    "clarification_match",
)


def _reward_info(
    *,
    score: float,
    exact_success: bool = False,
    action_type_match: bool = False,
    oracle_action_type: str = "",
    model_action_type: str = "",
    components: Optional[dict[str, Any]] = None,
    error: str = "",
) -> dict[str, Any]:
    """Build a fixed-shape reward dict for verl validation aggregation."""
    components = components or {}
    info: dict[str, Any] = {
        "score": float(score),
        "exact_success": float(exact_success),
        "action_type_match": float(action_type_match),
        "oracle_action_type": oracle_action_type,
        "model_action_type": model_action_type,
        "error": error,
    }
    for name in _COMPONENT_KEYS:
        info[f"component_{name}"] = float(components.get(name, 0.0))
    return info


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """SchemaShift reward function。

    Args:
        data_source: 数据源标识（"schemashift"）
        solution_str: 模型生成的完整 response 文本
        ground_truth: oracle 信息（JSON 序列化的 dict）
            {
                "oracle_actions": [...],  # 每步的 OracleAction
                "episode_type": "call_only" | "call_then_final" | ...
            }
        extra_info: 额外信息
            {
                "perturbation_level": "none" | "light" | "medium" | "heavy",
                "name_map": {...},
                "enum_map": {...},
                "scenario_type": "...",
            }

    Returns:
        dict with "score" key (float) + scalar/string diagnostic keys.
        verl validation aggregates non-string diagnostics with np.mean(),
        so nested dict/list values must not be returned here.
    """
    extra_info = extra_info or {}

    # 解析 ground_truth
    if isinstance(ground_truth, str):
        try:
            oracle_data = json.loads(ground_truth)
        except json.JSONDecodeError:
            # ground_truth 不是 JSON → 无法计算 reward
            return _reward_info(score=0.0, error="ground_truth not valid JSON")
    elif isinstance(ground_truth, dict):
        oracle_data = ground_truth
    else:
        return _reward_info(score=0.0, error=f"unexpected ground_truth type: {type(ground_truth)}")

    oracle_actions = oracle_data.get("oracle_actions", [])
    if not oracle_actions:
        return _reward_info(score=0.0, error="no oracle_actions in ground_truth")

    # 构建 metadata
    metadata = SampleMetadata(
        name_map=extra_info.get("name_map", {}),
        enum_map=extra_info.get("enum_map", {}),
        perturbation_level=extra_info.get("perturbation_level", "none"),
        scenario_type=extra_info.get("scenario_type", oracle_data.get("episode_type", "")),
    )

    # 当前 GRPO 入口没有接 MCPToolsAgentLoop，只生成初始 prompt 的下一步动作；
    # 因此这里按 next-action reward 只评估第一个 oracle action。
    oracle_action_data = oracle_actions[0]
    oracle_action = OracleAction(
        action_type=oracle_action_data.get("action_type", "tool_call"),
        tool_calls=oracle_action_data.get("tool_calls", []),
        match_mode=oracle_action_data.get("match_mode", "set"),
        final_answer=oracle_action_data.get("final_answer") or "",
        error_info=oracle_action_data.get("error_info") or "",
    )

    # 计算 reward
    result = _reward_fn.compute(
        model_output=solution_str,
        oracle=oracle_action,
        metadata=metadata,
    )

    return _reward_info(
        score=result.total_reward,
        exact_success=result.exact_success,
        action_type_match=result.action_type_match,
        oracle_action_type=result.oracle_action_type,
        model_action_type=result.model_action_type,
        components=result.components,
    )
