"""组件化 Reward 计算器。

实现 mcp_tools_rl_project_plan.md §11 的 reward 设计：
  - Step-level 5 组件：format_valid / schema_valid / tool_selection / argument_keys / argument_values
  - Trajectory-level signals：no_extra_call / final_answer_match / all_required_steps_exact
  - Action-type matrix（§11.1）
  - Correctness floor gate

用法:
    from src.reward.component_reward import ComponentReward
    reward_fn = ComponentReward()
    result = reward_fn.compute(model_output, oracle, metadata)
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from src.reward.action_parser import ActionParser, ParsedAction, parse_action


@dataclass
class RewardResult:
    """Reward 计算结果。"""

    total_reward: float  # effective_reward（经过 correctness floor）
    components: dict[str, float] = field(default_factory=dict)  # 各组件原始分
    exact_success: bool = False  # 是否精确匹配 oracle
    action_type_match: bool = False  # oracle 和 model 的 action type 是否一致
    oracle_action_type: str = ""  # oracle 的 action type
    model_action_type: str = ""  # model 的 action type
    diagnostics: dict[str, Any] = field(default_factory=dict)  # 诊断信息


@dataclass
class OracleAction:
    """Oracle（ground truth）action。"""

    action_type: str  # "tool_call" / "final_answer" / "ask_clarification" / "report_error"
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": ..., "arguments": ...}]
    final_answer: str = ""  # final_answer 内容
    error_info: str = ""  # report_error 内容
    match_mode: str = "set"  # "set"（无序匹配）或 "ordered"（顺序匹配）


@dataclass
class SampleMetadata:
    """样本元数据，用于 reward 计算。"""

    name_map: dict[str, str] = field(default_factory=dict)  # perturbed_name -> canonical_name
    enum_map: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)  # tool -> param -> perturbed -> canonical
    perturbation_level: str = "none"
    scenario_type: str = "single_step"


class ComponentReward:
    """组件化 reward 计算器。

    权重配置（mcp_tools_rl_project_plan.md §11）：
        format_valid:     0.10
        schema_valid:     0.15
        tool_selection:   0.30
        argument_keys:    0.20
        argument_values:  0.25

    注意：no_extra_call 属于 trajectory-level signal，不在 step-level partial reward 中。
    """

    DEFAULT_WEIGHTS = {
        "format": 0.10,
        "schema_valid": 0.15,
        "tool_selection": 0.30,
        "argument_keys": 0.20,
        "argument_values": 0.25,
    }

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        parser: Optional[ActionParser] = None,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.parser = parser or ActionParser(strict=False)

    def compute(
        self,
        model_output: str,
        oracle: OracleAction,
        metadata: Optional[SampleMetadata] = None,
    ) -> RewardResult:
        """计算组件化 reward。

        Args:
            model_output: 模型生成的原始文本。
            oracle: ground truth action。
            metadata: 样本元数据（name_map, enum_map 等）。

        Returns:
            RewardResult。
        """
        if metadata is None:
            metadata = SampleMetadata()

        # Step 1: 解析模型输出
        parsed = self.parser.parse(model_output)

        # Step 2: 防御 arguments 非 dict（parser 已标记 _args_was_invalid）
        # 如果任何 tool_call 的 arguments 原始值不是 dict，schema_valid 应为 0
        has_invalid_args = any(
            c.get("_args_was_invalid", False) for c in parsed.tool_calls
        ) if parsed.tool_calls else False

        # Step 3: Action-type matrix dispatch
        if parsed.action_type == "unparseable":
            return self._unparseable_result(parsed, oracle)

        if parsed.action_type != oracle.action_type:
            return self._action_type_mismatch(parsed, oracle, metadata)

        # Step 4: 按 action type 计算组件分
        if oracle.action_type == "tool_call":
            result = self._compute_tool_call_reward(parsed, oracle, metadata)
            # 如果 arguments 类型无效，schema_valid 强制为 0
            if has_invalid_args:
                result.components["schema_valid"] = 0.0
                # 重新计算 partial reward 和 effective reward
                weight_sum = sum(self.weights.values())
                partial_reward = sum(
                    self.weights.get(k, 0) * result.components.get(k, 0) for k in self.weights
                ) / weight_sum if weight_sum > 0 else 0.0
                result.diagnostics["partial_reward"] = partial_reward
                result.diagnostics["args_type_invalid"] = True
                if result.exact_success:
                    result.total_reward = 1.0 + 0.05 * partial_reward
                else:
                    result.total_reward = 0.3 * partial_reward
            return result
        elif oracle.action_type == "final_answer":
            return self._compute_final_answer_reward(parsed, oracle, metadata)
        elif oracle.action_type == "report_error":
            return self._compute_report_error_reward(parsed, oracle, metadata)
        elif oracle.action_type == "ask_clarification":
            return self._compute_ask_clarification_reward(parsed, oracle, metadata)
        else:
            return self._unparseable_result(parsed, oracle)

    def _compute_tool_call_reward(
        self,
        parsed: ParsedAction,
        oracle: OracleAction,
        metadata: SampleMetadata,
    ) -> RewardResult:
        """Oracle=tool_call, Model=tool_call 时的组件 reward。"""
        components = {}

        # format: 输出可解析
        components["format"] = 1.0 if parsed.parseable else 0.0

        # schema_valid: 参数符合 schema 约束（类型、required、enum 范围）
        components["schema_valid"] = self._compute_schema_valid(
            parsed.tool_calls, oracle.tool_calls, metadata, oracle.match_mode
        )

        # tool_selection: 函数名映射回 canonical 后正确
        model_names = [c["name"] for c in parsed.tool_calls]
        oracle_names = [c["name"] for c in oracle.tool_calls]
        components["tool_selection"] = self._compute_tool_selection(
            model_names, oracle_names, metadata.name_map
        )

        # argument_keys: required 参数和参数名正确
        components["argument_keys"] = self._compute_argument_keys(
            parsed.tool_calls, oracle.tool_calls, metadata.name_map, oracle.match_mode
        )

        # argument_values: 参数值、enum、类型正确
        components["argument_values"] = self._compute_argument_values(
            parsed.tool_calls, oracle.tool_calls, metadata, oracle.match_mode
        )

        # 计算 partial reward（所有 step-level 组件）
        weight_sum = sum(self.weights.values())
        partial_reward = sum(
            self.weights.get(k, 0) * components.get(k, 0) for k in self.weights
        ) / weight_sum if weight_sum > 0 else 0.0

        # exact_success 检查
        exact_success = self._check_exact_match_tool_call(parsed, oracle, metadata)

        # Correctness floor
        if exact_success:
            effective_reward = 1.0 + 0.05 * partial_reward
        else:
            effective_reward = 0.3 * partial_reward

        return RewardResult(
            total_reward=effective_reward,
            components=components,
            exact_success=exact_success,
            action_type_match=True,
            oracle_action_type=oracle.action_type,
            model_action_type=parsed.action_type,
            diagnostics={
                "partial_reward": partial_reward,
                "model_tool_names": model_names,
                "oracle_tool_names": oracle_names,
            },
        )

    def _compute_final_answer_reward(
        self,
        parsed: ParsedAction,
        oracle: OracleAction,
        metadata: SampleMetadata,
    ) -> RewardResult:
        """Oracle=final_answer, Model=final_answer 时的 reward。"""
        components = {}

        # format
        components["format"] = 1.0 if parsed.parseable else 0.0

        # no_extra_call: model 没有调用工具 = 1.0
        components["no_extra_call"] = 1.0

        # final_answer entity/state match
        # P0 实现：简单的字符串包含检查
        components["final_answer_match"] = self._compute_final_answer_match(
            parsed.content, oracle.final_answer
        )

        # exact_success: final_answer 内容完全匹配
        exact_success = components["final_answer_match"] >= 0.99

        # partial reward
        partial_reward = (
            0.3 * components["format"]
            + 0.2 * components["no_extra_call"]
            + 0.5 * components["final_answer_match"]
        )

        if exact_success:
            effective_reward = 1.0 + 0.05 * partial_reward
        else:
            effective_reward = 0.3 * partial_reward

        return RewardResult(
            total_reward=effective_reward,
            components=components,
            exact_success=exact_success,
            action_type_match=True,
            oracle_action_type=oracle.action_type,
            model_action_type=parsed.action_type,
        )

    def _compute_report_error_reward(
        self,
        parsed: ParsedAction,
        oracle: OracleAction,
        metadata: SampleMetadata,
    ) -> RewardResult:
        """Oracle=report_error, Model=report_error 时的 reward。"""
        components = {
            "format": 1.0 if parsed.parseable else 0.0,
            "no_extra_call": 1.0,  # 没有调用工具
            "error_type_match": 1.0,  # P0: action type 匹配即给分
        }

        partial_reward = 0.5 * components["format"] + 0.3 * components["no_extra_call"] + 0.2 * components["error_type_match"]
        effective_reward = 1.0 + 0.05 * partial_reward  # action type 匹配视为 exact

        return RewardResult(
            total_reward=effective_reward,
            components=components,
            exact_success=True,  # P0: action type 匹配即 exact
            action_type_match=True,
            oracle_action_type=oracle.action_type,
            model_action_type=parsed.action_type,
        )

    def _compute_ask_clarification_reward(
        self,
        parsed: ParsedAction,
        oracle: OracleAction,
        metadata: SampleMetadata,
    ) -> RewardResult:
        """Oracle=ask_clarification, Model=ask_clarification 时的 reward。"""
        components = {
            "format": 1.0 if parsed.parseable else 0.0,
            "no_extra_call": 1.0,
            "clarification_match": 1.0,  # P0: action type 匹配即给分
        }

        partial_reward = 0.5 * components["format"] + 0.3 * components["no_extra_call"] + 0.2 * components["clarification_match"]
        effective_reward = 1.0 + 0.05 * partial_reward

        return RewardResult(
            total_reward=effective_reward,
            components=components,
            exact_success=True,
            action_type_match=True,
            oracle_action_type=oracle.action_type,
            model_action_type=parsed.action_type,
        )

    def _action_type_mismatch(
        self,
        parsed: ParsedAction,
        oracle: OracleAction,
        metadata: SampleMetadata,
    ) -> RewardResult:
        """Oracle 和 model 的 action type 不同时的 reward。"""
        components = {}

        # format 仍可得分
        components["format"] = 1.0 if parsed.parseable else 0.0

        # 其余 step-level 组件全 0
        components["schema_valid"] = 0.0
        components["tool_selection"] = 0.0
        components["argument_keys"] = 0.0
        components["argument_values"] = 0.0

        # partial reward（与 _compute_tool_call_reward 一致，除以 weight_sum）
        weight_sum = sum(self.weights.values())
        partial = sum(self.weights.get(k, 0) * v for k, v in components.items()) / weight_sum if weight_sum > 0 else 0.0

        # action type 不匹配时 exact_success 必为 0，advantage 不得为正
        effective_reward = 0.3 * partial

        # 诊断标记
        diagnostics = {}
        if oracle.action_type == "tool_call" and parsed.action_type == "final_answer":
            diagnostics["error_type"] = "missing_call"
        elif oracle.action_type == "final_answer" and parsed.action_type == "tool_call":
            diagnostics["error_type"] = "unnecessary_call"
        elif oracle.action_type == "tool_call" and parsed.action_type == "report_error":
            diagnostics["error_type"] = "false_error_report"
        elif oracle.action_type == "report_error" and parsed.action_type == "tool_call":
            diagnostics["error_type"] = "unsafe_retry"
        else:
            diagnostics["error_type"] = "action_type_mismatch"

        return RewardResult(
            total_reward=effective_reward,
            components=components,
            exact_success=False,
            action_type_match=False,
            oracle_action_type=oracle.action_type,
            model_action_type=parsed.action_type,
            diagnostics=diagnostics,
        )

    def _unparseable_result(self, parsed: ParsedAction, oracle: OracleAction) -> RewardResult:
        """完全无法解析的输出。"""
        return RewardResult(
            total_reward=0.0,
            components={"format": 0.0, "schema_valid": 0.0, "tool_selection": 0.0, "argument_keys": 0.0, "argument_values": 0.0},
            exact_success=False,
            action_type_match=False,
            oracle_action_type=oracle.action_type,
            model_action_type="unparseable",
            diagnostics={"error_detail": parsed.error_detail},
        )

    # ========== 组件计算辅助方法 ==========

    def _compute_schema_valid(
        self,
        model_calls: list[dict],
        oracle_calls: list[dict],
        metadata: SampleMetadata,
        match_mode: str = "set",
    ) -> float:
        """计算 schema_valid 分数。

        检查模型输出的参数是否符合 schema 约束：
        - required 参数是否齐全
        - 参数值类型是否正确
        - enum 参数值是否在允许范围内

        当前 P0 实现：基于 oracle 的 required keys 覆盖率作为代理指标。
        后续可接入完整 JSON Schema 校验。
        """
        if not oracle_calls:
            return 1.0

        total_score = 0.0
        matched_pairs = self._match_call_pairs(model_calls, oracle_calls, metadata.name_map, match_mode)

        for model_call, oracle_call in matched_pairs:
            if model_call is None:
                total_score += 0.0
                continue

            oracle_args = oracle_call.get("arguments", {})
            model_args = model_call.get("arguments", {})

            if not oracle_args:
                # 无参数要求，只要 model 也没乱加就给满分
                total_score += 1.0
                continue

            # 检查 required keys 是否都存在且值类型合理
            valid_count = 0
            for key, oracle_val in oracle_args.items():
                if key not in model_args:
                    continue
                model_val = model_args[key]
                # 类型兼容性检查
                if self._type_compatible(model_val, oracle_val):
                    valid_count += 1

            score = valid_count / len(oracle_args) if oracle_args else 1.0
            total_score += score

        return total_score / len(oracle_calls) if oracle_calls else 1.0

    def _type_compatible(self, model_val: Any, oracle_val: Any) -> bool:
        """检查 model 值与 oracle 值的类型是否兼容。"""
        if model_val is None:
            return oracle_val is None
        if oracle_val is None:
            return True  # oracle 允许 null

        # 字符串类型
        if isinstance(oracle_val, str):
            return isinstance(model_val, str)
        # 布尔类型（必须在 int/float 之前，因为 bool 是 int 的子类）
        if isinstance(oracle_val, bool):
            return isinstance(model_val, bool) or (
                isinstance(model_val, str) and model_val.lower() in ("true", "false")
            )
        # 数值类型（int/float 互通）
        if isinstance(oracle_val, (int, float)):
            if isinstance(model_val, (int, float)):
                return True
            # 字符串形式的数字也接受
            if isinstance(model_val, str):
                try:
                    float(model_val)
                    return True
                except (ValueError, TypeError):
                    return False
            return False
        # 列表类型
        if isinstance(oracle_val, list):
            return isinstance(model_val, list)
        # 字典类型
        if isinstance(oracle_val, dict):
            return isinstance(model_val, dict)
        return True

    def _compute_tool_selection(
        self,
        model_names: list[str],
        oracle_names: list[str],
        name_map: dict[str, str],
    ) -> float:
        """计算 tool_selection 分数。

        将 model 输出的工具名通过 name_map 映射回 canonical，
        然后与 oracle 的 canonical 名比较。

        使用 multiset 匹配，分数限制在 [0, 1]。
        """
        if not oracle_names:
            return 1.0 if not model_names else 0.0

        # 映射 model names 到 canonical
        canonical_model = [name_map.get(n, n) for n in model_names]
        # oracle names 也映射（以防 oracle 本身是 perturbed 的）
        canonical_oracle = [name_map.get(n, n) for n in oracle_names]

        # multiset 匹配：每个 oracle name 最多被匹配一次
        oracle_remaining = list(canonical_oracle)
        matched = 0
        for m in canonical_model:
            if m in oracle_remaining:
                oracle_remaining.remove(m)
                matched += 1

        score = matched / len(canonical_oracle)
        return min(score, 1.0)

    def _compute_argument_keys(
        self,
        model_calls: list[dict],
        oracle_calls: list[dict],
        name_map: dict[str, str],
        match_mode: str = "set",
    ) -> float:
        """计算 argument_keys 分数。

        对每对匹配的 (model_call, oracle_call)，
        计算 model 提供的参数名与 oracle required 参数名的重合度。
        """
        if not oracle_calls:
            return 1.0

        total_score = 0.0
        matched_pairs = self._match_call_pairs(model_calls, oracle_calls, name_map, match_mode)

        for model_call, oracle_call in matched_pairs:
            oracle_args = oracle_call.get("arguments", {})
            model_args = model_call.get("arguments", {}) if model_call else {}

            if not oracle_args:
                # oracle 无参数，model 也无参数则满分
                total_score += 1.0 if not model_args else 0.5
                continue

            oracle_keys = set(oracle_args.keys())
            model_keys = set(model_args.keys())

            # 正确的 key 数 / oracle 要求的 key 数
            correct_keys = oracle_keys & model_keys
            score = len(correct_keys) / len(oracle_keys) if oracle_keys else 1.0
            total_score += score

        return total_score / len(oracle_calls) if oracle_calls else 1.0

    def _compute_argument_values(
        self,
        model_calls: list[dict],
        oracle_calls: list[dict],
        metadata: SampleMetadata,
        match_mode: str = "set",
    ) -> float:
        """计算 argument_values 分数。

        对每个匹配的参数，检查值是否正确（考虑 enum_map 映射）。
        """
        if not oracle_calls:
            return 1.0

        total_score = 0.0
        matched_pairs = self._match_call_pairs(model_calls, oracle_calls, metadata.name_map, match_mode)

        for model_call, oracle_call in matched_pairs:
            oracle_args = oracle_call.get("arguments", {})
            model_args = model_call.get("arguments", {}) if model_call else {}

            if not oracle_args:
                total_score += 1.0
                continue

            correct_values = 0
            for key, oracle_val in oracle_args.items():
                if key not in model_args:
                    continue
                model_val = model_args[key]

                # 通过 enum_map 映射
                tool_name = oracle_call.get("name", "")
                canonical_model_val = self._map_enum_value(
                    tool_name, key, model_val, metadata.enum_map
                )

                if self._values_match(canonical_model_val, oracle_val):
                    correct_values += 1

            score = correct_values / len(oracle_args) if oracle_args else 1.0
            total_score += score

        return total_score / len(oracle_calls) if oracle_calls else 1.0

    def _compute_final_answer_match(
        self,
        model_answer: Any,
        oracle_answer: str,
    ) -> float:
        """计算 final_answer 匹配分数。

        P0 实现：简单的归一化编辑距离 / 关键词包含。
        """
        if not oracle_answer:
            return 1.0 if not model_answer else 0.5

        model_str = str(model_answer) if model_answer else ""

        # 精确匹配
        if model_str.strip() == oracle_answer.strip():
            return 1.0

        # 包含检查（oracle 的关键词是否出现在 model 中）
        oracle_words = set(oracle_answer.lower().split())
        model_words = set(model_str.lower().split())
        if oracle_words and oracle_words.issubset(model_words):
            return 0.8

        # 部分重合
        overlap = oracle_words & model_words
        if oracle_words:
            return 0.5 * len(overlap) / len(oracle_words)

        return 0.0

    def _check_exact_match_tool_call(
        self,
        parsed: ParsedAction,
        oracle: OracleAction,
        metadata: SampleMetadata,
    ) -> bool:
        """检查 tool_call 是否精确匹配 oracle（支持 set matching）。"""
        if len(parsed.tool_calls) != len(oracle.tool_calls):
            return False

        # 使用 match_mode 配对
        matched_pairs = self._match_call_pairs(
            parsed.tool_calls, oracle.tool_calls, metadata.name_map, oracle.match_mode
        )

        for model_call, oracle_call in matched_pairs:
            if model_call is None:
                return False

            # 工具名匹配（通过 name_map）
            model_name = metadata.name_map.get(model_call["name"], model_call["name"])
            oracle_name = metadata.name_map.get(oracle_call["name"], oracle_call["name"])
            if model_name != oracle_name:
                return False

            # 参数匹配
            model_args = model_call.get("arguments", {})
            oracle_args = oracle_call.get("arguments", {})

            if not isinstance(model_args, dict):
                return False

            if set(model_args.keys()) != set(oracle_args.keys()):
                return False

            for key, oracle_val in oracle_args.items():
                model_val = model_args.get(key)
                tool_name = oracle_call.get("name", "")
                canonical_model_val = self._map_enum_value(
                    tool_name, key, model_val, metadata.enum_map
                )
                if not self._values_match(canonical_model_val, oracle_val):
                    return False

        return True

    def _match_call_pairs(
        self,
        model_calls: list[dict],
        oracle_calls: list[dict],
        name_map: dict[str, str],
        match_mode: str = "set",
    ) -> list[tuple[Optional[dict], dict]]:
        """将 model calls 与 oracle calls 配对。

        match_mode="set" 时使用 unordered matching。
        match_mode="ordered" 时按顺序匹配。
        model 不足时补 None。

        对同名工具不同参数的情况，使用参数值最佳匹配（greedy best-match）。
        """
        if match_mode == "ordered" or len(oracle_calls) <= 1:
            # 顺序匹配
            pairs = []
            for i, oracle_call in enumerate(oracle_calls):
                if i < len(model_calls):
                    pairs.append((model_calls[i], oracle_call))
                else:
                    pairs.append((None, oracle_call))
            return pairs

        # unordered / set matching：按 canonical name + 参数值最佳匹配
        remaining_model = list(enumerate(model_calls))  # (original_idx, call)
        pairs = []

        for oracle_call in oracle_calls:
            oracle_name = name_map.get(oracle_call.get("name", ""), oracle_call.get("name", ""))
            oracle_args = oracle_call.get("arguments", {})

            # 找所有 name 匹配的候选
            candidates = []
            for idx, (orig_idx, mc) in enumerate(remaining_model):
                model_name = name_map.get(mc.get("name", ""), mc.get("name", ""))
                if model_name == oracle_name:
                    candidates.append((idx, mc))

            if not candidates:
                pairs.append((None, oracle_call))
                continue

            if len(candidates) == 1:
                # 唯一候选，直接匹配
                rm_idx = candidates[0][0]
                matched = remaining_model.pop(rm_idx)[1]
                pairs.append((matched, oracle_call))
                continue

            # 多个同名候选：选参数值匹配度最高的
            best_score = -1
            best_rm_idx = candidates[0][0]
            for rm_idx, mc in candidates:
                model_args = mc.get("arguments", {})
                if not oracle_args:
                    score = 1.0 if not model_args else 0.5
                else:
                    matched_vals = 0
                    for k, ov in oracle_args.items():
                        if k in model_args and self._values_match(model_args[k], ov):
                            matched_vals += 1
                    score = matched_vals / len(oracle_args)
                if score > best_score:
                    best_score = score
                    best_rm_idx = rm_idx

            matched = remaining_model.pop(best_rm_idx)[1]
            pairs.append((matched, oracle_call))

        return pairs

    def _map_enum_value(
        self,
        tool_name: str,
        param_name: str,
        value: Any,
        enum_map: dict,
    ) -> Any:
        """通过 enum_map 将 perturbed value 映射回 canonical。"""
        if not enum_map:
            return value
        tool_map = enum_map.get(tool_name, {})
        param_map = tool_map.get(param_name, {})
        if isinstance(value, str) and value in param_map:
            return param_map[value]
        return value

    def _values_match(self, model_val: Any, oracle_val: Any) -> bool:
        """Canonical 值匹配（递归实现）。

        规则（mcp_tools_rl_project_plan.md §10.1.1）：
        - None / null / "" / "null" 视为等价（null-equivalence）
        - 字符串比较：strip() + lower()
        - 数值比较：float(a) == float(b) with tolerance 1e-9
        - 布尔比较：两侧 cast 到 bool
        - 列表比较：sorted element-wise（无序匹配）
        - Dict/object 比较：递归 key-value match
        """
        # null-equivalence
        if self._is_null_equivalent(model_val) and self._is_null_equivalent(oracle_val):
            return True
        if self._is_null_equivalent(model_val) or self._is_null_equivalent(oracle_val):
            return False

        # 布尔比较（必须在数值之前，因为 bool 是 int 子类）
        if isinstance(oracle_val, bool):
            return self._bool_match(model_val, oracle_val)
        if isinstance(model_val, bool):
            return self._bool_match(model_val, oracle_val)

        # 数值比较
        if isinstance(oracle_val, (int, float)) or isinstance(model_val, (int, float)):
            try:
                return abs(float(model_val) - float(oracle_val)) < 1e-9
            except (ValueError, TypeError):
                return False

        # 字符串比较
        if isinstance(oracle_val, str) and isinstance(model_val, str):
            return model_val.strip().lower() == oracle_val.strip().lower()

        # 列表比较（无序，recursive multiset matching）
        if isinstance(oracle_val, list) and isinstance(model_val, list):
            if len(model_val) != len(oracle_val):
                return False
            # Recursive multiset matching：每个 oracle element 从剩余 model elements 中找匹配
            remaining = list(model_val)
            for o_elem in oracle_val:
                found = False
                for i, m_elem in enumerate(remaining):
                    if self._values_match(m_elem, o_elem):
                        remaining.pop(i)
                        found = True
                        break
                if not found:
                    return False
            return True

        # Dict 递归比较
        if isinstance(oracle_val, dict) and isinstance(model_val, dict):
            if set(model_val.keys()) != set(oracle_val.keys()):
                return False
            return all(
                self._values_match(model_val[k], oracle_val[k])
                for k in oracle_val
            )

        # 字符串化 fallback
        return str(model_val).strip().lower() == str(oracle_val).strip().lower()

    @staticmethod
    def _is_null_equivalent(val: Any) -> bool:
        """判断值是否为 null 等价。"""
        if val is None:
            return True
        if isinstance(val, str) and val.strip().lower() in ("", "null", "none"):
            return True
        return False

    @staticmethod
    def _bool_match(model_val: Any, oracle_val: Any) -> bool:
        """布尔值匹配，处理字符串形式的布尔。"""
        def to_bool(v: Any) -> Optional[bool]:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                low = v.strip().lower()
                if low in ("true", "1", "yes"):
                    return True
                if low in ("false", "0", "no"):
                    return False
            if isinstance(v, (int, float)):
                return bool(v)
            return None

        mb = to_bool(model_val)
        ob = to_bool(oracle_val)
        if mb is None or ob is None:
            return False
        return mb == ob
