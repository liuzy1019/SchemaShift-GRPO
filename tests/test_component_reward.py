"""Action Parser + Component Reward 单元测试。

覆盖：
  - 4 种 action type 解析
  - unparseable 处理
  - Qwen 风格 JSON fallback
  - 并行调用解析
  - Action-type matrix 全部 8 种 oracle×model 组合
  - Correctness floor
  - 组件分计算
"""

import pytest

from src.reward.action_parser import ActionParser, ParsedAction, parse_action
from src.reward.component_reward import (
    ComponentReward,
    OracleAction,
    RewardResult,
    SampleMetadata,
)


# ============================================================
# Action Parser 测试
# ============================================================


class TestActionParser:
    """Action Parser 测试。"""

    def test_parse_tool_call_tagged(self):
        """标签格式 tool_call。"""
        output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>'
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is True
        assert result.tool_name == "get_weather"
        assert result.arguments == {"city": "Beijing"}

    def test_parse_tool_call_array(self):
        """并行调用（JSON 数组）。"""
        output = '<tool_call>[{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "get_time", "arguments": {"zone": "CST"}}]</tool_call>'
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_time"

    def test_parse_final_answer_tagged(self):
        """标签格式 final_answer。"""
        output = "<final_answer>The weather in Beijing is sunny, 25°C.</final_answer>"
        result = parse_action(output)
        assert result.action_type == "final_answer"
        assert result.parseable is True
        assert "sunny" in result.content

    def test_parse_ask_clarification_tagged(self):
        """标签格式 ask_clarification。"""
        output = "<ask_clarification>Which city do you mean?</ask_clarification>"
        result = parse_action(output)
        assert result.action_type == "ask_clarification"
        assert result.parseable is True

    def test_parse_report_error_tagged(self):
        """标签格式 report_error。"""
        output = "<report_error>API timeout after 30s</report_error>"
        result = parse_action(output)
        assert result.action_type == "report_error"
        assert result.parseable is True

    def test_parse_json_fallback(self):
        """Qwen 风格 JSON fallback。"""
        output = '{"name": "search_api", "arguments": {"query": "test"}}'
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is True
        assert result.tool_name == "search_api"

    def test_parse_plain_text_final_answer(self):
        """纯文本 fallback 为 final_answer。"""
        output = "The answer to your question is that the stock price went up by 5% today."
        result = parse_action(output)
        assert result.action_type == "final_answer"
        assert result.parseable is True

    def test_parse_empty_output(self):
        """空输出 → unparseable。"""
        result = parse_action("")
        assert result.action_type == "unparseable"
        assert result.parseable is False

    def test_parse_short_garbage(self):
        """短垃圾输出 → unparseable。"""
        result = parse_action("ok")
        assert result.action_type == "unparseable"
        assert result.parseable is False

    def test_strict_mode_rejects_json(self):
        """严格模式不接受 JSON fallback。"""
        output = '{"name": "search_api", "arguments": {"query": "test"}}'
        result = parse_action(output, strict=True)
        assert result.action_type == "unparseable"

    def test_parse_invalid_json_in_tag(self):
        """标签内 JSON 无效。"""
        output = "<tool_call>not valid json</tool_call>"
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is False


# ============================================================
# Component Reward 测试
# ============================================================


class TestComponentReward:
    """Component Reward 测试。"""

    @pytest.fixture
    def reward_fn(self):
        return ComponentReward()

    @pytest.fixture
    def simple_oracle(self):
        return OracleAction(
            action_type="tool_call",
            tool_calls=[{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}],
        )

    @pytest.fixture
    def metadata(self):
        return SampleMetadata()

    def test_exact_match_reward(self, reward_fn, simple_oracle, metadata):
        """精确匹配 → exact_success=True, reward > 1.0。"""
        output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>'
        result = reward_fn.compute(output, simple_oracle, metadata)
        assert result.exact_success is True
        assert result.action_type_match is True
        assert result.total_reward > 1.0
        assert result.components["format"] == 1.0
        assert result.components["tool_selection"] == 1.0
        assert result.components["argument_keys"] == 1.0
        assert result.components["argument_values"] == 1.0

    def test_wrong_tool_name(self, reward_fn, simple_oracle, metadata):
        """工具名错误 → tool_selection=0, exact_success=False。"""
        output = '<tool_call>{"name": "get_time", "arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>'
        result = reward_fn.compute(output, simple_oracle, metadata)
        assert result.exact_success is False
        assert result.components["tool_selection"] == 0.0
        assert result.total_reward < 1.0
        assert result.total_reward > 0.0  # partial reward 仍有

    def test_wrong_argument_key(self, reward_fn, simple_oracle, metadata):
        """参数名错误 → argument_keys < 1.0。"""
        output = '<tool_call>{"name": "get_weather", "arguments": {"location": "Beijing", "unit": "celsius"}}</tool_call>'
        result = reward_fn.compute(output, simple_oracle, metadata)
        assert result.exact_success is False
        assert result.components["argument_keys"] < 1.0
        assert result.components["tool_selection"] == 1.0

    def test_wrong_argument_value(self, reward_fn, simple_oracle, metadata):
        """参数值错误 → argument_values < 1.0。"""
        output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Shanghai", "unit": "celsius"}}</tool_call>'
        result = reward_fn.compute(output, simple_oracle, metadata)
        assert result.exact_success is False
        assert result.components["argument_values"] < 1.0
        assert result.components["argument_keys"] == 1.0

    def test_format_error(self, reward_fn, simple_oracle, metadata):
        """格式错误 → format=0, total_reward=0。"""
        output = ""
        result = reward_fn.compute(output, simple_oracle, metadata)
        assert result.total_reward == 0.0
        assert result.exact_success is False

    def test_action_type_mismatch_missing_call(self, reward_fn, simple_oracle, metadata):
        """Oracle=tool_call, Model=final_answer → missing_call。"""
        output = "<final_answer>I don't know how to help with that.</final_answer>"
        result = reward_fn.compute(output, simple_oracle, metadata)
        assert result.action_type_match is False
        assert result.exact_success is False
        assert result.diagnostics.get("error_type") == "missing_call"
        assert result.total_reward < 0.5

    def test_action_type_mismatch_unnecessary_call(self, reward_fn, metadata):
        """Oracle=final_answer, Model=tool_call → unnecessary_call。"""
        oracle = OracleAction(action_type="final_answer", final_answer="The answer is 42.")
        output = '<tool_call>{"name": "calculate", "arguments": {"expr": "6*7"}}</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.action_type_match is False
        assert result.exact_success is False
        assert result.diagnostics.get("error_type") == "unnecessary_call"
        # no_extra_call 已移到 trajectory-level，step-level 不再包含
        assert "no_extra_call" not in result.components

    def test_action_type_mismatch_unsafe_retry(self, reward_fn, metadata):
        """Oracle=report_error, Model=tool_call → unsafe_retry。"""
        oracle = OracleAction(action_type="report_error", error_info="API timeout")
        output = '<tool_call>{"name": "retry_api", "arguments": {}}</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.action_type_match is False
        assert result.diagnostics.get("error_type") == "unsafe_retry"

    def test_correctness_floor_exact_vs_partial(self, reward_fn, simple_oracle, metadata):
        """Correctness floor: exact=1 时 reward > 1.0, exact=0 时 reward < 0.3。"""
        # exact match
        exact_output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>'
        exact_result = reward_fn.compute(exact_output, simple_oracle, metadata)

        # partial match (wrong value)
        partial_output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Shanghai", "unit": "fahrenheit"}}</tool_call>'
        partial_result = reward_fn.compute(partial_output, simple_oracle, metadata)

        assert exact_result.total_reward > 1.0
        assert partial_result.total_reward < 0.3
        assert exact_result.total_reward > partial_result.total_reward

    def test_name_map_mapping(self, reward_fn):
        """name_map 映射：perturbed name → canonical name。"""
        oracle = OracleAction(
            action_type="tool_call",
            tool_calls=[{"name": "get_weather", "arguments": {"city": "Beijing"}}],
        )
        metadata = SampleMetadata(name_map={"fetch_climate_data": "get_weather"})
        output = '<tool_call>{"name": "fetch_climate_data", "arguments": {"city": "Beijing"}}</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.components["tool_selection"] == 1.0
        assert result.exact_success is True

    def test_enum_map_mapping(self, reward_fn):
        """enum_map 映射：perturbed enum value → canonical value。"""
        oracle = OracleAction(
            action_type="tool_call",
            tool_calls=[{"name": "get_weather", "arguments": {"unit": "celsius"}}],
        )
        metadata = SampleMetadata(
            enum_map={"get_weather": {"unit": {"metric_celsius": "celsius"}}}
        )
        output = '<tool_call>{"name": "get_weather", "arguments": {"unit": "metric_celsius"}}</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.components["argument_values"] == 1.0
        assert result.exact_success is True

    def test_final_answer_match(self, reward_fn, metadata):
        """Oracle=final_answer, Model=final_answer 精确匹配。"""
        oracle = OracleAction(action_type="final_answer", final_answer="The answer is 42.")
        output = "<final_answer>The answer is 42.</final_answer>"
        result = reward_fn.compute(output, oracle, metadata)
        assert result.action_type_match is True
        assert result.exact_success is True
        assert result.total_reward > 1.0

    def test_parallel_calls_exact(self, reward_fn, metadata):
        """并行调用精确匹配。"""
        oracle = OracleAction(
            action_type="tool_call",
            tool_calls=[
                {"name": "get_weather", "arguments": {"city": "Beijing"}},
                {"name": "get_time", "arguments": {"zone": "CST"}},
            ],
        )
        output = '<tool_call>[{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "get_time", "arguments": {"zone": "CST"}}]</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.exact_success is True
        assert result.components["tool_selection"] == 1.0

    def test_parallel_calls_wrong_order(self, reward_fn, metadata):
        """并行调用顺序不同 → set matching 下视为正确。"""
        oracle = OracleAction(
            action_type="tool_call",
            tool_calls=[
                {"name": "get_weather", "arguments": {"city": "Beijing"}},
                {"name": "get_time", "arguments": {"zone": "CST"}},
            ],
        )
        output = '<tool_call>[{"name": "get_time", "arguments": {"zone": "CST"}}, {"name": "get_weather", "arguments": {"city": "Beijing"}}]</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        # set matching: 顺序无关，应视为精确匹配
        assert result.exact_success is True
        assert result.components["tool_selection"] == 1.0


# ============================================================
# 回归测试：覆盖 review 中指出的边界情况
# ============================================================


class TestRegressionInvalidArguments:
    """P1 回归：非 dict arguments 不抛异常且 schema_valid=0。"""

    @pytest.fixture
    def reward_fn(self):
        return ComponentReward()

    @pytest.fixture
    def metadata(self):
        return SampleMetadata()

    def test_string_arguments_no_crash(self, reward_fn, metadata):
        """arguments 为字符串时不抛异常。"""
        oracle = OracleAction(
            action_type="tool_call",
            tool_calls=[{"name": "get_weather", "arguments": {"city": "Beijing"}}],
        )
        output = '<tool_call>{"name": "get_weather", "arguments": "city=Beijing"}</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.total_reward >= 0
        assert result.components["schema_valid"] == 0.0
        assert result.diagnostics.get("args_type_invalid") is True

    def test_list_arguments_no_crash(self, reward_fn, metadata):
        """arguments 为列表时不抛异常。"""
        oracle = OracleAction(
            action_type="tool_call",
            tool_calls=[{"name": "get_weather", "arguments": {"city": "Beijing"}}],
        )
        output = '<tool_call>{"name": "get_weather", "arguments": ["city", "Beijing"]}</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.total_reward >= 0
        assert result.components["schema_valid"] == 0.0


class TestRegressionSameToolParallel:
    """P1 回归：同名工具不同参数的 parallel matching。"""

    @pytest.fixture
    def reward_fn(self):
        return ComponentReward()

    @pytest.fixture
    def metadata(self):
        return SampleMetadata()

    def test_same_tool_different_args_swapped(self, reward_fn, metadata):
        """同名工具不同参数调换顺序 → exact success。"""
        oracle = OracleAction(
            action_type="tool_call",
            tool_calls=[
                {"name": "get_weather", "arguments": {"city": "Beijing"}},
                {"name": "get_weather", "arguments": {"city": "Shanghai"}},
            ],
        )
        output = '<tool_call>[{"name": "get_weather", "arguments": {"city": "Shanghai"}}, {"name": "get_weather", "arguments": {"city": "Beijing"}}]</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.exact_success is True
        assert result.components["argument_values"] == 1.0

    def test_same_tool_three_calls_swapped(self, reward_fn, metadata):
        """三个同名工具调换顺序 → exact success。"""
        oracle = OracleAction(
            action_type="tool_call",
            tool_calls=[
                {"name": "search", "arguments": {"q": "A"}},
                {"name": "search", "arguments": {"q": "B"}},
                {"name": "search", "arguments": {"q": "C"}},
            ],
        )
        output = '<tool_call>[{"name": "search", "arguments": {"q": "C"}}, {"name": "search", "arguments": {"q": "A"}}, {"name": "search", "arguments": {"q": "B"}}]</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.exact_success is True


class TestRegressionValuesMatch:
    """P1 回归：_values_match canonical 规则。"""

    @pytest.fixture
    def reward_fn(self):
        return ComponentReward()

    def test_list_unordered_strings(self, reward_fn):
        """列表无序 + 字符串 case-insensitive。"""
        assert reward_fn._values_match(["B", "a"], ["a", "b"]) is True

    def test_list_unordered_dicts(self, reward_fn):
        """列表无序 + dict 递归。"""
        assert reward_fn._values_match(
            [{"x": "B"}, {"x": "a"}],
            [{"x": "a"}, {"x": "b"}],
        ) is True

    def test_dict_recursive_case_insensitive(self, reward_fn):
        """dict 递归 + 字符串 case-insensitive。"""
        assert reward_fn._values_match(
            {"name": "ALICE", "age": 30},
            {"name": "alice", "age": 30},
        ) is True

    def test_numeric_tolerance(self, reward_fn):
        """数值 tolerance 1e-9。"""
        assert reward_fn._values_match(1.0, 1.0 + 1e-10) is True
        assert reward_fn._values_match(1.0, 1.1) is False

    def test_bool_string_match(self, reward_fn):
        """布尔值 vs 字符串形式。"""
        assert reward_fn._values_match("true", True) is True
        assert reward_fn._values_match("false", False) is True
        assert reward_fn._values_match("yes", True) is True

    def test_null_equivalence(self, reward_fn):
        """null-equivalence 规则。"""
        assert reward_fn._values_match(None, "") is True
        assert reward_fn._values_match("null", None) is True
        assert reward_fn._values_match("none", None) is True
        assert reward_fn._values_match("hello", None) is False


class TestRegressionMatchMode:
    """P1 回归：match_mode 从 OracleAction 贯穿。"""

    @pytest.fixture
    def reward_fn(self):
        return ComponentReward()

    @pytest.fixture
    def metadata(self):
        return SampleMetadata()

    def test_ordered_mode_rejects_swap(self, reward_fn, metadata):
        """ordered 模式下调换顺序 → 不是 exact match。"""
        oracle = OracleAction(
            action_type="tool_call",
            tool_calls=[
                {"name": "step_a", "arguments": {"x": 1}},
                {"name": "step_b", "arguments": {"y": 2}},
            ],
            match_mode="ordered",
        )
        output = '<tool_call>[{"name": "step_b", "arguments": {"y": 2}}, {"name": "step_a", "arguments": {"x": 1}}]</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.exact_success is False

    def test_set_mode_accepts_swap(self, reward_fn, metadata):
        """set 模式下调换顺序 → exact match。"""
        oracle = OracleAction(
            action_type="tool_call",
            tool_calls=[
                {"name": "step_a", "arguments": {"x": 1}},
                {"name": "step_b", "arguments": {"y": 2}},
            ],
            match_mode="set",
        )
        output = '<tool_call>[{"name": "step_b", "arguments": {"y": 2}}, {"name": "step_a", "arguments": {"x": 1}}]</tool_call>'
        result = reward_fn.compute(output, oracle, metadata)
        assert result.exact_success is True


class TestRegressionToolSelection:
    """P2 回归：tool_selection 分数不超过 1。"""

    @pytest.fixture
    def reward_fn(self):
        return ComponentReward()

    def test_score_capped_at_one(self, reward_fn):
        """重复工具名不会导致分数超过 1。"""
        score = reward_fn._compute_tool_selection(
            ["a", "a", "a"], ["a", "b"], {}
        )
        assert score <= 1.0
        assert score == 0.5  # 只有 1 个 "a" 能匹配 oracle 的 "a"

    def test_multiset_matching(self, reward_fn):
        """multiset 匹配正确计数。"""
        score = reward_fn._compute_tool_selection(
            ["a", "b", "c"], ["a", "b"], {}
        )
        assert score == 1.0  # 2/2 matched
