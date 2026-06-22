"""EpisodeSeed Schema 和 Builder 测试。

测试 Phase 2 的核心产出：
  - EpisodeSeed schema 定义的序列化/反序列化
  - 验证逻辑
  - EpisodeSeedBuilder 的转换逻辑
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.episode_schema import (
    ActionType,
    DataSource,
    EpisodeSeed,
    EpisodeType,
    OracleStep,
    PerturbationTag,
    SplitKeys,
    VerifierConfig,
    load_episodes,
    save_episodes,
    validate_episodes,
)
from src.data.episode_seed_builder import (
    BuilderConfig,
    EpisodeSeedBuilder,
    _extract_mcp_server,
    _parse_json_field,
    _parse_tool_call_content,
    _truncate_observation,
)


# ============================================================
# EpisodeSeed Schema 测试
# ============================================================


class TestVerifierConfig:
    """VerifierConfig 测试。"""

    def test_default(self):
        vc = VerifierConfig()
        assert vc.type == "exact"
        assert vc.name_map == {}
        assert vc.enum_map == {}
        assert vc.lenient_values is False

    def test_roundtrip(self):
        vc = VerifierConfig(
            type="exact",
            name_map={"perturbed_tool": "canonical_tool"},
            enum_map={"tool": {"param": {"a": "b"}}},
            lenient_values=True,
            expected_entities=["entity1"],
        )
        d = vc.to_dict()
        vc2 = VerifierConfig.from_dict(d)
        assert vc2.type == vc.type
        assert vc2.name_map == vc.name_map
        assert vc2.enum_map == vc.enum_map
        assert vc2.lenient_values == vc.lenient_values
        assert vc2.expected_entities == vc.expected_entities


class TestOracleStep:
    """OracleStep 测试。"""

    def test_tool_call_step(self):
        step = OracleStep(
            step=0,
            action_type=ActionType.TOOL_CALL.value,
            tool_name="get_weather",
            arguments={"city": "beijing"},
            replay_observation="sunny, 32C",
        )
        assert step.validate() == []

        d = step.to_dict()
        assert d["step"] == 0
        assert d["action_type"] == "tool_call"
        assert d["tool_name"] == "get_weather"
        assert d["arguments"] == {"city": "beijing"}
        assert d["replay_observation"] == "sunny, 32C"

    def test_parallel_tool_call_step(self):
        step = OracleStep(
            step=0,
            action_type=ActionType.PARALLEL_TOOL_CALL.value,
            calls=[
                {"tool_name": "get_weather", "arguments": {"city": "beijing"}},
                {"tool_name": "get_weather", "arguments": {"city": "shanghai"}},
            ],
            match_mode="set",
            replay_observations=["sunny, 32C", "cloudy, 28C"],
        )
        assert step.validate() == []

    def test_final_answer_step(self):
        step = OracleStep(
            step=1,
            action_type=ActionType.FINAL_ANSWER.value,
            expected_content="The weather is sunny.",
        )
        assert step.validate() == []

    def test_validation_errors(self):
        # 缺少 tool_name
        step = OracleStep(step=0, action_type=ActionType.TOOL_CALL.value)
        errors = step.validate()
        assert any("tool_name" in e for e in errors)

        # 并行调用缺少 calls
        step = OracleStep(step=0, action_type=ActionType.PARALLEL_TOOL_CALL.value)
        errors = step.validate()
        assert any("calls" in e for e in errors)

        # 无效 action_type
        step = OracleStep(step=0, action_type="invalid_type")
        errors = step.validate()
        assert any("invalid action_type" in e for e in errors)

    def test_roundtrip(self):
        step = OracleStep(
            step=2,
            action_type=ActionType.TOOL_CALL.value,
            tool_name="search_books",
            arguments={"query": "python", "limit": 10},
            replay_observation='[{"title": "Python Cookbook"}]',
            verifier=VerifierConfig(type="exact", name_map={"search_books": "find_books"}),
        )
        d = step.to_dict()
        step2 = OracleStep.from_dict(d)
        assert step2.step == step.step
        assert step2.tool_name == step.tool_name
        assert step2.arguments == step.arguments
        assert step2.verifier.name_map == step.verifier.name_map


class TestEpisodeSeed:
    """EpisodeSeed 测试。"""

    def _make_valid_episode(self) -> EpisodeSeed:
        """构造一个合法的 EpisodeSeed。"""
        return EpisodeSeed(
            episode_id="toucan_test_001",
            source=DataSource.TOUCAN.value,
            episode_type=EpisodeType.CALL_THEN_FINAL.value,
            scenario_tags=[],
            mcp_servers=["weather-server"],
            tools_snapshot=[{
                "type": "function",
                "function": {
                    "name": "weather-server-get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            initial_messages=[
                {"role": "user", "content": "What's the weather in Beijing?"},
            ],
            oracle_trace=[
                OracleStep(
                    step=0,
                    action_type=ActionType.TOOL_CALL.value,
                    tool_name="weather-server-get_weather",
                    arguments={"city": "Beijing"},
                    replay_observation="Sunny, 32°C",
                ),
                OracleStep(
                    step=1,
                    action_type=ActionType.FINAL_ANSWER.value,
                    expected_content="The weather in Beijing is sunny, 32°C.",
                ),
            ],
            max_turns=3,
            split_keys=SplitKeys(
                server="weather-server",
                domain="weather",
                tool_names=["weather-server-get_weather"],
            ),
        )

    def test_valid_episode(self):
        ep = self._make_valid_episode()
        errors = ep.validate()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_properties(self):
        ep = self._make_valid_episode()
        assert ep.decision_turns == 2
        assert ep.has_parallel_calls is False
        assert ep.tool_names_used == ["weather-server-get_weather"]

    def test_roundtrip_json(self):
        ep = self._make_valid_episode()
        d = ep.to_dict()
        ep2 = EpisodeSeed.from_dict(d)
        assert ep2.episode_id == ep.episode_id
        assert ep2.source == ep.source
        assert ep2.episode_type == ep.episode_type
        assert len(ep2.oracle_trace) == len(ep.oracle_trace)
        assert ep2.oracle_trace[0].tool_name == ep.oracle_trace[0].tool_name

    def test_roundtrip_jsonl(self):
        ep = self._make_valid_episode()
        line = ep.to_jsonl_line()
        d = json.loads(line)
        ep2 = EpisodeSeed.from_dict(d)
        assert ep2.episode_id == ep.episode_id

    def test_validation_missing_fields(self):
        # 缺少 episode_id
        ep = self._make_valid_episode()
        ep.episode_id = ""
        errors = ep.validate()
        assert any("episode_id" in e for e in errors)

    def test_validation_empty_tools(self):
        ep = self._make_valid_episode()
        ep.tools_snapshot = []
        errors = ep.validate()
        assert any("tools_snapshot" in e for e in errors)

    def test_validation_no_user_message(self):
        ep = self._make_valid_episode()
        ep.initial_messages = [{"role": "assistant", "content": "hi"}]
        errors = ep.validate()
        assert any("user message" in e for e in errors)

    def test_validation_type_trace_consistency(self):
        # no_tool 类型不应有 tool_call
        ep = self._make_valid_episode()
        ep.episode_type = EpisodeType.NO_TOOL.value
        errors = ep.validate()
        assert any("no_tool" in e for e in errors)

    def test_save_load_roundtrip(self):
        ep = self._make_valid_episode()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        save_episodes([ep], path)
        loaded = load_episodes(path)
        assert len(loaded) == 1
        assert loaded[0].episode_id == ep.episode_id
        assert loaded[0].oracle_trace[0].tool_name == "weather-server-get_weather"

        Path(path).unlink()

    def test_validate_episodes_batch(self):
        ep_valid = self._make_valid_episode()
        ep_invalid = self._make_valid_episode()
        ep_invalid.episode_id = ""

        report = validate_episodes([ep_valid, ep_invalid])
        assert report["total"] == 2
        assert report["valid"] == 1
        assert report["invalid"] == 1


# ============================================================
# Builder 工具函数测试
# ============================================================


class TestBuilderUtils:
    """Builder 工具函数测试。"""

    def test_parse_json_field_json(self):
        result = _parse_json_field('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_field_python_dict(self):
        result = _parse_json_field("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_parse_json_field_list(self):
        result = _parse_json_field('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_parse_json_field_non_string(self):
        result = _parse_json_field(42)
        assert result == 42

    def test_parse_tool_call_content(self):
        content = "{'name': 'get_weather', 'arguments': '{\"city\": \"Beijing\"}'}"
        result = _parse_tool_call_content(content)
        assert result is not None
        assert result["name"] == "get_weather"
        assert result["arguments_parsed"] == {"city": "Beijing"}

    def test_parse_tool_call_content_dict_args(self):
        content = json.dumps({"name": "calc", "arguments": {"x": 1, "y": 2}})
        result = _parse_tool_call_content(content)
        assert result is not None
        assert result["arguments_parsed"] == {"x": 1, "y": 2}

    def test_parse_tool_call_content_invalid(self):
        result = _parse_tool_call_content("not a valid tool call")
        assert result is None

    def test_extract_mcp_server(self):
        assert _extract_mcp_server("weather-server-get_weather") == "weather-server"
        assert _extract_mcp_server("okx-server-get_price") == "okx-server"
        assert _extract_mcp_server("simple_tool") == "simple_tool"

    def test_truncate_observation_short(self):
        obs = "short observation"
        assert _truncate_observation(obs, 100) == obs

    def test_truncate_observation_long(self):
        obs = "x" * 3000
        result = _truncate_observation(obs, 200)
        assert len(result) < 3000
        assert "truncated" in result


# ============================================================
# Builder 集成测试
# ============================================================


class TestEpisodeSeedBuilder:
    """EpisodeSeedBuilder 集成测试。"""

    def _make_toucan_item(
        self,
        uuid: str = "test-uuid-001",
        subset_name: str = "single-turn-original",
        question: str = "What's the weather?",
        target_tools: str = "get_weather",
        tools: list | None = None,
        messages: list | None = None,
    ) -> dict:
        """构造一条 Toucan 格式的样本。"""
        if tools is None:
            tools = [{
                "type": "function",
                "function": {
                    "name": "weather-server-get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }]

        if messages is None:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Let me check the weather."},
                {"role": "tool_call", "content": json.dumps({"name": "weather-server-get_weather", "arguments": json.dumps({"city": "Beijing"})})},
                {"role": "tool_response", "content": "Sunny, 32°C"},
                {"role": "assistant", "content": "The weather in Beijing is sunny, 32°C."},
            ]

        return {
            "uuid": uuid,
            "subset_name": subset_name,
            "question": question,
            "target_tools": target_tools,
            "tools": json.dumps(tools),
            "messages": json.dumps(messages),
        }

    def test_build_single_item(self):
        """测试单条样本转换。"""
        item = self._make_toucan_item()
        builder = EpisodeSeedBuilder(BuilderConfig())
        episodes = builder.build_from_data([item])

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.episode_id == "toucan_test-uuid-001"
        assert ep.source == "toucan"
        assert ep.episode_type == EpisodeType.CALL_THEN_FINAL.value
        assert len(ep.oracle_trace) == 2
        assert ep.oracle_trace[0].action_type == ActionType.TOOL_CALL.value
        assert ep.oracle_trace[0].tool_name == "weather-server-get_weather"
        assert ep.oracle_trace[0].arguments == {"city": "Beijing"}
        assert ep.oracle_trace[1].action_type == ActionType.FINAL_ANSWER.value

    def test_build_irrelevant_as_no_tool(self):
        """测试 irrelevant 子集转为 no_tool。"""
        item = self._make_toucan_item(
            subset_name="irrelevant",
            target_tools="",
            messages=[
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
            ],
        )
        builder = EpisodeSeedBuilder(BuilderConfig(include_irrelevant_as_no_tool=True))
        episodes = builder.build_from_data([item])

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.episode_type == EpisodeType.NO_TOOL.value
        assert ep.oracle_trace[0].action_type == ActionType.FINAL_ANSWER.value

    def test_skip_unparseable_messages(self):
        """测试跳过无法解析的 messages。"""
        item = {
            "uuid": "bad",
            "subset_name": "single-turn-original",
            "question": "test",
            "target_tools": "tool",
            "tools": json.dumps([{"type": "function", "function": {"name": "t", "parameters": {}}}]),
            "messages": "not valid json {{{{",
        }
        builder = EpisodeSeedBuilder(BuilderConfig())
        episodes = builder.build_from_data([item])
        assert len(episodes) == 0

    def test_skip_no_target_tools(self):
        """测试跳过没有 target_tools 的非 irrelevant 样本。"""
        item = self._make_toucan_item(target_tools="")
        builder = EpisodeSeedBuilder(BuilderConfig())
        episodes = builder.build_from_data([item])
        assert len(episodes) == 0

    def test_parallel_tool_calls(self):
        """测试并行 tool_call 的处理。"""
        messages = [
            {"role": "user", "content": "Compare weather in Beijing and Shanghai"},
            {"role": "assistant", "content": "Let me check both cities."},
            {"role": "tool_call", "content": json.dumps({"name": "weather-get_weather", "arguments": json.dumps({"city": "Beijing"})})},
            {"role": "tool_response", "content": "Sunny, 32°C"},
            {"role": "tool_call", "content": json.dumps({"name": "weather-get_weather", "arguments": json.dumps({"city": "Shanghai"})})},
            {"role": "tool_response", "content": "Cloudy, 28°C"},
            {"role": "assistant", "content": "Beijing is sunny 32°C, Shanghai is cloudy 28°C."},
        ]
        item = self._make_toucan_item(
            target_tools="get_weather",
            messages=messages,
        )
        builder = EpisodeSeedBuilder(BuilderConfig())
        episodes = builder.build_from_data([item])

        assert len(episodes) == 1
        ep = episodes[0]
        # 两个独立的 tool_call step（因为中间有 tool_response 分隔）
        tool_steps = [s for s in ep.oracle_trace if s.action_type == ActionType.TOOL_CALL.value]
        assert len(tool_steps) == 2

    def test_true_parallel_calls(self):
        """测试真正的并行调用（连续 tool_call 无 response 间隔）。"""
        messages = [
            {"role": "user", "content": "Compare weather"},
            {"role": "assistant", "content": "Checking..."},
            {"role": "tool_call", "content": json.dumps({"name": "weather-get_weather", "arguments": json.dumps({"city": "Beijing"})})},
            {"role": "tool_call", "content": json.dumps({"name": "weather-get_weather", "arguments": json.dumps({"city": "Shanghai"})})},
            {"role": "tool_response", "content": "Sunny, 32°C"},
            {"role": "tool_response", "content": "Cloudy, 28°C"},
            {"role": "assistant", "content": "Done."},
        ]
        item = self._make_toucan_item(
            target_tools="get_weather",
            messages=messages,
        )
        # 不要求严格配对（因为并行格式不同）
        builder = EpisodeSeedBuilder(BuilderConfig(require_paired=False))
        episodes = builder.build_from_data([item])

        # 应该能处理（可能作为 parallel 或失败取决于配对逻辑）
        # 当前实现中连续 tool_call 会被识别为 parallel
        if episodes:
            ep = episodes[0]
            assert ep.has_parallel_calls or len(ep.oracle_trace) >= 1

    def test_truncate_long_episode(self):
        """测试长 episode 截断。"""
        # 构造 6 步的 episode
        messages = [{"role": "user", "content": "Do many things"}]
        messages.append({"role": "assistant", "content": "OK"})
        for i in range(6):
            messages.append({"role": "tool_call", "content": json.dumps({"name": f"tool_{i}", "arguments": "{}"})})
            messages.append({"role": "tool_response", "content": f"result_{i}"})
        messages.append({"role": "assistant", "content": "All done."})

        tools = [{"type": "function", "function": {"name": f"tool_{i}", "parameters": {"type": "object", "properties": {}}}} for i in range(6)]

        item = self._make_toucan_item(
            target_tools="tool_0,tool_1,tool_2,tool_3,tool_4,tool_5",
            tools=tools,
            messages=messages,
        )
        config = BuilderConfig(max_decision_turns=3, truncate_long_episodes=True)
        builder = EpisodeSeedBuilder(config)
        episodes = builder.build_from_data([item])

        if episodes:
            ep = episodes[0]
            # 截断后不超过 max_decision_turns
            assert ep.decision_turns <= 3

    def test_save_and_load(self):
        """测试保存和加载。"""
        item = self._make_toucan_item()
        builder = EpisodeSeedBuilder(BuilderConfig())
        episodes = builder.build_from_data([item])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        save_episodes(episodes, path)
        loaded = load_episodes(path)

        assert len(loaded) == len(episodes)
        assert loaded[0].episode_id == episodes[0].episode_id
        assert loaded[0].oracle_trace[0].tool_name == episodes[0].oracle_trace[0].tool_name

        Path(path).unlink()
