"""SFT Step Exporter 测试。

测试 Phase 3 的核心产出：
  - SFTStepExporter 的展开逻辑
  - 格式化函数
  - 多步展开
  - 边界情况
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
    SplitKeys,
    VerifierConfig,
)
from src.data.sft_step_exporter import (
    SYSTEM_PROMPT,
    ExporterConfig,
    SFTSample,
    SFTStepExporter,
    _format_observation,
    _format_tool_call_completion,
    _format_tools_section,
)


# ============================================================
# 测试数据构造
# ============================================================


def _make_call_then_final_episode() -> EpisodeSeed:
    """构造一个 call_then_final 类型的 episode。"""
    return EpisodeSeed(
        episode_id="test_ep_001",
        source=DataSource.TOUCAN.value,
        episode_type=EpisodeType.CALL_THEN_FINAL.value,
        mcp_servers=["weather-server"],
        tools_snapshot=[{
            "type": "function",
            "function": {
                "name": "weather-server-get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
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
                arguments={"city": "Beijing", "unit": "celsius"},
                replay_observation='{"temperature": 32, "condition": "sunny"}',
            ),
            OracleStep(
                step=1,
                action_type=ActionType.FINAL_ANSWER.value,
                expected_content="The weather in Beijing is sunny with a temperature of 32°C.",
            ),
        ],
        max_turns=3,
        split_keys=SplitKeys(server="weather-server", domain="weather"),
    )


def _make_parallel_episode() -> EpisodeSeed:
    """构造一个包含并行调用的 episode。"""
    return EpisodeSeed(
        episode_id="test_ep_002",
        source=DataSource.TOUCAN.value,
        episode_type=EpisodeType.CALL_THEN_FINAL.value,
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
            {"role": "user", "content": "Compare weather in Beijing and Shanghai"},
        ],
        oracle_trace=[
            OracleStep(
                step=0,
                action_type=ActionType.PARALLEL_TOOL_CALL.value,
                calls=[
                    {"tool_name": "weather-server-get_weather", "arguments": {"city": "Beijing"}},
                    {"tool_name": "weather-server-get_weather", "arguments": {"city": "Shanghai"}},
                ],
                replay_observations=["Sunny, 32C", "Cloudy, 28C"],
            ),
            OracleStep(
                step=1,
                action_type=ActionType.FINAL_ANSWER.value,
                expected_content="Beijing is sunny 32C, Shanghai is cloudy 28C.",
            ),
        ],
        max_turns=3,
    )


def _make_no_tool_episode() -> EpisodeSeed:
    """构造一个 no_tool 类型的 episode。"""
    return EpisodeSeed(
        episode_id="test_ep_003",
        source=DataSource.TOUCAN.value,
        episode_type=EpisodeType.NO_TOOL.value,
        tools_snapshot=[{
            "type": "function",
            "function": {
                "name": "some-tool",
                "description": "A tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        initial_messages=[
            {"role": "user", "content": "Tell me a joke."},
        ],
        oracle_trace=[
            OracleStep(
                step=0,
                action_type=ActionType.FINAL_ANSWER.value,
                expected_content="Why did the chicken cross the road? To get to the other side!",
            ),
        ],
        max_turns=1,
    )


def _make_multi_step_episode() -> EpisodeSeed:
    """构造一个 3 步的 call_then_call episode。"""
    return EpisodeSeed(
        episode_id="test_ep_004",
        source=DataSource.TOUCAN.value,
        episode_type=EpisodeType.CALL_THEN_CALL.value,
        mcp_servers=["fs-server"],
        tools_snapshot=[
            {
                "type": "function",
                "function": {
                    "name": "fs-server-read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fs-server-write_file",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ],
        initial_messages=[
            {"role": "user", "content": "Read config.json and add a new field 'version': '2.0'"},
        ],
        oracle_trace=[
            OracleStep(
                step=0,
                action_type=ActionType.TOOL_CALL.value,
                tool_name="fs-server-read_file",
                arguments={"path": "config.json"},
                replay_observation='{"name": "myapp", "debug": true}',
            ),
            OracleStep(
                step=1,
                action_type=ActionType.TOOL_CALL.value,
                tool_name="fs-server-write_file",
                arguments={"path": "config.json", "content": '{"name": "myapp", "debug": true, "version": "2.0"}'},
                replay_observation="File written successfully.",
            ),
            OracleStep(
                step=2,
                action_type=ActionType.FINAL_ANSWER.value,
                expected_content="Done. I've added the 'version': '2.0' field to config.json.",
            ),
        ],
        max_turns=4,
    )


# ============================================================
# 格式化函数测试
# ============================================================


class TestFormatFunctions:
    """格式化函数测试。"""

    def test_format_tool_call_completion_single(self):
        step = OracleStep(
            step=0,
            action_type=ActionType.TOOL_CALL.value,
            tool_name="get_weather",
            arguments={"city": "Beijing"},
        )
        result = _format_tool_call_completion(step)
        assert result.startswith("<tool_call>")
        assert result.endswith("</tool_call>")
        # 验证 JSON 可解析
        json_str = result[len("<tool_call>"):-len("</tool_call>")]
        parsed = json.loads(json_str)
        assert parsed["name"] == "get_weather"
        assert parsed["arguments"] == {"city": "Beijing"}

    def test_format_tool_call_completion_parallel(self):
        step = OracleStep(
            step=0,
            action_type=ActionType.PARALLEL_TOOL_CALL.value,
            calls=[
                {"tool_name": "get_weather", "arguments": {"city": "Beijing"}},
                {"tool_name": "get_weather", "arguments": {"city": "Shanghai"}},
            ],
        )
        result = _format_tool_call_completion(step)
        assert result.startswith("<tool_call>")
        assert result.endswith("</tool_call>")
        json_str = result[len("<tool_call>"):-len("</tool_call>")]
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "get_weather"
        assert parsed[1]["arguments"] == {"city": "Shanghai"}

    def test_format_tool_call_completion_final_answer(self):
        step = OracleStep(
            step=0,
            action_type=ActionType.FINAL_ANSWER.value,
            expected_content="The answer is 42.",
        )
        result = _format_tool_call_completion(step)
        assert result == "<final_answer>The answer is 42.</final_answer>"

    def test_format_tool_call_completion_ask_clarification(self):
        step = OracleStep(
            step=0,
            action_type=ActionType.ASK_CLARIFICATION.value,
            expected_content="Which city do you mean?",
        )
        result = _format_tool_call_completion(step)
        assert result == "<ask_clarification>Which city do you mean?</ask_clarification>"

    def test_format_tool_call_completion_report_error(self):
        step = OracleStep(
            step=0,
            action_type=ActionType.REPORT_ERROR.value,
            expected_content="The tool returned an error.",
        )
        result = _format_tool_call_completion(step)
        assert result == "<report_error>The tool returned an error.</report_error>"

    def test_format_tools_section(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }]
        result = _format_tools_section(tools)
        assert "get_weather" in result
        assert "city" in result
        assert "(required)" in result
        assert "enum:" in result

    def test_format_tools_section_empty(self):
        result = _format_tools_section([])
        assert result == ""

    def test_format_observation_single(self):
        step = OracleStep(
            step=0,
            action_type=ActionType.TOOL_CALL.value,
            tool_name="test",
            replay_observation="result data",
        )
        result = _format_observation(step)
        assert result == "result data"

    def test_format_observation_parallel(self):
        step = OracleStep(
            step=0,
            action_type=ActionType.PARALLEL_TOOL_CALL.value,
            calls=[
                {"tool_name": "tool_a", "arguments": {}},
                {"tool_name": "tool_b", "arguments": {}},
            ],
            replay_observations=["result_a", "result_b"],
        )
        result = _format_observation(step)
        assert "[tool_a]:" in result
        assert "[tool_b]:" in result
        assert "result_a" in result
        assert "result_b" in result

    def test_format_observation_truncation(self):
        step = OracleStep(
            step=0,
            action_type=ActionType.TOOL_CALL.value,
            tool_name="test",
            replay_observation="x" * 3000,
        )
        result = _format_observation(step, max_length=200)
        assert len(result) < 3000
        assert "truncated" in result


# ============================================================
# SFTStepExporter 测试
# ============================================================


class TestSFTStepExporter:
    """SFTStepExporter 测试。"""

    def test_export_call_then_final(self):
        """测试 call_then_final episode 展开为 2 条样本。"""
        ep = _make_call_then_final_episode()
        exporter = SFTStepExporter(ExporterConfig())
        samples = exporter.export_single(ep)

        assert len(samples) == 2

        # 第一条：tool_call
        s0 = samples[0]
        assert s0.action_type == ActionType.TOOL_CALL.value
        assert s0.step_idx == 0
        assert "<tool_call>" in s0.completion
        assert "weather-server-get_weather" in s0.completion
        # prompt 中应有 user message
        assert any("Beijing" in m.get("content", "") for m in s0.prompt_messages)
        # prompt 中应有 system prompt
        assert s0.messages[0]["role"] == "system"

        # 第二条：final_answer
        s1 = samples[1]
        assert s1.action_type == ActionType.FINAL_ANSWER.value
        assert s1.step_idx == 1
        assert "<final_answer>" in s1.completion
        # prompt 中应包含前一步的 action 和 observation
        prompt_text = " ".join(m.get("content", "") for m in s1.prompt_messages)
        assert "weather-server-get_weather" in prompt_text  # 前一步 action
        assert "temperature" in prompt_text  # observation

    def test_export_parallel_calls(self):
        """测试并行调用 episode 的展开。"""
        ep = _make_parallel_episode()
        exporter = SFTStepExporter(ExporterConfig())
        samples = exporter.export_single(ep)

        assert len(samples) == 2

        # 第一条应该是并行调用格式
        s0 = samples[0]
        assert "<tool_call>" in s0.completion
        # 解析 JSON 验证是数组
        json_str = s0.completion[len("<tool_call>"):-len("</tool_call>")]
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_export_no_tool(self):
        """测试 no_tool episode 的展开。"""
        ep = _make_no_tool_episode()
        exporter = SFTStepExporter(ExporterConfig(skip_no_tool_episodes=False))
        samples = exporter.export_single(ep)

        assert len(samples) == 1
        assert samples[0].action_type == ActionType.FINAL_ANSWER.value
        assert "<final_answer>" in samples[0].completion

    def test_skip_no_tool(self):
        """测试跳过 no_tool episode。"""
        ep = _make_no_tool_episode()
        exporter = SFTStepExporter(ExporterConfig(skip_no_tool_episodes=True))
        samples = exporter.export_single(ep)
        assert len(samples) == 0

    def test_export_multi_step(self):
        """测试多步 episode 展开为 3 条样本。"""
        ep = _make_multi_step_episode()
        exporter = SFTStepExporter(ExporterConfig())
        samples = exporter.export_single(ep)

        assert len(samples) == 3

        # 验证上下文累积
        # 第三条样本的 prompt 应包含前两步的 action 和 observation
        s2 = samples[2]
        prompt_text = " ".join(m.get("content", "") for m in s2.prompt_messages)
        assert "read_file" in prompt_text  # step 0 action
        assert "write_file" in prompt_text  # step 1 action
        assert "myapp" in prompt_text  # step 0 observation
        assert "File written" in prompt_text  # step 1 observation

    def test_first_step_only(self):
        """测试只导出第一步。"""
        ep = _make_multi_step_episode()
        exporter = SFTStepExporter(ExporterConfig(expand_all_steps=False))
        samples = exporter.export_single(ep)

        assert len(samples) == 1
        assert samples[0].step_idx == 0

    def test_completion_parseable_by_action_parser(self):
        """验证生成的 completion 能被 action_parser 正确解析。"""
        from src.reward.action_parser import parse_action

        ep = _make_call_then_final_episode()
        exporter = SFTStepExporter(ExporterConfig())
        samples = exporter.export_single(ep)

        for sample in samples:
            parsed = parse_action(sample.completion, strict=True)
            assert parsed.parseable, f"Completion not parseable: {sample.completion}"
            assert parsed.action_type != "unparseable"

    def test_parallel_completion_parseable(self):
        """验证并行调用的 completion 能被 action_parser 正确解析。"""
        from src.reward.action_parser import parse_action

        ep = _make_parallel_episode()
        exporter = SFTStepExporter(ExporterConfig())
        samples = exporter.export_single(ep)

        s0 = samples[0]
        parsed = parse_action(s0.completion, strict=True)
        assert parsed.parseable
        assert parsed.action_type == "tool_call"
        assert len(parsed.tool_calls) == 2

    def test_export_batch(self):
        """测试批量导出。"""
        episodes = [
            _make_call_then_final_episode(),
            _make_parallel_episode(),
            _make_no_tool_episode(),
            _make_multi_step_episode(),
        ]
        exporter = SFTStepExporter(ExporterConfig())
        samples = exporter.export_from_episodes(episodes)

        # 2 + 2 + 1 + 3 = 8 条样本
        assert len(samples) == 8

    def test_save_and_load(self):
        """测试保存和加载。"""
        ep = _make_call_then_final_episode()
        config = ExporterConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_path = str(Path(tmpdir) / "sft_train.jsonl")
            config.stats_output_path = str(Path(tmpdir) / "stats.json")

            exporter = SFTStepExporter(config)
            samples = exporter.export_from_episodes([ep])
            exporter.save(samples)

            # 验证文件存在
            assert Path(config.output_path).exists()
            assert Path(config.stats_output_path).exists()

            # 验证 JSONL 可读
            with open(config.output_path) as f:
                lines = f.readlines()
            assert len(lines) == 2

            # 验证每行是合法 JSON
            for line in lines:
                item = json.loads(line)
                assert "messages" in item
                assert "metadata" in item

            # 验证统计
            with open(config.stats_output_path) as f:
                stats = json.load(f)
            assert stats["total_samples"] == 2

    def test_system_prompt_contains_format_instructions(self):
        """验证 system prompt 包含格式说明。"""
        assert "<tool_call>" in SYSTEM_PROMPT
        assert "<final_answer>" in SYSTEM_PROMPT
        assert "<ask_clarification>" in SYSTEM_PROMPT
        assert "<report_error>" in SYSTEM_PROMPT

    def test_tools_in_system_prompt(self):
        """验证工具 schema 被包含在 system prompt 中。"""
        ep = _make_call_then_final_episode()
        exporter = SFTStepExporter(ExporterConfig(tools_in_system=True))
        samples = exporter.export_single(ep)

        # system message 应包含工具信息
        system_msg = samples[0].messages[0]
        assert system_msg["role"] == "system"
        assert "weather-server-get_weather" in system_msg["content"]
        assert "city" in system_msg["content"]

    def test_no_oracle_trace_in_prompt(self):
        """关键测试：验证后续步骤的 oracle action/observation 不会泄露到当前步骤的 prompt 中。

        注意：工具名出现在 tools_snapshot（system prompt）中是合法的，
        因为 tools_snapshot 对模型可见。这里检查的是后续步骤的
        具体 action 调用和 observation 结果不会提前泄露。
        """
        ep = _make_multi_step_episode()
        exporter = SFTStepExporter(ExporterConfig())
        samples = exporter.export_single(ep)

        # 第一条样本的 prompt 不应包含后续步骤的 observation
        s0 = samples[0]
        prompt_text = " ".join(m.get("content", "") for m in s0.prompt_messages)
        # step 0 的 prompt 不应包含 step 1 的 observation
        assert "File written successfully" not in prompt_text
        # step 0 的 prompt 不应包含 step 2 的 expected_content
        assert "I've added" not in prompt_text

        # 第二条样本应包含 step 0 的 observation，但不包含 step 2 的
        s1 = samples[1]
        prompt_text_1 = " ".join(m.get("content", "") for m in s1.prompt_messages)
        assert "myapp" in prompt_text_1  # step 0 observation 应该在
        assert "I've added" not in prompt_text_1  # step 2 内容不应在

    def test_sample_to_dict_format(self):
        """验证 SFTSample.to_dict() 的格式。"""
        ep = _make_call_then_final_episode()
        exporter = SFTStepExporter(ExporterConfig())
        samples = exporter.export_single(ep)

        d = samples[0].to_dict()
        assert "messages" in d
        assert "metadata" in d
        assert d["metadata"]["episode_id"] == "test_ep_001"
        assert d["metadata"]["step_idx"] == 0
        assert d["metadata"]["action_type"] == "tool_call"
