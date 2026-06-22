from src.live_mcp.reward import RewardComposer
from src.live_mcp.types import LiveTask, OracleCall, OracleProgram, RolloutTrace, ToolCall, ToolExecutionResult, TraceTurn


def test_reward_has_flat_five_components():
    task = LiveTask(
        task_id="t",
        source="test",
        suite_name="s",
        user_prompt="p",
        session_id="sid",
        session_seed=1,
        target_servers=["calendar"],
        visible_tools=[],
        required_tools=["list_events"],
        expected_outcome={},
        success_criteria=[],
        oracle_program=OracleProgram("t", [OracleCall("list_events", {})], []),
        sampling_context={},
        max_turns=1,
    )
    result = ToolExecutionResult(True, "list_events", "list_events", "c", "sid", {}, None, "", True, False, 1)
    trace = RolloutTrace("tr", "t", "sid", "m", "", None, [TraceTurn(0, "h", "", "tool_call", [ToolCall("list_events", {}, "c")], [result], "", False)], "", {})
    reward = RewardComposer().compute(task, trace, final_state={"calendar": {}})
    assert set(reward) >= {
        "component_validity",
        "component_coverage",
        "component_efficiency",
        "component_tool_selection",
        "component_argument_value",
    }
    assert all(not isinstance(value, (dict, list)) for value in reward.values())


def test_reward_penalizes_hidden_tool_call_on_missing_function_task():
    task = LiveTask(
        task_id="t",
        source="test",
        suite_name="s",
        user_prompt="p",
        session_id="sid",
        session_seed=1,
        target_servers=["calendar"],
        visible_tools=[],
        required_tools=["list_events"],
        expected_outcome={"abstain": True},
        success_criteria=[{"type": "missing_function", "server": "calendar", "tool": "update_event"}],
        oracle_program=OracleProgram("t", [], [{"type": "missing_function", "server": "calendar", "tool": "update_event"}]),
        sampling_context={},
        max_turns=1,
        task_type="missing_function",
        hidden_tools=["update_event"],
        metadata={"unavailable_required_tool": "update_event"},
    )
    hidden_result = ToolExecutionResult(True, "update_event", "update_event", "c", "sid", {}, None, "", True, True, 1)
    trace = RolloutTrace(
        "tr",
        "t",
        "sid",
        "m",
        "",
        None,
        [TraceTurn(0, "h", "", "tool_call", [ToolCall("update_event", {}, "c")], [hidden_result], "", False)],
        "",
        {},
    )

    reward = RewardComposer().compute(task, trace, final_state={"calendar": {}})

    assert reward["component_abstention"] == 0.0
    assert reward["component_coverage"] == 0.0
    assert reward["component_tool_selection"] == 0.0
    assert reward["score"] < 0.5


def test_reward_gives_full_score_for_missing_function_report_error():
    task = LiveTask(
        task_id="t",
        source="test",
        suite_name="s",
        user_prompt="p",
        session_id="sid",
        session_seed=1,
        target_servers=["calendar"],
        visible_tools=[],
        required_tools=[],
        expected_outcome={"abstain": True},
        success_criteria=[{"type": "missing_function", "server": "calendar", "tool": "update_event"}],
        oracle_program=OracleProgram("t", [], [{"type": "missing_function", "server": "calendar", "tool": "update_event"}]),
        sampling_context={},
        max_turns=1,
        task_type="missing_function",
        hidden_tools=["update_event"],
        metadata={"unavailable_required_tool": "update_event"},
    )
    trace = RolloutTrace(
        "tr",
        "t",
        "sid",
        "m",
        "",
        None,
        [TraceTurn(0, "h", "<report_error>Required tool update_event is not available.</report_error>", "report_error", [], [], "", True)],
        "",
        {},
    )

    reward = RewardComposer().compute(task, trace, final_state={"calendar": {}})

    assert reward["component_abstention"] == 1.0
    assert reward["component_coverage"] == 1.0
    assert reward["score"] == 1.0
