from src.live_mcp import errors
from src.live_mcp.types import ToolCall


def test_unknown_tool_does_not_mutate_state(live_manager, executor):
    session = live_manager.create_session(seed=2)
    before = live_manager.get_state(session.session_id)
    result = executor.execute(session.session_id, ToolCall("missing_tool", {}, "x"))
    after = live_manager.get_state(session.session_id)
    assert result.error_type == errors.UNKNOWN_TOOL
    assert before == after


def test_schema_invalid_does_not_call_server(live_manager, executor):
    session = live_manager.create_session(seed=2)
    before = live_manager.get_state(session.session_id, "calendar")
    result = executor.execute(session.session_id, ToolCall("update_event", {"event_id": "evt_001"}, "x"))
    after = live_manager.get_state(session.session_id, "calendar")
    assert result.error_type == errors.SCHEMA_INVALID
    assert before == after


def test_mutating_tool_sets_state_changed(live_manager, executor):
    session = live_manager.create_session(seed=2)
    result = executor.execute(
        session.session_id,
        ToolCall("update_event", {"event_id": "evt_001", "fields": {"start_time": "2026-07-03T10:00"}}, "x"),
    )
    assert result.success is True
    assert result.state_changed is True
