from src.live_mcp.types import ToolCall


def test_reset_session_is_deterministic(live_manager):
    session = live_manager.create_session(seed=1)
    state_a = live_manager.get_state(session.session_id)
    live_manager.call_tool(
        "calendar",
        session.session_id,
        "update_event",
        {"event_id": "evt_001", "fields": {"start_time": "2026-07-01T09:00"}},
    )
    live_manager.reset_session(session.session_id, seed=1)
    state_b = live_manager.get_state(session.session_id)
    assert state_a == state_b


def test_sessions_are_isolated(live_manager, executor):
    a = live_manager.create_session(seed=1)
    b = live_manager.create_session(seed=1)
    executor.execute(
        a.session_id,
        ToolCall("update_event", {"event_id": "evt_001", "fields": {"start_time": "2026-07-02T09:00"}}, "c1"),
    )
    state_a = live_manager.get_state(a.session_id, "calendar")["calendar"]
    state_b = live_manager.get_state(b.session_id, "calendar")["calendar"]
    assert state_a["events"]["evt_001"]["start_time"] != state_b["events"]["evt_001"]["start_time"]
