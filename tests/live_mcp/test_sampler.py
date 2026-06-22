import random

from src.live_mcp.sampler import CalendarSampler


def test_sampler_uses_existing_entity(live_manager):
    session = live_manager.create_session(seed=3)
    task = CalendarSampler().sample_task(session.session_id, live_manager, "easy", random.Random(3))
    state = live_manager.get_state(session.session_id, "calendar")["calendar"]
    assert task.slots["event_id"] in state["events"]
