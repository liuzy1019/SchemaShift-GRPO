from src.live_mcp.dependency_graph import ToolChain
from src.live_mcp.query_generator import QueryGenerator, StructuredTask


def test_query_generator_keeps_hidden_id_out():
    task = StructuredTask(
        task_id="calendar_update_existing_event:template",
        server_name="calendar",
        tool_chain=ToolChain("c", "calendar", ["list_events", "update_event"], [], "easy"),
        slots={"title": "Team Sync", "old_time": "9", "new_time": "10", "event_id": "evt_001"},
        user_visible_slots=["title", "old_time", "new_time"],
        hidden_slots=["event_id"],
        success_criteria=[],
        required_tools=["list_events", "update_event"],
        difficulty="easy",
    )
    gen = QueryGenerator({"calendar_update_existing_event": ["Move {title} from {old_time} to {new_time}."]})
    query = gen.render(task)
    result = gen.validate_query(query, task)
    assert result.valid is True
    assert "evt_001" not in query
