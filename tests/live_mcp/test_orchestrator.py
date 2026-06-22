from src.live_mcp.orchestrator import StateMachineOrchestrator


def test_orchestrator_generates_valid_tasks_with_knobs(suite, live_manager, executor):
    orchestrator = StateMachineOrchestrator(suite, live_manager, executor)
    tasks = orchestrator.generate_many("all", count=10, seed=10, difficulty_mix={"easy": 1.0})
    assert len(tasks) == 10
    assert any(task.task_type == "missing_function" for task in tasks)
    assert any(task.metadata.get("has_distractors") for task in tasks)
    assert all(task.metadata.get("oracle_validated") for task in tasks)


def test_missing_function_task_abstains_instead_of_calling_hidden_tool(suite, live_manager, executor):
    orchestrator = StateMachineOrchestrator(suite, live_manager, executor)
    tasks = orchestrator.generate_many("calendar", count=5, seed=10, difficulty_mix={"easy": 1.0})
    task = [candidate for candidate in tasks if candidate.task_type == "missing_function"][0]

    assert task.hidden_tools
    hidden = task.hidden_tools[0]
    assert hidden not in {tool["name"] for tool in task.visible_tools}
    assert hidden not in task.required_tools
    assert hidden not in [call.tool_name for call in task.oracle_program.calls]
    assert task.oracle_program.calls == []
    assert task.success_criteria == [{"type": "missing_function", "server": "calendar", "tool": hidden}]
    assert task.metadata["original_oracle_program"]["calls"]
