from src.live_mcp.api import LiveMCPBranch, load_live_tasks


def test_live_mcp_branch_generates_and_loads_tasks(tmp_path):
    output = tmp_path / "tasks.jsonl"
    with LiveMCPBranch.from_suite("configs/live_mcp/suite_mvp.yaml") as branch:
        summary = branch.generate_tasks_to_file(
            output_path=output,
            server_name="calendar",
            count=2,
            seed=31,
        )
    tasks = load_live_tasks(output)
    assert summary.tasks_written == 2
    assert len(tasks) == 2
    assert all("calendar" in task.target_servers for task in tasks)
