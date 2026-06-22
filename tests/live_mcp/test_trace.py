from src.live_mcp.trace import TraceRecorder
from src.live_mcp.types import LiveTask, OracleProgram


def test_trace_saves_task_metadata(tmp_path):
    task = LiveTask(
        task_id="t",
        source="test",
        suite_name="suite",
        user_prompt="prompt",
        session_id="sid",
        session_seed=1,
        target_servers=[],
        visible_tools=[],
        required_tools=[],
        expected_outcome={},
        success_criteria=[],
        oracle_program=OracleProgram("t", [], []),
        sampling_context={},
        max_turns=1,
    )
    recorder = TraceRecorder(tmp_path)
    trace = recorder.start(task, "model")
    recorder.finish(trace, "success", {"score": 1.0})
    path = recorder.save(trace)
    assert path.exists()
    assert "suite" in str(path)
