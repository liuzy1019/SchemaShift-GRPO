from src.live_mcp.agent_loop import AgentLoopConfig, MCPToolsAgentLoop, OracleGenerationBackend
from src.live_mcp.orchestrator import StateMachineOrchestrator
from src.live_mcp.trace import TraceRecorder
from src.reward.action_parser import ActionParser


def test_agent_loop_completes_calendar_multistep(tmp_path, suite, live_manager, executor):
    orchestrator = StateMachineOrchestrator(suite, live_manager, executor)
    task = orchestrator.generate_one("calendar", seed=20, difficulty="easy")
    session = live_manager.create_session(seed=20)
    live_manager.discover_tools(session.session_id)
    task.session_id = session.session_id
    loop = MCPToolsAgentLoop(
        live_manager,
        executor,
        ActionParser(strict=False),
        TraceRecorder(tmp_path),
        AgentLoopConfig(max_turns=8),
    )
    trace = loop.rollout(task, OracleGenerationBackend(task))
    assert trace.final_status == "success"
    assert trace.reward["score"] > 0


def test_agent_loop_missing_function_oracle_reports_error_without_tool_calls(tmp_path, suite, live_manager, executor):
    orchestrator = StateMachineOrchestrator(suite, live_manager, executor)
    task = [
        candidate
        for candidate in orchestrator.generate_many("calendar", count=5, seed=20, difficulty_mix={"easy": 1.0})
        if candidate.task_type == "missing_function"
    ][0]
    session = live_manager.create_session(seed=24)
    live_manager.discover_tools(session.session_id)
    task.session_id = session.session_id
    loop = MCPToolsAgentLoop(
        live_manager,
        executor,
        ActionParser(strict=False),
        TraceRecorder(tmp_path),
        AgentLoopConfig(max_turns=8),
    )

    trace = loop.rollout(task, OracleGenerationBackend(task))

    assert trace.final_status == "success"
    assert trace.turns[-1].parsed_action_type == "report_error"
    assert trace.reward["component_abstention"] == 1.0
    assert trace.reward["num_tool_calls"] == 0.0
