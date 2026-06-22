"""Oracle planning and validation for Live MCP tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.query_generator import StructuredTask
from src.live_mcp.types import OracleCall, OracleProgram, ToolCall, ToolExecutionResult


@dataclass
class OracleValidationResult:
    valid: bool
    execution_results: list[ToolExecutionResult]
    final_state: dict[str, Any]
    failed_criteria: list[dict[str, Any]]
    error: str = ""


class OraclePlanner:
    def plan(self, task: StructuredTask) -> OracleProgram:
        slots = task.slots
        if task.server_name == "calendar":
            calls = [
                OracleCall("list_events", {}),
                OracleCall("update_event", {"event_id": slots["event_id"], "fields": {"start_time": slots["new_time"]}}),
            ]
        elif task.server_name == "shopping":
            calls = [
                OracleCall("search_products", {"category": slots["category"], "max_price": slots["max_price"]}),
                OracleCall("add_to_cart", {"product_id": slots["product_id"], "quantity": slots["quantity"]}),
                OracleCall("checkout", {}),
            ]
        else:
            calls = []
        return OracleProgram(task_id=task.task_id, calls=calls, success_criteria=task.success_criteria)


class OracleValidator:
    def validate(
        self,
        task: StructuredTask,
        oracle_program: OracleProgram,
        manager: LiveMCPManager,
        executor: LiveMCPExecutor,
        seed: int,
    ) -> OracleValidationResult:
        session = manager.create_session(seed=seed)
        manager.discover_tools(session.session_id)
        results: list[ToolExecutionResult] = []
        try:
            for idx, call in enumerate(oracle_program.calls):
                result = executor.execute(
                    session.session_id,
                    ToolCall(call.tool_name, dict(call.arguments), call_id=f"oracle_{idx}"),
                )
                results.append(result)
                if not result.success:
                    return OracleValidationResult(False, results, manager.get_state(session.session_id), oracle_program.success_criteria, result.error_message)
            final_state = manager.get_state(session.session_id)
            failed = [criterion for criterion in oracle_program.success_criteria if not criterion_satisfied(final_state, criterion)]
            return OracleValidationResult(not failed, results, final_state, failed)
        finally:
            manager.close_session(session.session_id)


def criterion_satisfied(final_state: dict[str, Any], criterion: dict[str, Any]) -> bool:
    kind = criterion.get("type")
    server = criterion.get("server")
    state = final_state.get(server, {})
    if kind == "state_equals":
        return _get_path(state, str(criterion["path"])) == criterion.get("value")
    if kind == "cart_empty":
        return state.get("cart") == []
    if kind == "order_contains_product":
        product_id = criterion.get("product_id")
        for order in state.get("orders", {}).values():
            if any(item.get("product_id") == product_id for item in order.get("items", [])):
                return True
        return False
    if kind == "missing_function":
        return True
    return False


def _get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
