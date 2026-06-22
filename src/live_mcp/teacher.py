"""Teacher adapters for deterministic Live MCP data generation."""

from __future__ import annotations

import json
import random
from typing import Any, Protocol

from src.live_mcp.query_generator import QueryGenerator, StructuredTask
from src.live_mcp.types import RolloutTrace, ToolExecutionResult


class TeacherAdapter(Protocol):
    def generate_query(
        self,
        task: StructuredTask,
        sampling_context: dict[str, Any],
        persona: str,
        reference_date: str,
        information_level: str,
    ) -> str: ...

    def generate_assistant_turn(
        self,
        messages: list[dict[str, str]],
        tool_schemas: list[dict[str, Any]],
        state: Any,
    ) -> str: ...

    def decide_continuation(
        self,
        trace: RolloutTrace,
        min_turns: int,
        max_turns: int,
        rng: random.Random,
    ) -> str: ...

    def recover(
        self,
        failed_result: ToolExecutionResult,
        messages: list[dict[str, str]],
        tool_schemas: list[dict[str, Any]],
    ) -> str: ...


class DeterministicTeacherAdapter:
    def __init__(self, query_generator: QueryGenerator):
        self.query_generator = query_generator

    def generate_query(
        self,
        task: StructuredTask,
        sampling_context: dict[str, Any],
        persona: str = "default",
        reference_date: str = "2026-06-22",
        information_level: str = "grounded",
    ) -> str:
        return self.query_generator.render(task)

    def generate_assistant_turn(
        self,
        messages: list[dict[str, str]],
        tool_schemas: list[dict[str, Any]],
        state: Any,
    ) -> str:
        calls = getattr(state, "oracle_calls", [])
        turn_idx = getattr(state, "turn_idx", 0)
        if turn_idx < len(calls):
            call = calls[turn_idx]
            payload = {"name": call.tool_name, "arguments": call.arguments}
            return f"<tool_call>{json.dumps(payload, ensure_ascii=True)}</tool_call>"
        return "<final_answer>Done.</final_answer>"

    def decide_continuation(
        self,
        trace: RolloutTrace,
        min_turns: int,
        max_turns: int,
        rng: random.Random,
    ) -> str:
        return "end" if len(trace.turns) >= min_turns else "follow_up"

    def recover(
        self,
        failed_result: ToolExecutionResult,
        messages: list[dict[str, str]],
        tool_schemas: list[dict[str, Any]],
    ) -> str:
        return "<report_error>I could not complete the tool action.</report_error>"
