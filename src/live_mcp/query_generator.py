"""Grounded deterministic query rendering and validation."""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Any

from src.live_mcp.dependency_graph import ToolChain


@dataclass
class StructuredTask:
    task_id: str
    server_name: str
    tool_chain: ToolChain
    slots: dict[str, Any]
    user_visible_slots: list[str]
    hidden_slots: list[str]
    success_criteria: list[dict[str, Any]]
    required_tools: list[str]
    difficulty: str


@dataclass
class QueryValidationResult:
    valid: bool
    missing_visible_slots: list[str]
    leaked_hidden_slots: list[str]
    error: str = ""


class QueryGenerator:
    def __init__(self, templates: dict[str, list[str]] | None = None):
        self.templates = templates or {}

    def render(self, structured_task: StructuredTask, style: str = "default") -> str:
        template_key = structured_task.task_id.split(":", 1)[0]
        choices = self.templates.get(template_key) or [self._default_template(structured_task)]
        template = choices[0]
        allowed_fields = {
            name
            for _, name, _, _ in Formatter().parse(template)
            if name is not None and name != ""
        }
        values = {key: structured_task.slots.get(key, "") for key in allowed_fields}
        return template.format(**values)

    def validate_query(
        self,
        query: str,
        structured_task: StructuredTask,
    ) -> QueryValidationResult:
        missing = []
        for slot in structured_task.user_visible_slots:
            value = structured_task.slots.get(slot)
            if value is not None and str(value) not in query:
                missing.append(slot)
        leaked = []
        allow_hidden = structured_task.slots.get("_allow_hidden_slots", [])
        for slot in structured_task.hidden_slots:
            if slot in allow_hidden:
                continue
            value = structured_task.slots.get(slot)
            if value is not None and str(value) in query:
                leaked.append(slot)
        return QueryValidationResult(valid=not missing and not leaked, missing_visible_slots=missing, leaked_hidden_slots=leaked)

    def _default_template(self, task: StructuredTask) -> str:
        if task.server_name == "calendar":
            return "Update {title} from {old_time} to {new_time}, then tell me the new time."
        return "Find a {category} under {max_price} dollars, add it to the cart, and checkout."
