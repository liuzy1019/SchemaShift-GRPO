"""Helpers for simple line-delimited JSON stdio servers."""

from __future__ import annotations

import copy
import json
import sys
from typing import Any, Callable

from src.live_mcp import errors
from src.live_mcp.state_seeder import StateSeeder


class StatefulToolServer:
    def __init__(self, server_name: str, tools: list[dict[str, Any]]):
        self.server_name = server_name
        self.tools = tools
        self.seeder = StateSeeder()
        self.sessions: dict[str, dict[str, Any]] = {}
        self.handlers: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {}

    def handle_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "healthcheck":
            return {"result": {"ok": True, "server_name": self.server_name}}
        if method == "shutdown":
            return {"result": {"ok": True}}
        if method == "tools/list":
            return {"result": {"tools": copy.deepcopy(self.tools)}}
        if method == "session/reset":
            session_id = str(params["session_id"])
            seed = int(params.get("seed", 42))
            self.sessions[session_id] = self.seeder.reset_state(self.server_name, session_id, seed)
            return {"result": {"ok": True}}
        if method == "session/close":
            self.sessions.pop(str(params["session_id"]), None)
            return {"result": {"ok": True}}
        if method == "debug/get_state":
            state = self._state(str(params["session_id"]))
            return {"result": {"state": copy.deepcopy(state)}}
        if method == "tools/call":
            return {"result": self._call_tool(params)}
        return {
            "error": {
                "type": errors.UNKNOWN_TOOL,
                "message": f"unknown method: {method}",
            }
        }

    def _state(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = self.seeder.reset_state(self.server_name, session_id, 42)
        return self.sessions[session_id]

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params["session_id"])
        name = str(params["name"])
        arguments = params.get("arguments") or {}
        handler = self.handlers.get(name)
        if handler is None:
            return _result(False, None, errors.UNKNOWN_TOOL, f"unknown tool: {name}", False)
        try:
            return handler(session_id, arguments)
        except KeyError as exc:
            return _result(False, None, errors.PRECONDITION_FAILED, str(exc), False)
        except Exception as exc:  # pragma: no cover - defensive server boundary
            return _result(False, None, errors.EXECUTION_ERROR, str(exc), False)


def _result(
    success: bool,
    observation: dict[str, Any] | None,
    error_type: str | None,
    error_message: str,
    state_changed: bool,
) -> dict[str, Any]:
    return {
        "success": success,
        "observation": observation,
        "error_type": error_type,
        "error_message": error_message,
        "state_changed": state_changed,
    }


def serve(server: StatefulToolServer) -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            req_id = request.get("id")
            response = server.handle_request(request.get("method", ""), request.get("params") or {})
            response["id"] = req_id
        except Exception as exc:  # pragma: no cover - server safety net
            response = {
                "id": None,
                "error": {"type": errors.EXECUTION_ERROR, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        sys.stdout.flush()
