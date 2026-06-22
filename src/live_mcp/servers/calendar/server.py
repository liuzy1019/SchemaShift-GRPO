"""Stateful calendar server for Live MCP smoke tests."""

from __future__ import annotations

from typing import Any

from src.live_mcp.server_base import StatefulToolServer, _result, serve


TOOLS = [
    {
        "name": "list_events",
        "description": "List calendar events.",
        "input_schema": {
            "type": "object",
            "properties": {"date_range": {"type": "string"}},
            "required": [],
        },
        "annotations": {"readonly": True, "mutating": False},
    },
    {
        "name": "create_event",
        "description": "Create a calendar event.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "attendees": {"type": "array"},
            },
            "required": ["title", "start_time", "end_time"],
        },
        "annotations": {"mutating": True},
    },
    {
        "name": "update_event",
        "description": "Update an existing calendar event.",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "fields": {"type": "object"},
            },
            "required": ["event_id", "fields"],
        },
        "annotations": {"mutating": True},
    },
    {
        "name": "delete_event",
        "description": "Delete an existing calendar event.",
        "input_schema": {
            "type": "object",
            "properties": {"event_id": {"type": "string"}},
            "required": ["event_id"],
        },
        "annotations": {"mutating": True},
    },
]


class CalendarServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("calendar", TOOLS)
        self.handlers = {
            "list_events": self.list_events,
            "create_event": self.create_event,
            "update_event": self.update_event,
            "delete_event": self.delete_event,
        }

    def list_events(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        events = list(state["events"].values())
        return _result(True, {"events": events}, None, "", False)

    def create_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        event_id = f"evt_{state['next_event_num']:03d}"
        state["next_event_num"] += 1
        event = {
            "event_id": event_id,
            "title": arguments["title"],
            "start_time": arguments["start_time"],
            "end_time": arguments["end_time"],
            "attendees": list(arguments.get("attendees", [])),
        }
        state["events"][event_id] = event
        return _result(True, {"event": event}, None, "", True)

    def update_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        event_id = arguments["event_id"]
        if event_id not in state["events"]:
            raise KeyError(f"event not found: {event_id}")
        allowed = {"title", "start_time", "end_time", "attendees"}
        for key, value in arguments["fields"].items():
            if key in allowed:
                state["events"][event_id][key] = value
        return _result(True, {"event": state["events"][event_id]}, None, "", True)

    def delete_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        event_id = arguments["event_id"]
        if event_id not in state["events"]:
            raise KeyError(f"event not found: {event_id}")
        event = state["events"].pop(event_id)
        return _result(True, {"deleted_event": event}, None, "", True)


if __name__ == "__main__":
    serve(CalendarServer())
