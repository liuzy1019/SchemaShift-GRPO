"""Rollout trace recording."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.live_mcp.types import LiveTask, RolloutTrace, TraceTurn, to_plain


class TraceRecorder:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def start(self, task: LiveTask, model_name: str) -> RolloutTrace:
        trace_id = hashlib.sha1(f"{task.task_id}:{task.session_id}:{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:16]
        return RolloutTrace(
            trace_id=trace_id,
            task_id=task.task_id,
            session_id=task.session_id,
            model_name=model_name,
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=None,
            turns=[],
            final_status="running",
            reward={},
            metadata={"task": to_plain(task), "visible_tools_hash": _hash_json(task.visible_tools)},
        )

    def append_turn(self, trace: RolloutTrace, turn: TraceTurn) -> None:
        trace.turns.append(turn)

    def finish(self, trace: RolloutTrace, final_status: str, reward: dict[str, float]) -> None:
        trace.final_status = final_status
        trace.reward = reward
        trace.ended_at = datetime.now(timezone.utc).isoformat()

    def save(self, trace: RolloutTrace) -> Path:
        suite_name = str(trace.metadata.get("task", {}).get("suite_name", "live_mcp"))
        date = datetime.now(timezone.utc).date().isoformat()
        path = self.output_dir / suite_name / date / f"{trace.trace_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(to_plain(trace), f, ensure_ascii=True, indent=2)
        return path


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:16]


def _hash_json(value: object) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16]
