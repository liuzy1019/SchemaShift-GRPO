#!/usr/bin/env python
"""Run an offline Live MCP rollout smoke with subprocess stdio servers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live_mcp.api import LiveMCPBranch, load_live_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="configs/live_mcp/suite_mvp.yaml")
    parser.add_argument("--tasks", default="data/live_mcp/tasks/live_mcp_mvp.jsonl")
    parser.add_argument("--server", default="calendar", choices=["all", "calendar", "shopping"])
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_live_tasks(Path(args.tasks))
    with LiveMCPBranch.from_suite(args.suite) as branch:
        _, summary = branch.run_oracle_smoke(
            tasks=tasks,
            server_name=args.server,
            num_tasks=args.num_tasks,
            seed=args.seed,
        )
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
