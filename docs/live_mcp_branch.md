# Live MCP Branch

This branch is an optional PROVE-style live environment path. It is isolated
from the default replay GRPO training path.

Default training still uses:

```text
Toucan / EpisodeSeed
-> ReplayMCPExecutor
-> src/reward/schemashift_reward_fn.py
-> verl GRPO
```

The live branch uses:

```text
src.live_mcp.api.LiveMCPBranch
-> subprocess stdio calendar / shopping servers
-> grounded LiveTask generation
-> offline deterministic rollout smoke
-> trace + execution-aware reward
```

## Public API

```python
from src.live_mcp.api import LiveMCPBranch, load_live_tasks

with LiveMCPBranch.from_suite("configs/live_mcp/suite_mvp.yaml") as branch:
    summary = branch.generate_tasks_to_file(
        output_path="data/live_mcp/tasks/live_mcp_mvp.jsonl",
        server_name="all",
        count=20,
        seed=42,
    )

tasks = load_live_tasks("data/live_mcp/tasks/live_mcp_mvp.jsonl")
with LiveMCPBranch.from_suite("configs/live_mcp/suite_mvp.yaml") as branch:
    traces, smoke = branch.run_oracle_smoke(
        tasks=tasks,
        server_name="calendar",
        num_tasks=10,
        seed=42,
    )
```

## CLI

```bash
python scripts/generate_live_mcp_tasks.py \
  --suite configs/live_mcp/suite_mvp.yaml \
  --output data/live_mcp/tasks/live_mcp_mvp.jsonl \
  --num-tasks 20 \
  --seed 42

python scripts/run_live_mcp_smoke.py \
  --suite configs/live_mcp/suite_mvp.yaml \
  --tasks data/live_mcp/tasks/live_mcp_mvp.jsonl \
  --server calendar \
  --num-tasks 10 \
  --seed 42
```

## Boundaries

- Do not import `src.live_mcp.api` from `scripts/train_grpo.py`,
  `src/reward/schemashift_reward_fn.py`, or the default GRPO shell scripts.
- Do not replace the existing next-action reward with live trajectory reward.
- Keep `configs/live_mcp/suite_mvp.yaml` default backend as `replay`.
- Treat live traces as generated data under `data/live_mcp/`, not committed code.
