#!/usr/bin/env python3
"""Task generation for GRPO training via PROVE-style state-machine teacher.

The teacher uses LLM-in-the-loop at every turn: the LLM sees the full domain
context (tools, live state, execution history) and decides the next action
(tool_call with real arguments, or terminal). The resulting oracle trace is
built by actual execution against live MCP servers.

Deployment modes:
  1. Local transformers:  --model models/Qwen3-8B
  2. vLLM server:         --model Qwen3-8B --api-base http://localhost:8000/v1

PROVE-aligned defaults:
  - Difficulty mix: complete=60%, missing=20%, minimal=20%
  - Irrelevance ratio: 5%
  - Distractor rate: 30% (injects 3-8 irrelevant tools)
  - Missing function rate: 20% (hides one required tool)
  - Enum stripping: 30% per domain
  - Jaccard dedup threshold: 0.70
  - Turn-decay schedule: chain_len-based (min~2, max~6 with perturbations)
  - Personas: 10 role templates, reference dates: 10 anchors
  - Recovery: explicit retry_same / retry_alt / give_up states
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate GRPO GRPO training data with LLM teacher"
    )
    p.add_argument("--count", type=int, default=500,
                    help="Number of training tasks to generate")
    p.add_argument("--val-count", type=int, default=50,
                    help="Number of validation tasks to generate")
    p.add_argument("--domain", default="all",
                    help="Domain (all, banking, calendar, etc.). "
                         "Comma-separated list supported, e.g. calendar,shopping,banking")
    p.add_argument("--model", required=True,
                    help="Model name (vLLM served name) or local path (models/Qwen/Qwen3-8B). "
                         "vLLM mode: must match --served-model-name from vLLM startup. "
                         "Local mode: absolute or relative path to model directory.")
    p.add_argument("--api-base", default=None,
                    help="OpenAI-compatible API base URL (local transformers if unset)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", default="data/train.parquet",
                    help="Training data output path")
    p.add_argument("--val-output", default="data/val.parquet",
                    help="Validation data output path")
    p.add_argument("--suite", default="configs/live_mcp/suite_mvp.yaml",
                    help="Suite config path")
    p.add_argument("--irrelevance-ratio", type=float, default=0.05,
                    help="Fraction of tasks that require report_error (0 to disable)")
    p.add_argument("--distractor-rate", type=float, default=0.30,
                    help="Probability of injecting distractor tools (0 to disable)")
    p.add_argument("--missing-function-rate", type=float, default=0.20,
                    help="Probability of hiding a required tool (0 to disable)")
    p.add_argument("--device", type=int, default=None,
                    help="GPU device ID for local inference (default: auto). "
                         "Use with CUDA_VISIBLE_DEVICES for multi-GPU data-parallel via "
                         "scripts/generate_data_parallel.sh")
    p.add_argument("--experiment-tag", default=None,
                    help="Tag for experiment tracking. If set, writes config.json and "
                         "result.json to data/experiments/{YYYY-MM-DD}_{tag}/")
    p.add_argument("--log-file", default=None,
                    help="Write all logs to this file (auto-flushed, avoids pipe buffering)")
    return p


def generate_data(args: argparse.Namespace):
    """Generate GRPO training data with LLM teacher."""
    from src.live_mcp.api import LiveMCPBranch

    # ── Validate parameters ──
    if args.count < 1:
        raise ValueError(f"--count must be >= 1, got {args.count}")
    if args.val_count < 0:
        raise ValueError(f"--val-count must be >= 0, got {args.val_count}")
    for name, val in [("irrelevance_ratio", args.irrelevance_ratio),
                       ("distractor_rate", args.distractor_rate),
                       ("missing_function_rate", args.missing_function_rate)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"--{name} must be in [0.0, 1.0], got {val}")
    if Path(args.output).resolve() == Path(args.val_output).resolve():
        raise ValueError(
            f"--output and --val-output point to the same file: {args.output}"
        )

    # 如果指定了 --log-file，添加文件 sink 确保实时可见
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            enqueue=False,
            catch=True,
        )

    start_time = datetime.now(timezone.utc)

    print(f"[generate_data] Target: {args.count} train + {args.val_count} val tasks, domain={args.domain}, model={args.model}")
    logger.info(f"Generating GRPO data: {args.count} train + {args.val_count} val tasks")
    logger.info(f"  Domain: {args.domain}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Difficulty mix: complete=60%, missing=20%, minimal=20%")

    branch = LiveMCPBranch.from_suite(args.suite)
    difficulty_mix = {"complete": 0.6, "missing": 0.2, "minimal": 0.2}

    try:
        branch.start()
        print(f"[generate_data] Generating {args.count} train tasks...", flush=True)
        train_tasks = branch.generate_tasks_llm(
            server_name=args.domain, count=args.count, seed=args.seed,
            difficulty_mix=difficulty_mix, model_path=args.model,
            api_base=args.api_base,
            device=args.device,
            irrelevance_ratio=args.irrelevance_ratio,
            distractor_rate=args.distractor_rate,
            missing_function_rate=args.missing_function_rate,
        )
        print(f"[generate_data] Train done: {len(train_tasks)} tasks", flush=True)
        print(f"[generate_data] Generating {args.val_count} val tasks...", flush=True)
        val_tasks = branch.generate_tasks_llm(
            server_name=args.domain, count=args.val_count, seed=args.seed + 10000,
            difficulty_mix=difficulty_mix, model_path=args.model,
            api_base=args.api_base,
            device=args.device,
            irrelevance_ratio=args.irrelevance_ratio,
            distractor_rate=args.distractor_rate,
            missing_function_rate=args.missing_function_rate,
        )
        print(f"[generate_data] Val done: {len(val_tasks)} tasks", flush=True)
        all_rows = _tasks_to_rows(train_tasks, args.seed)
        val_rows = _tasks_to_rows(val_tasks, args.seed + 10000)
    finally:
        branch.stop()

    df_train = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    df_val = pd.DataFrame(val_rows) if val_rows else pd.DataFrame()

    # Ensure output parent directories exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(Path(args.output), index=False)
    df_val.to_parquet(Path(args.val_output), index=False)

    if args.experiment_tag:
        _save_experiment_record(args, df_train, df_val, start_time)

    _print_stats(df_train, df_val, Path(args.output), Path(args.val_output), args)


def _tasks_to_rows(tasks: list, base_seed: int) -> list[dict]:
    """Convert LiveTask list to verl-compatible data rows."""
    rows = []
    for i, task in enumerate(tasks):
        # Determine visible tools — use task-provided tools, fall back to required
        visible_tools = task.visible_tools if task.visible_tools else []
        if not visible_tools:
            logger.warning(
                f"Skipping task {task.task_id}: no visible_tools "
                f"(required_tools={task.required_tools}, "
                f"oracle_calls={len(task.oracle_program.calls) if task.oracle_program else 0})"
            )
            continue  # Skip tasks without tool schemas

        tools_desc_lines = []
        for t in visible_tools:
            name = t.get("name", "")
            desc = t.get("description", "")
            params = t.get("input_schema", {}).get("properties", {})
            required = t.get("input_schema", {}).get("required", [])
            tools_desc_lines.append(f"- {name}: {desc}")
            for pname, pinfo in params.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                tools_desc_lines.append(
                    f"    - {pname} ({ptype}{req}): {pdesc}"
                )
        tools_block = "\n".join(tools_desc_lines)

        domain = task.target_servers[0] if task.target_servers else "unknown"

        system_prompt = (
            f"You are a helpful assistant with access to the following tools. "
            f"Use them when needed to answer the user's question.\n\n"
            f"Available tools:\n{tools_block}\n\n"
            f"Response format:\n"
            f'- To call a tool: <tool_call>{{"name": "tool_name", "arguments": {{...}}}}</tool_call>\n'
            f"- To give final answer: <final_answer>your answer</final_answer>\n"
            f"- To report error: <report_error>error description</report_error>\n"
            f"- To ask clarification: <ask_clarification>your question</ask_clarification>"
        )

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.user_prompt},
        ]

        task_type = task.task_type
        has_distractors = task.metadata.get("has_distractors", False)
        has_missing_func = task.metadata.get("has_missing_function", False)

        if has_missing_func:
            scenario_type = "missing_function"
            perturbation_level = "hard"
        elif has_distractors:
            scenario_type = "distractor"
            perturbation_level = "medium"
        else:
            scenario_type = task_type or "normal"
            perturbation_level = task.difficulty

        # 每个 task 独立一组：verl repeat(N) 后同一 prompt 的 N 个 rollout
        # 自然形成一个 group，回归标准 GRPO per-prompt 对比语义
        group_id = task.task_id

        # Serialize oracle program calls (with real arguments) for reward matching
        oracle_calls_serialized = []
        if task.oracle_program and task.oracle_program.calls:
            for oc in task.oracle_program.calls:
                oracle_calls_serialized.append({
                    "tool_name": oc.tool_name,
                    "arguments": dict(oc.arguments) if oc.arguments else {},
                    # P0-3: 保留 action 字段。澄清任务是 action="clarification"，
                    # 丢失后会被 reward 当成普通 tool_call，导致模型输出
                    # ask_clarification 时无法匹配。
                    "action": getattr(oc, "action", "tool_call"),
                })

        success_criteria = (
            list(task.oracle_program.success_criteria)
            if task.oracle_program and task.oracle_program.success_criteria
            else task.success_criteria if hasattr(task, "success_criteria") else []
        )

        # P0-1 fix: success_criteria is a list[dict] whose 'value' field holds
        # mixed types (str status, numeric balance, etc.). Storing the raw list
        # in a Parquet dict column makes pyarrow infer a single value type and
        # crash with ArrowInvalid ("Could not convert 'paid' with type str:
        # tried to convert to double"). Serialize to JSON string for safe
        # round-trip; reward side parses it back via json.loads.
        success_criteria_json = json.dumps(
            success_criteria, ensure_ascii=False, default=str
        )

        extra_info = {
            "task_id": task.task_id,
            "domain": domain,
            "target_servers": task.target_servers,
            "required_tools": task.required_tools,
            "session_seed": task.session_seed,
            "budget": task.max_turns,
            "perturbation_level": perturbation_level,
            "scenario_type": scenario_type,
            "group_id": group_id,
            "uid": task.task_id,
            "has_distractors": has_distractors,
            "has_missing_function": has_missing_func,
            "generation_method": task.metadata.get("generation_method", "task_planner"),
            "oracle_calls": oracle_calls_serialized,
            "success_criteria": success_criteria_json,
            "hidden_tools": list(task.hidden_tools) if task.hidden_tools else [],
        }

        row = {
            "prompt": json.dumps(prompt, ensure_ascii=False),
            "data_source": "live_mcp_state_machine",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "task_id": task.task_id,
                    "oracle_calls": oracle_calls_serialized,
                    "success_criteria": success_criteria_json,
                    "required_tools": task.required_tools,
                },
            },
            "extra_info": extra_info,
            "uid": extra_info["uid"],
            "group_id": group_id,
            "perturbation_level": perturbation_level,
            "scenario_type": scenario_type,
        }
        rows.append(row)

    return rows


def _save_experiment_record(args, df_train, df_val, start_time: datetime):
    """Save experiment config and results to data/experiments/{date}_{tag}/."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    exp_dir = PROJECT_ROOT / "data" / "experiments" / f"{today}_{args.experiment_tag}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Git commit hash
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, text=True,
        ).strip()
    except Exception:
        git_commit = "unknown"

    # GPU info
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0] if shutil.which("nvidia-smi") else "unknown"
    except Exception:
        gpu_info = "unknown"

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    # config.json
    config = {
        "run_id": f"{today}_{args.experiment_tag}",
        "command": " ".join(sys.argv),
        "model": args.model,
        "api_base": args.api_base or "local",
        "domain": args.domain,
        "count": args.count,
        "val_count": args.val_count,
        "seed": args.seed,
        "distractor_rate": args.distractor_rate,
        "missing_function_rate": args.missing_function_rate,
        "irrelevance_ratio": args.irrelevance_ratio,
        "difficulty_mix": {"complete": 0.6, "missing": 0.2, "minimal": 0.2},
        "git_commit": git_commit,
        "gpu_model": gpu_info,
        "timestamp": start_time.isoformat(),
    }
    (exp_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )

    # result.json
    if len(df_train) > 0:
        domain_dist = (
            df_train["extra_info"].apply(lambda x: x.get("domain")).value_counts().to_dict()
        )
        scenario_dist = df_train["scenario_type"].value_counts().to_dict()
        difficulty_dist = df_train["perturbation_level"].value_counts().to_dict()
    else:
        domain_dist = {}
        scenario_dist = {}
        difficulty_dist = {}

    result = {
        "run_id": config["run_id"],
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "yield": round(len(df_train) / args.count, 3) if args.count > 0 else 0.0,
        "duration_seconds": round(duration, 1),
        "domain_distribution": domain_dist,
        "scenario_distribution": scenario_dist,
        "difficulty_distribution": difficulty_dist,
        "timestamp": end_time.isoformat(),
    }
    (exp_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    logger.info(f"Experiment record saved: {exp_dir}")


def _print_stats(df_train, df_val, train_path, val_path, args):
    """Print generation statistics."""
    print(f"\n{'='*60}")
    print(f"GRPO Data Generation Complete")
    print(f"{'='*60}")
    print(f"Train: {len(df_train)} rows → {train_path}")
    print(f"Val:   {len(df_val)} rows → {val_path}")

    if len(df_train) == 0:
        print("\nWARNING: No training data generated!")
        return

    # Per-domain stats
    domains = set()
    for _, row in df_train.iterrows():
        domains.add(row["extra_info"]["domain"])
    for domain in sorted(domains):
        domain_rows = df_train[
            df_train["extra_info"].apply(lambda x: x.get("domain") == domain)
        ]
        print(f"\n  {domain}: {len(domain_rows)} rows")
        for stype in ["normal", "distractor", "missing_function", "irrelevant", "task_planner"]:
            count = len(
                domain_rows[domain_rows["scenario_type"] == stype]
            )
            if count > 0:
                print(f"    {stype}: {count}")

    # Difficulty distribution
    difficulty_dist = df_train["perturbation_level"].value_counts().to_dict()
    print(f"\n  Difficulty distribution: {difficulty_dist}")

    # Sample queries (show 3)
    print("\n  Sample queries:")
    for i in range(min(3, len(df_train))):
        ei = df_train.iloc[i]["extra_info"]
        prompt_raw = df_train.iloc[i]["prompt"]
        prompt = json.loads(prompt_raw) if isinstance(prompt_raw, str) else prompt_raw
        user_msg = prompt[1]["content"] if isinstance(prompt, list) and len(prompt) > 1 else str(prompt_raw)[:100]
        print(
            f"    [{i}] domain={ei['domain']} scenario={ei['scenario_type']} "
            f"query=\"{str(user_msg)[:100]}...\""
        )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    generate_data(args)
