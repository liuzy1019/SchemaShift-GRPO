"""将 EpisodeSeed 数据转为 verl GRPO 训练所需的 parquet 格式。

verl 的 RLHFDataset 需要 parquet 文件包含以下字段：
  - prompt: list[dict]（chat messages 格式）
  - data_source: str（用于选择 reward function）
  - reward_model.ground_truth: str（oracle 信息，JSON 序列化）
  - extra_info: dict（perturbation_level, name_map 等）

Usage:
    python scripts/prepare_grpo_data.py \
        --episode_seeds data/episode_seeds.jsonl \
        --output data/grpo_train.parquet \
        --max_samples 100
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.episode_schema import EpisodeSeed


def episode_to_verl_row(episode: EpisodeSeed) -> dict:
    """将单个 EpisodeSeed 转为 verl parquet 行。"""
    # 构建 prompt（system + tools + user messages）
    tools_desc = format_tools_for_prompt(episode.tools_snapshot)
    system_msg = (
        "You are a helpful assistant with access to the following tools. "
        "Use them when needed to answer the user's question.\n\n"
        f"Available tools:\n{tools_desc}\n\n"
        "Response format:\n"
        "- To call a tool: <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>\n"
        "- To give final answer: <final_answer>your answer</final_answer>\n"
        "- To report error: <report_error>error description</report_error>\n"
        "- To ask clarification: <ask_clarification>your question</ask_clarification>"
    )

    messages = [{"role": "system", "content": system_msg}]
    for msg in episode.initial_messages:
        messages.append(msg)

    # 构建 ground_truth（oracle 信息）
    # oracle_trace 是 dict 列表或 OracleStep 对象
    oracle_actions = []
    first_action_type = ""
    first_tool_name = ""
    for step in episode.oracle_trace:
        action_type = step.get("action_type", "tool_call") if isinstance(step, dict) else getattr(step, "action_type", "tool_call")
        tool_name = step.get("tool_name") if isinstance(step, dict) else getattr(step, "tool_name", None)
        arguments = step.get("arguments", {}) if isinstance(step, dict) else getattr(step, "arguments", {})
        calls = step.get("calls", []) if isinstance(step, dict) else getattr(step, "calls", [])
        expected_content = step.get("expected_content", "") if isinstance(step, dict) else getattr(step, "expected_content", "")
        match_mode = step.get("match_mode", "set") if isinstance(step, dict) else getattr(step, "match_mode", "set")

        action_data = {"action_type": action_type}

        if action_type == "tool_call" and tool_name:
            action_data["tool_calls"] = [{"name": tool_name, "arguments": arguments}]
            action_data["match_mode"] = "exact"
            first_tool_name = first_tool_name or tool_name
        elif action_type == "parallel_tool_call" and calls:
            # 转换 parallel calls 为 reward 可识别的 tool_calls 格式
            action_data["action_type"] = "tool_call"
            action_data["tool_calls"] = [
                {"name": c.get("tool_name", c.get("name", "")), "arguments": c.get("arguments", {})}
                for c in calls
            ]
            action_data["match_mode"] = match_mode
            first_tool_name = first_tool_name or ",".join(
                c.get("tool_name", c.get("name", "")) for c in calls
            )
        elif action_type in ("final_answer", "report_error", "ask_clarification"):
            if expected_content:
                action_data["final_answer"] = expected_content

        first_action_type = first_action_type or action_data["action_type"]
        oracle_actions.append(action_data)

    ground_truth = json.dumps({
        "oracle_actions": oracle_actions,
        "episode_type": episode.episode_type,
    }, ensure_ascii=False)

    perturbation_level = episode.metadata.get("perturbation_level", "none")
    scenario_type = episode.episode_type
    group_id = episode.metadata.get("group_id") or episode.episode_id
    uid = f"{group_id}___{perturbation_level}___{scenario_type}"

    # 构建 extra_info（去掉空 dict 字段避免 parquet 空 struct 问题）
    extra_info = {
        "perturbation_level": perturbation_level,
        "scenario_type": scenario_type,
        "episode_id": episode.episode_id,
        "group_id": group_id,
        "uid": uid,
        "action_type": first_action_type,
        "tool_name": first_tool_name,
    }
    # 只在非空时添加 map 字段
    name_map = episode.metadata.get("name_map")
    if name_map:
        extra_info["name_map"] = name_map
    enum_map = episode.metadata.get("enum_map")
    if enum_map:
        extra_info["enum_map"] = enum_map

    return {
        "prompt": messages,
        "data_source": "schemashift",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
        # Keep these at top-level so verl's non_tensor_batch can preserve them
        # through _get_gen_batch and the schemashift_grpo estimator.
        "uid": uid,
        "group_id": group_id,
        "perturbation_level": perturbation_level,
        "scenario_type": scenario_type,
        "action_type": first_action_type,
        "tool_name": first_tool_name,
    }


def format_tools_for_prompt(tools: list[dict]) -> str:
    """格式化工具描述。"""
    lines = []
    for tool in tools:
        name = tool.get("name", tool.get("function", {}).get("name", "unknown"))
        desc = tool.get("description", tool.get("function", {}).get("description", ""))
        params = tool.get("parameters", tool.get("function", {}).get("parameters", {}))

        lines.append(f"- {name}: {desc}")
        if params and isinstance(params, dict):
            properties = params.get("properties", {})
            required = params.get("required", [])
            for pname, pinfo in properties.items():
                req_mark = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                lines.append(f"    - {pname} ({ptype}{req_mark}): {pdesc}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="EpisodeSeed → verl parquet")
    # 输入输出
    parser.add_argument("--episode_seeds", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--val_output", type=str, default=None)
    args = parser.parse_args()

    # 读取 episode seeds
    episodes = []
    input_path = Path(args.episode_seeds)
    if input_path.suffix == ".jsonl":
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    episodes.append(EpisodeSeed(**data))
    elif input_path.suffix == ".json":
        with open(input_path) as f:
            data_list = json.load(f)
            for data in data_list:
                episodes.append(EpisodeSeed(**data))
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    if args.max_samples > 0:
        episodes = episodes[:args.max_samples]

    print(f"Loaded {len(episodes)} episodes")

    # 转换
    rows = [episode_to_verl_row(ep) for ep in episodes]
    df = pd.DataFrame(rows)

    # 分割 train/val
    if args.val_split > 0 and args.val_output:
        n_val = max(1, int(len(df) * args.val_split))
        val_df = df.tail(n_val)
        train_df = df.head(len(df) - n_val)

        train_df.to_parquet(args.output, index=False)
        val_df.to_parquet(args.val_output, index=False)
        print(f"Train: {len(train_df)} → {args.output}")
        print(f"Val: {len(val_df)} → {args.val_output}")
    else:
        df.to_parquet(args.output, index=False)
        print(f"Output: {len(df)} → {args.output}")


if __name__ == "__main__":
    main()
