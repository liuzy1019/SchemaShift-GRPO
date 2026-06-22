#!/usr/bin/env python3
"""ToolACE decision_steps 全量统计。

补齐 mcp_tools_rl_project_plan.md 要求的数据统计项，输出到 data/toolace/processed/stats_full.json。
统计项：
  - 基础计数（原始样本、有效样本、decision steps、multi-step）
  - 有 tool_output 的样本数（messages 中含 tool role）
  - 并行调用比例（ground_truth_calls 中有 2+ 个 call 的 step）
  - 工具数分布（per sample 的 tools 列表长度）
  - 工具域分布（按工具名前缀/关键词粗分域）
  - 参数数量分布（ground_truth_calls 中 arguments 的 key 数）
  - enum 参数比例（tools schema 中含 enum 的参数占比）
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "toolace" / "processed" / "decision_steps.jsonl"
OUTPUT_PATH = DATA_PATH.parent / "stats_full.json"


def has_tool_output(messages: list[dict]) -> bool:
    """检查 messages 中是否包含 tool output（role=tool 或 role=function）。"""
    return any(m.get("role") in ("tool", "function") for m in messages)


def count_enum_params(tools: list[dict]) -> tuple[int, int]:
    """统计工具 schema 中 enum 参数数和总参数数。"""
    total_params = 0
    enum_params = 0
    for tool in tools:
        params = tool.get("parameters", {})
        properties = params.get("properties", {})
        for prop_name, prop_def in properties.items():
            total_params += 1
            if "enum" in prop_def:
                enum_params += 1
    return enum_params, total_params


def extract_domain_hint(tool_name: str) -> str:
    """从工具名中提取粗略领域标签。"""
    name_lower = tool_name.lower()
    domain_keywords = {
        "finance": ["stock", "market", "trading", "finance", "crypto", "sec", "invest", "bank", "payment"],
        "travel": ["travel", "flight", "hotel", "booking", "trip", "airport"],
        "weather": ["weather", "climate", "temperature", "forecast"],
        "file": ["file", "directory", "folder", "download", "upload", "storage"],
        "message": ["message", "email", "sms", "chat", "notification", "mail"],
        "search": ["search", "query", "find", "lookup", "get"],
        "math": ["math", "calculate", "compute", "convert", "unit"],
        "social": ["social", "post", "tweet", "comment", "like", "follow"],
        "code": ["code", "compile", "debug", "git", "repo", "api"],
        "health": ["health", "medical", "doctor", "patient", "drug"],
    }
    for domain, keywords in domain_keywords.items():
        if any(kw in name_lower for kw in keywords):
            return domain
    return "other"


def main():
    if not DATA_PATH.exists():
        print(f"错误: 数据文件不存在: {DATA_PATH}")
        sys.exit(1)

    print(f"读取数据: {DATA_PATH}")

    # 统计变量
    total_steps = 0
    samples_with_tool_output = 0
    parallel_call_steps = 0  # ground_truth_calls 有 2+ 个 call
    tool_count_dist = Counter()  # num_tools 分布
    arg_count_dist = Counter()  # 每个 call 的参数数量
    domain_dist = Counter()  # 工具域分布
    total_enum_params = 0
    total_all_params = 0
    step_dist = Counter()  # total_steps_in_sample 分布
    gt_call_count_dist = Counter()  # 每步 ground_truth_calls 数量

    # 按 sample 聚合
    sample_ids = set()  # 去重用 (task_id 去掉 _step_N)
    multi_step_samples = set()

    with open(DATA_PATH) as f:
        for line in f:
            d = json.loads(line)
            total_steps += 1

            task_id = d["task_id"]
            # 提取 sample_id: toolace_XXXXX
            sample_id = "_".join(task_id.split("_")[:2])  # toolace_00000
            sample_ids.add(sample_id)

            total_steps_in_sample = d.get("total_steps_in_sample", 1)
            if total_steps_in_sample > 1:
                multi_step_samples.add(sample_id)
            step_dist[total_steps_in_sample] += 1

            # tool_output 检查
            messages = d.get("messages", [])
            if has_tool_output(messages):
                samples_with_tool_output += 1

            # 并行调用
            gt_calls = d.get("ground_truth_calls", [])
            num_gt_calls = len(gt_calls)
            gt_call_count_dist[num_gt_calls] += 1
            if num_gt_calls >= 2:
                parallel_call_steps += 1

            # 工具数
            tools = d.get("tools", [])
            num_tools = d.get("num_tools", len(tools))
            tool_count_dist[num_tools] += 1

            # 参数数量
            for call in gt_calls:
                args = call.get("arguments", {})
                arg_count_dist[len(args)] += 1

            # 域分布
            for tool in tools:
                domain = extract_domain_hint(tool.get("name", ""))
                domain_dist[domain] += 1

            # enum 参数
            enum_count, param_count = count_enum_params(tools)
            total_enum_params += enum_count
            total_all_params += param_count

    # 汇总
    stats = {
        "basic": {
            "total_decision_steps": total_steps,
            "unique_samples": len(sample_ids),
            "multi_step_samples": len(multi_step_samples),
            "avg_steps_per_sample": round(total_steps / len(sample_ids), 4) if sample_ids else 0,
        },
        "tool_output": {
            "steps_with_tool_output": samples_with_tool_output,
            "ratio": round(samples_with_tool_output / total_steps, 4) if total_steps else 0,
            "note": "messages 中含 role=tool/function 的 step 数",
        },
        "parallel_calls": {
            "steps_with_parallel_calls": parallel_call_steps,
            "ratio": round(parallel_call_steps / total_steps, 4) if total_steps else 0,
            "gt_call_count_distribution": dict(sorted(gt_call_count_dist.items())),
        },
        "tool_count_distribution": {
            "distribution": dict(sorted(tool_count_dist.items())),
            "mean": round(sum(k * v for k, v in tool_count_dist.items()) / total_steps, 2) if total_steps else 0,
        },
        "domain_distribution": dict(domain_dist.most_common(20)),
        "argument_count_distribution": {
            "distribution": dict(sorted(arg_count_dist.items())),
            "mean": round(
                sum(k * v for k, v in arg_count_dist.items()) / sum(arg_count_dist.values()), 2
            ) if arg_count_dist else 0,
        },
        "enum_params": {
            "total_enum_params": total_enum_params,
            "total_all_params": total_all_params,
            "enum_ratio": round(total_enum_params / total_all_params, 4) if total_all_params else 0,
        },
        "step_distribution": dict(sorted(step_dist.items())),
    }

    # 输出
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n统计结果已写入: {OUTPUT_PATH}")
    print(f"\n{'='*60}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
