#!/usr/bin/env python3
"""Toucan 数据子集 Inspection 报告。

对下载的 Toucan 子集进行定量分析，输出 verifier-readiness report。
这是项目 Phase 1 的核心产出，决定后续 EpisodeSeed 构建策略。

输出指标:
  - parseable_rate: 可解析为结构化 tool_call/observation 对的比例
  - verifier_ready_rate: 有明确 oracle action + expected output 的比例
  - short_episode_rate: ≤ 3 decision turns 的比例
  - tool_call_observation_pairing_rate: tool_call 和 observation 一一配对的比例
  - final_answer_structured_rate: final_answer 可做结构化精确匹配的比例
  - parallel_call_rate: 包含并行 tool call 的比例
  - subset_name_distribution: 子集类型分布
"""

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "toucan"


# ============================================================
# 解析工具
# ============================================================


def load_toucan_data(path: Path) -> list[dict]:
    """加载 JSONL 格式的 Toucan 数据。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_json_field(value: Any) -> Any:
    """解析可能是 JSON 字符串的字段。"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        # 尝试 Python literal eval（Toucan 有些字段用 Python dict 格式）
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def extract_messages(item: dict) -> list[dict] | None:
    """从 Toucan 样本中提取消息列表。

    Toucan SFT 格式: messages 字段是 JSON 字符串。
    """
    raw = item.get("messages")
    if raw is None:
        return None

    messages = parse_json_field(raw)
    if isinstance(messages, list):
        return messages
    return None


def extract_tools_schema(item: dict) -> list[dict] | None:
    """提取工具 schema 定义。

    Toucan SFT 格式: tools 字段是 JSON 字符串。
    """
    raw = item.get("tools")
    if raw is None:
        return None

    tools = parse_json_field(raw)
    if isinstance(tools, list):
        return tools
    return None


def parse_tool_call_content(content: str) -> dict | None:
    """解析 tool_call 消息的 content。

    Toucan 格式: {'name': 'tool_name', 'arguments': '{...}'}
    """
    parsed = parse_json_field(content)
    if isinstance(parsed, dict) and "name" in parsed:
        # 解析 arguments（可能是嵌套的 JSON 字符串）
        args = parsed.get("arguments", {})
        if isinstance(args, str):
            try:
                parsed["arguments_parsed"] = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed["arguments_parsed"] = ast.literal_eval(args)
                except (ValueError, SyntaxError):
                    parsed["arguments_parsed"] = None
        else:
            parsed["arguments_parsed"] = args
        return parsed
    return None


def count_decision_turns(messages: list[dict]) -> int:
    """计算 decision turns 数量。

    在 Toucan 中，一个 decision turn = 一组连续的 tool_call 消息
    或一个 final assistant 消息。
    """
    turns = 0
    prev_role = None
    for msg in messages:
        role = msg.get("role", "")
        if role == "tool_call" and prev_role != "tool_call":
            # 新的 tool-call turn 开始
            turns += 1
        elif role == "assistant" and prev_role != "assistant":
            # assistant 回复（可能是 final answer 或中间说明）
            turns += 1
        prev_role = role
    return turns


def has_parallel_calls(messages: list[dict]) -> bool:
    """检查是否包含并行 tool call。

    并行 = 连续多个 tool_call 消息之间没有 tool_response。
    """
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "tool_call":
            # 计算连续 tool_call 数量
            consecutive = 0
            j = i
            while j < len(messages) and messages[j].get("role") == "tool_call":
                consecutive += 1
                j += 1
            if consecutive > 1:
                return True
            i = j
        else:
            i += 1
    return False


def check_tool_observation_pairing(messages: list[dict]) -> tuple[bool, int, int]:
    """检查 tool_call 和 tool_response 是否一一配对。

    Returns:
        (is_paired, n_tool_calls, n_tool_responses)
    """
    n_calls = sum(1 for m in messages if m.get("role") == "tool_call")
    n_responses = sum(1 for m in messages if m.get("role") == "tool_response")

    # 严格配对: 每个 tool_call 后面紧跟一个 tool_response
    paired = True
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "tool_call":
            # 下一条应该是 tool_response
            if i + 1 < len(messages) and messages[i + 1].get("role") == "tool_response":
                i += 2
            else:
                paired = False
                break
        else:
            i += 1

    return paired, n_calls, n_responses


def check_arguments_parseable(messages: list[dict]) -> tuple[int, int]:
    """检查 tool_call 的 arguments 是否可解析。

    Returns:
        (parseable_count, total_count)
    """
    total = 0
    parseable = 0
    for msg in messages:
        if msg.get("role") == "tool_call":
            total += 1
            parsed = parse_tool_call_content(msg.get("content", ""))
            if parsed and parsed.get("arguments_parsed") is not None:
                parseable += 1
    return parseable, total


def check_final_answer(messages: list[dict]) -> tuple[bool, bool]:
    """检查是否有 final answer 以及是否可结构化匹配。

    Returns:
        (has_final_answer, is_structured)
    """
    # 找最后一个 assistant 消息（不是 tool_call）
    last_assistant = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg
            break

    if last_assistant is None:
        return False, False

    content = last_assistant.get("content", "")
    if not content or not content.strip():
        return False, False

    # 判断是否可结构化匹配
    # 短回答（< 300 字符）更可能是精确匹配的
    is_structured = len(content.strip()) < 300

    return True, is_structured


def extract_tool_names_from_messages(messages: list[dict]) -> list[str]:
    """从消息中提取所有被调用的工具名。"""
    names = []
    for msg in messages:
        if msg.get("role") == "tool_call":
            parsed = parse_tool_call_content(msg.get("content", ""))
            if parsed and "name" in parsed:
                names.append(parsed["name"])
    return names


def extract_mcp_server_from_tool_name(tool_name: str) -> str | None:
    """从工具名中提取 MCP server 信息。

    Toucan 工具名格式通常是: server-name-tool_name
    例如: lotus-wisdom-lotuswisdom
    """
    parts = tool_name.rsplit("-", 1)
    if len(parts) >= 2:
        return parts[0]  # server 部分
    return None


# ============================================================
# Inspection 主逻辑
# ============================================================


def inspect_single_item(item: dict) -> dict[str, Any]:
    """对单条 Toucan 样本进行 inspection。"""
    result = {
        "parseable": False,
        "messages_parseable": False,
        "tools_parseable": False,
        "has_tool_calls": False,
        "has_observations": False,
        "tool_observation_paired": False,
        "verifier_ready": False,
        "short_episode": False,
        "decision_turns": 0,
        "has_parallel_calls": False,
        "has_final_answer": False,
        "final_answer_structured": False,
        "arguments_parseable": False,
        "tool_names": [],
        "mcp_servers": [],
        "subset_name": item.get("subset_name", "unknown"),
        "target_tools": item.get("target_tools", ""),
        "n_messages": 0,
        "n_tool_calls": 0,
        "n_tool_responses": 0,
    }

    # 解析 messages
    messages = extract_messages(item)
    if messages is None:
        return result
    result["messages_parseable"] = True
    result["n_messages"] = len(messages)

    # 解析 tools schema
    tools_schema = extract_tools_schema(item)
    result["tools_parseable"] = tools_schema is not None and len(tools_schema) > 0

    # 整体 parseable = messages + tools 都能解析
    result["parseable"] = result["messages_parseable"] and result["tools_parseable"]

    # Tool calls 和 observations
    has_tool_calls = any(m.get("role") == "tool_call" for m in messages)
    has_observations = any(m.get("role") == "tool_response" for m in messages)
    result["has_tool_calls"] = has_tool_calls
    result["has_observations"] = has_observations

    # Pairing
    paired, n_calls, n_responses = check_tool_observation_pairing(messages)
    result["tool_observation_paired"] = paired
    result["n_tool_calls"] = n_calls
    result["n_tool_responses"] = n_responses

    # Arguments parseable
    args_parseable, args_total = check_arguments_parseable(messages)
    result["arguments_parseable"] = args_total > 0 and args_parseable == args_total

    # Decision turns
    decision_turns = count_decision_turns(messages)
    result["decision_turns"] = decision_turns
    result["short_episode"] = decision_turns <= 3

    # Parallel calls
    result["has_parallel_calls"] = has_parallel_calls(messages)

    # Final answer
    has_final, is_structured = check_final_answer(messages)
    result["has_final_answer"] = has_final
    result["final_answer_structured"] = is_structured

    # Tool names
    tool_names = extract_tool_names_from_messages(messages)
    result["tool_names"] = tool_names

    # MCP servers (从工具名推断)
    servers = set()
    for name in tool_names:
        server = extract_mcp_server_from_tool_name(name)
        if server:
            servers.add(server)
    result["mcp_servers"] = list(servers)

    # Verifier ready 条件:
    # 1. messages 和 tools 都可解析
    # 2. 有 tool calls 且有 observations
    # 3. tool_call 和 tool_response 一一配对
    # 4. arguments 可解析
    # 5. 有 target_tools（知道正确答案）
    result["verifier_ready"] = (
        result["parseable"]
        and result["has_tool_calls"]
        and result["has_observations"]
        and result["tool_observation_paired"]
        and result["arguments_parseable"]
        and bool(result["target_tools"])
    )

    return result


def generate_report(data: list[dict], output_path: Path | None = None) -> dict:
    """生成完整的 inspection report。"""
    total = len(data)
    if total == 0:
        print("错误: 数据为空")
        return {}

    print(f"\n{'='*70}")
    print(f"  Toucan Subset Inspection Report")
    print(f"  样本数: {total}")
    print(f"{'='*70}\n")

    # 逐条 inspect
    results = [inspect_single_item(item) for item in data]

    # 汇总统计
    parseable_count = sum(1 for r in results if r["parseable"])
    messages_parseable = sum(1 for r in results if r["messages_parseable"])
    tools_parseable = sum(1 for r in results if r["tools_parseable"])
    has_tool_calls_count = sum(1 for r in results if r["has_tool_calls"])
    has_observations_count = sum(1 for r in results if r["has_observations"])
    paired_count = sum(1 for r in results if r["tool_observation_paired"])
    verifier_ready_count = sum(1 for r in results if r["verifier_ready"])
    short_episode_count = sum(1 for r in results if r["short_episode"])
    parallel_count = sum(1 for r in results if r["has_parallel_calls"])
    has_final_count = sum(1 for r in results if r["has_final_answer"])
    final_structured_count = sum(1 for r in results if r["final_answer_structured"])
    args_parseable_count = sum(1 for r in results if r["arguments_parseable"])

    # Decision turns 分布
    turn_dist = Counter(r["decision_turns"] for r in results if r["messages_parseable"])

    # Subset name 分布
    subset_dist = Counter(r["subset_name"] for r in results)

    # Tool name 分布
    all_tool_names = []
    for r in results:
        all_tool_names.extend(r["tool_names"])
    tool_name_dist = Counter(all_tool_names)

    # MCP server 分布
    all_servers = []
    for r in results:
        all_servers.extend(r["mcp_servers"])
    server_dist = Counter(all_servers)

    # 输出报告
    report = {
        "total_samples": total,
        "parseable_rate": parseable_count / total,
        "messages_parseable_rate": messages_parseable / total,
        "tools_parseable_rate": tools_parseable / total,
        "has_tool_calls_rate": has_tool_calls_count / total,
        "has_observations_rate": has_observations_count / total,
        "tool_observation_pairing_rate": paired_count / total,
        "arguments_parseable_rate": args_parseable_count / total,
        "verifier_ready_rate": verifier_ready_count / total,
        "short_episode_rate": short_episode_count / total,
        "parallel_call_rate": parallel_count / total,
        "has_final_answer_rate": has_final_count / total,
        "final_answer_structured_rate": final_structured_count / total,
        "decision_turns_distribution": dict(sorted(turn_dist.items())),
        "subset_name_distribution": dict(subset_dist.most_common()),
        "unique_tool_names": len(tool_name_dist),
        "top_30_tools": tool_name_dist.most_common(30),
        "unique_mcp_servers": len(server_dist),
        "top_20_servers": server_dist.most_common(20),
    }

    # 打印核心指标
    print("📊 核心指标:")
    print(f"  {'指标':<40} {'比例':>8} {'数量':>12}")
    print(f"  {'-'*62}")
    print(f"  {'messages_parseable_rate':<40} {messages_parseable/total:.4f} ({messages_parseable}/{total})")
    print(f"  {'tools_parseable_rate':<40} {tools_parseable/total:.4f} ({tools_parseable}/{total})")
    print(f"  {'parseable_rate (both)':<40} {parseable_count/total:.4f} ({parseable_count}/{total})")
    print(f"  {'has_tool_calls_rate':<40} {has_tool_calls_count/total:.4f} ({has_tool_calls_count}/{total})")
    print(f"  {'has_observations_rate':<40} {has_observations_count/total:.4f} ({has_observations_count}/{total})")
    print(f"  {'tool_observation_pairing_rate':<40} {paired_count/total:.4f} ({paired_count}/{total})")
    print(f"  {'arguments_parseable_rate':<40} {args_parseable_count/total:.4f} ({args_parseable_count}/{total})")
    print(f"  {'verifier_ready_rate':<40} {verifier_ready_count/total:.4f} ({verifier_ready_count}/{total})")
    print(f"  {'short_episode_rate (≤3 turns)':<40} {short_episode_count/total:.4f} ({short_episode_count}/{total})")
    print(f"  {'parallel_call_rate':<40} {parallel_count/total:.4f} ({parallel_count}/{total})")
    print(f"  {'has_final_answer_rate':<40} {has_final_count/total:.4f} ({has_final_count}/{total})")
    print(f"  {'final_answer_structured_rate':<40} {final_structured_count/total:.4f} ({final_structured_count}/{total})")

    print(f"\n📈 Decision Turns 分布:")
    for turns, count in sorted(turn_dist.items())[:15]:
        bar = "█" * min(count * 50 // total, 50)
        print(f"  {turns:3d} turns: {count:5d} ({count/total:.3f}) {bar}")
    if len(turn_dist) > 15:
        remaining = sum(c for t, c in turn_dist.items() if t > 14)
        print(f"  15+ turns: {remaining:5d} ({remaining/total:.3f})")

    print(f"\n📦 Subset Name 分布:")
    for name, count in subset_dist.most_common(10):
        print(f"  {name:<40} {count:5d} ({count/total:.3f})")

    print(f"\n🔧 工具统计:")
    print(f"  唯一工具名数量: {report['unique_tool_names']}")
    print(f"  总 tool_call 数: {sum(r['n_tool_calls'] for r in results)}")
    print(f"  Top 15 工具:")
    for name, count in tool_name_dist.most_common(15):
        print(f"    {name}: {count}")

    print(f"\n🖥️ MCP Server 分布 (从工具名推断):")
    print(f"  唯一 server 数量: {report['unique_mcp_servers']}")
    if server_dist:
        print(f"  Top 15 servers:")
        for server, count in server_dist.most_common(15):
            print(f"    {server}: {count}")
    else:
        print("  (未检测到 MCP server 信息)")

    # 输出决策建议
    print(f"\n{'='*70}")
    print("📋 决策建议:")
    print(f"{'='*70}")

    if report["verifier_ready_rate"] >= 0.5:
        print("  ✅ verifier_ready_rate >= 50%，Toucan 可作为主数据源")
        print("     建议: 直接用 Toucan 构建 EpisodeSeed")
    elif report["verifier_ready_rate"] >= 0.2:
        print("  ⚠️  verifier_ready_rate 20-50%，Toucan 可用但需补充")
        print("     建议: Toucan 提供 MCP schema/tool 分布，ToolACE 补充 verifier-ready 样本")
    else:
        print("  ❌ verifier_ready_rate < 20%，Toucan 仅提供 schema/tool 分布")
        print("     建议: ToolACE 作为主要 verifier-ready 数据源")

    if report["short_episode_rate"] >= 0.6:
        print("  ✅ short_episode_rate >= 60%，适合 max_decision_turns=2-3 的 smoke")
    else:
        print("  ⚠️  short_episode_rate < 60%，需要过滤或截断长轨迹")

    if report["arguments_parseable_rate"] >= 0.8:
        print("  ✅ arguments 解析率 >= 80%，可直接构建 exact verifier")
    else:
        print(f"  ⚠️  arguments 解析率 {report['arguments_parseable_rate']:.1%}，需要改进解析器")

    # 保存报告
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 报告已保存到: {output_path}")

    # 保存 verifier-ready 样本用于人工检查
    sample_path = DATA_DIR / "inspection_samples.jsonl"
    ready_samples = [
        data[i] for i, r in enumerate(results) if r["verifier_ready"]
    ][:20]
    with open(sample_path, "w", encoding="utf-8") as f:
        for s in ready_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"📄 Verifier-ready 样本 (前20条) 保存到: {sample_path}")

    # 保存失败样本用于调试
    fail_path = DATA_DIR / "inspection_failures.jsonl"
    fail_samples = []
    for i, r in enumerate(results):
        if r["parseable"] and not r["verifier_ready"]:
            fail_samples.append({
                "item": data[i],
                "reason": {
                    "has_tool_calls": r["has_tool_calls"],
                    "has_observations": r["has_observations"],
                    "tool_observation_paired": r["tool_observation_paired"],
                    "arguments_parseable": r["arguments_parseable"],
                    "has_target_tools": bool(r["target_tools"]),
                },
            })
            if len(fail_samples) >= 10:
                break
    with open(fail_path, "w", encoding="utf-8") as f:
        for s in fail_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"📄 失败样本 (前10条) 保存到: {fail_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Toucan 数据子集 Inspection")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入 JSONL 文件路径 (默认: 自动查找 data/toucan/ 下最新文件)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告 JSON 路径 (默认: data/toucan/inspection_report.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多分析的样本数 (默认: 全部)",
    )
    args = parser.parse_args()

    # 确定输入文件
    if args.input:
        input_path = Path(args.input)
    else:
        # 自动查找 data/toucan/ 下的 JSONL 文件
        jsonl_files = sorted(DATA_DIR.glob("toucan_*_subset_*.jsonl"))
        if not jsonl_files:
            print(f"错误: 在 {DATA_DIR} 下未找到 Toucan 数据文件")
            print("请先运行: python scripts/download_toucan.py")
            sys.exit(1)
        input_path = jsonl_files[-1]  # 取最新的

    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}")
        sys.exit(1)

    print(f"加载数据: {input_path}")
    data = load_toucan_data(input_path)
    print(f"加载了 {len(data)} 条样本")

    # 限制样本数
    if args.max_samples and args.max_samples < len(data):
        data = data[: args.max_samples]
        print(f"限制为前 {args.max_samples} 条")

    # 先看一下数据结构
    if data:
        print(f"\n📦 数据字段 (第一条样本):")
        first = data[0]
        for key in first.keys():
            val = first[key]
            if isinstance(val, str) and len(val) > 200:
                print(f"  {key}: str len={len(val)} (JSON parseable: {_check_json(val)})")
            elif isinstance(val, str):
                print(f"  {key}: \"{val[:80]}{'...' if len(val) > 80 else ''}\"")
            else:
                print(f"  {key}: {type(val).__name__} = {val}")

    # 生成报告
    output_path = Path(args.output) if args.output else DATA_DIR / "inspection_report.json"
    generate_report(data, output_path)


def _check_json(s: str) -> bool:
    """检查字符串是否是有效 JSON。"""
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


if __name__ == "__main__":
    main()
