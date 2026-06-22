#!/usr/bin/env python3
"""ToolACE 数据格式转换（完整版）。

将 ToolACE 原始格式转换为 SchemaShift 训练格式，提取全部可用数据：

1. decision_steps.jsonl — 每个 assistant tool_call turn 拆为一个 decision step
2. conditioned_steps.jsonl — 真实的 tool_output → next_action 链路
   - conditioned_tool_call: tool_output 后继续调用工具
   - final_answer: tool_output 后给出最终回答
3. no_tool_steps.jsonl — assistant 选择不调用工具的样本
   - ask_clarification: 缺少参数/信息不足
   - no_tool_needed: 不需要工具即可回答，训练 action type 规范化为 final_answer

设计原则：
  - 不丢弃任何有训练价值的数据
  - conditioned_steps 中的 tool_output 是真实的（非合成）
  - no_tool_steps 是 no_extra_call / ask_clarification 的天然正例
"""

import json
import re
import sys
from pathlib import Path
from loguru import logger


def extract_tools_from_system(system_text: str) -> list[dict]:
    """从 system prompt 中提取 JSON 格式的 tool schema 列表。

    ToolACE 的 system prompt 格式：
    "... Here is a list of functions in JSON format that you can invoke:\n[{...}, {...}]"
    """
    match = re.search(r'\[(\s*\{.*)', system_text, re.DOTALL)
    if not match:
        return []

    json_str = match.group(0)
    try:
        tools = json.loads(json_str)
        if isinstance(tools, list):
            return tools
    except json.JSONDecodeError:
        pass

    # 如果直接解析失败，尝试找到匹配的右括号
    bracket_count = 0
    end_idx = -1
    for i, ch in enumerate(json_str):
        if ch == '[':
            bracket_count += 1
        elif ch == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i + 1
                break

    if end_idx > 0:
        try:
            tools = json.loads(json_str[:end_idx])
            if isinstance(tools, list):
                return tools
        except json.JSONDecodeError:
            pass

    return []


def parse_tool_call(call_str: str) -> list[dict]:
    """解析 ToolACE 的 tool call 格式。

    格式: [FuncName(param1="value1", param2=123)]
    可能有多个并行调用: [Func1(a=1), Func2(b=2)]

    Returns:
        [{"name": "FuncName", "arguments": {"param1": "value1", ...}}]
    """
    calls = []
    call_str = call_str.strip()
    if call_str.startswith('[') and call_str.endswith(']'):
        call_str = call_str[1:-1].strip()

    func_calls = _split_top_level_calls(call_str)

    for fc in func_calls:
        fc = fc.strip()
        if not fc:
            continue
        m = re.match(r'([A-Za-z_][\w\s]*?)\((.*)\)$', fc, re.DOTALL)
        if not m:
            continue
        func_name = m.group(1).strip()
        args_str = m.group(2).strip()

        arguments = _parse_arguments(args_str)
        calls.append({"name": func_name, "arguments": arguments})

    return calls


def _split_top_level_calls(s: str) -> list[str]:
    """按顶层逗号分割多个函数调用（不分割括号内的逗号）。"""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch in '([{':
            depth += 1
            current.append(ch)
        elif ch in ')]}':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


def _parse_arguments(args_str: str) -> dict:
    """解析函数参数字符串为字典。"""
    if not args_str.strip():
        return {}

    arguments = {}
    params = _split_top_level_calls(args_str)

    for param in params:
        param = param.strip()
        if not param or '=' not in param:
            continue
        eq_idx = param.index('=')
        key = param[:eq_idx].strip()
        value_str = param[eq_idx + 1:].strip()

        value = _parse_value(value_str)
        if key:
            arguments[key] = value

    return arguments


def _parse_value(value_str: str):
    """解析参数值。"""
    if not value_str:
        return ""

    try:
        return json.loads(value_str)
    except (json.JSONDecodeError, ValueError):
        pass

    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]

    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    if value_str.lower() == 'none' or value_str.lower() == 'null':
        return None

    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass

    return value_str


def normalize_tool_schema(tool: dict) -> dict:
    """将 ToolACE 的 tool schema 标准化为 OpenAI function calling 格式。"""
    normalized = dict(tool)
    if "parameters" in normalized:
        params = normalized["parameters"]
        if isinstance(params, dict):
            if params.get("type") == "dict":
                params["type"] = "object"
            if "properties" in params:
                for prop_name, prop_spec in params["properties"].items():
                    if isinstance(prop_spec, dict):
                        ptype = prop_spec.get("type", "")
                        if isinstance(ptype, list):
                            ptype = ptype[0] if ptype else ""
                        if not isinstance(ptype, str):
                            ptype = str(ptype) if ptype else ""
                        type_map = {
                            "str": "string",
                            "int": "integer",
                            "float": "number",
                            "bool": "boolean",
                            "list": "array",
                            "dict": "object",
                        }
                        if ptype in type_map:
                            prop_spec["type"] = type_map[ptype]
    return normalized


def _is_tool_call(value: str) -> bool:
    """判断 assistant 回复是否包含 tool call。"""
    return '[' in value and '(' in value


def _classify_no_tool_response(text: str) -> str:
    """分类不调用工具的 assistant 回复类型。

    Returns:
        "ask_clarification" | "no_tool_needed"
    """
    text_lower = text.lower()
    clarification_signals = [
        "please provide", "could you provide", "i need",
        "lacks the", "missing", "required",
        "can you specify", "what is the", "which",
        "more information", "more details",
        "need to know", "not specified",
    ]
    if any(signal in text_lower for signal in clarification_signals):
        return "ask_clarification"
    return "no_tool_needed"


def convert_sample_full(sample: dict, sample_idx: int) -> dict:
    """将一条 ToolACE 样本完整转换，提取所有可用数据。

    Returns:
        {
            "decision_steps": [...],      # tool_call decision steps
            "conditioned_steps": [...],   # tool_output → next_action 链路
            "no_tool_steps": [...],       # 不调用工具的样本
        }
    """
    system_text = sample.get("system", "")
    conversations = sample.get("conversations", [])

    tools = extract_tools_from_system(system_text)
    if not tools:
        return {"decision_steps": [], "conditioned_steps": [], "no_tool_steps": []}

    tools = [normalize_tool_schema(t) for t in tools]

    decision_steps = []
    conditioned_steps = []
    no_tool_steps = []

    tool_call_step_idx = 0

    for conv_idx, conv in enumerate(conversations):
        if conv["from"] != "assistant":
            continue

        value = conv["value"]

        # 构建到当前 turn 之前的对话历史
        context_messages = []
        for prev in conversations[:conv_idx]:
            role_map = {"user": "user", "assistant": "assistant", "tool": "tool"}
            role = role_map.get(prev["from"], prev["from"])
            context_messages.append({"role": role, "content": prev["value"]})

        # 先尝试解析 tool call，只有解析成功才视为 tool_call turn
        tool_calls = parse_tool_call(value) if _is_tool_call(value) else []

        if tool_calls:
            # === 这是一个 tool_call turn ===
            task_id = f"toolace_{sample_idx:05d}_step_{tool_call_step_idx}"

            # 判断是否是 conditioned decision（前面有 tool_output）
            has_prior_tool_output = any(
                m.get("role") == "tool" for m in context_messages
            )

            decision_steps.append({
                "task_id": task_id,
                "source": "toolace",
                "tools": tools,
                "messages": context_messages,
                "ground_truth_calls": tool_calls,
                "num_tools": len(tools),
                "num_prior_turns": len(context_messages),
                "step_index": tool_call_step_idx,
                "total_steps_in_sample": -1,  # 后面回填
            })

            if has_prior_tool_output:
                # 这是一个真实的 conditioned_tool_call
                conditioned_steps.append({
                    "task_id": f"{task_id}___conditioned",
                    "source": "toolace",
                    "tools": tools,
                    "messages": context_messages,
                    "ground_truth_action": {
                        "type": "tool_call",
                        "tool_calls": tool_calls,
                    },
                    "action_type": "conditioned_tool_call",
                    "provenance": "real",
                    "scenario_type": "conditioned_tool_call",
                    "step_index": tool_call_step_idx,
                })

            tool_call_step_idx += 1

        else:
            # === 这是一个纯文本 assistant turn（不调用工具）===
            has_prior_tool_output = any(
                m.get("role") == "tool" for m in context_messages
            )

            if has_prior_tool_output:
                # tool_output 后的 final_answer（真实 conditioned decision）
                conditioned_steps.append({
                    "task_id": f"toolace_{sample_idx:05d}_final_{conv_idx}",
                    "source": "toolace",
                    "tools": tools,
                    "messages": context_messages,
                    "ground_truth_action": {
                        "type": "final_answer",
                        "content": value,
                    },
                    "action_type": "final_answer",
                    "provenance": "real",
                    "scenario_type": "final_answer",
                    "step_index": tool_call_step_idx,
                })
            else:
                # 没有 tool_output 上下文，assistant 选择不调用工具
                no_tool_subtype = _classify_no_tool_response(value)
                oracle_action_type = (
                    "ask_clarification"
                    if no_tool_subtype == "ask_clarification"
                    else "final_answer"
                )
                no_tool_steps.append({
                    "task_id": f"toolace_{sample_idx:05d}_notool_{conv_idx}",
                    "source": "toolace",
                    "tools": tools,
                    "messages": context_messages,
                    "ground_truth_action": {
                        "type": oracle_action_type,
                        "content": value,
                        "no_tool_subtype": no_tool_subtype,
                    },
                    "action_type": oracle_action_type,
                    "no_tool_subtype": no_tool_subtype,
                    "provenance": "real",
                    "scenario_type": "no_tool",
                    "step_index": 0,
                })

    # 回填 total_steps_in_sample
    for ds in decision_steps:
        ds["total_steps_in_sample"] = len(decision_steps)

    return {
        "decision_steps": decision_steps,
        "conditioned_steps": conditioned_steps,
        "no_tool_steps": no_tool_steps,
    }


def main():
    input_path = Path(__file__).resolve().parent.parent / "data" / "toolace" / "raw" / "toolace_train.jsonl"
    output_dir = Path(__file__).resolve().parent.parent / "data" / "toolace" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_decision = output_dir / "decision_steps.jsonl"
    output_conditioned = output_dir / "conditioned_steps.jsonl"
    output_no_tool = output_dir / "no_tool_steps.jsonl"

    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        logger.error("请先运行: python scripts/download_toolace.py")
        sys.exit(1)

    logger.info(f"读取: {input_path}")

    total_samples = 0
    skipped = 0
    total_decision_steps = 0
    total_conditioned_steps = 0
    total_no_tool_steps = 0
    multi_step_samples = 0

    # 细分统计
    conditioned_tool_call_count = 0
    conditioned_final_answer_count = 0
    no_tool_ask_clarification_count = 0
    no_tool_no_needed_count = 0

    with open(input_path) as fin, \
         open(output_decision, "w", encoding="utf-8") as f_decision, \
         open(output_conditioned, "w", encoding="utf-8") as f_conditioned, \
         open(output_no_tool, "w", encoding="utf-8") as f_no_tool:

        for i, line in enumerate(fin):
            sample = json.loads(line)
            total_samples += 1

            result = convert_sample_full(sample, i)

            if not result["decision_steps"] and not result["conditioned_steps"] and not result["no_tool_steps"]:
                skipped += 1
                continue

            # 写 decision_steps
            if len(result["decision_steps"]) > 1:
                multi_step_samples += 1
            for step in result["decision_steps"]:
                f_decision.write(json.dumps(step, ensure_ascii=False) + "\n")
                total_decision_steps += 1

            # 写 conditioned_steps
            for step in result["conditioned_steps"]:
                f_conditioned.write(json.dumps(step, ensure_ascii=False) + "\n")
                total_conditioned_steps += 1
                if step["action_type"] == "conditioned_tool_call":
                    conditioned_tool_call_count += 1
                elif step["action_type"] == "final_answer":
                    conditioned_final_answer_count += 1

            # 写 no_tool_steps
            for step in result["no_tool_steps"]:
                f_no_tool.write(json.dumps(step, ensure_ascii=False) + "\n")
                total_no_tool_steps += 1
                no_tool_subtype = step.get("no_tool_subtype", step["action_type"])
                if no_tool_subtype == "ask_clarification":
                    no_tool_ask_clarification_count += 1
                else:
                    no_tool_no_needed_count += 1

            if (i + 1) % 2000 == 0:
                logger.info(
                    f"  处理 {i+1}/{11300}... "
                    f"decision={total_decision_steps}, "
                    f"conditioned={total_conditioned_steps}, "
                    f"no_tool={total_no_tool_steps}"
                )

    logger.info("=" * 60)
    logger.info("转换完成:")
    logger.info(f"  输入样本: {total_samples}")
    logger.info(f"  跳过（无法提取 tool schema）: {skipped}")
    logger.info(f"  有效样本: {total_samples - skipped}")
    logger.info(f"  多步样本（>1 tool call）: {multi_step_samples}")
    logger.info("=" * 60)
    logger.info(f"  decision_steps: {total_decision_steps} 条 → {output_decision}")
    logger.info(f"  conditioned_steps: {total_conditioned_steps} 条 → {output_conditioned}")
    logger.info(f"    ├ conditioned_tool_call: {conditioned_tool_call_count}")
    logger.info(f"    └ final_answer: {conditioned_final_answer_count}")
    logger.info(f"  no_tool_steps: {total_no_tool_steps} 条 → {output_no_tool}")
    logger.info(f"    ├ ask_clarification: {no_tool_ask_clarification_count}")
    logger.info(f"    └ no_tool_needed: {no_tool_no_needed_count}")
    logger.info("=" * 60)

    # 保存统计信息
    stats = {
        "total_samples": total_samples,
        "skipped": skipped,
        "valid_samples": total_samples - skipped,
        "multi_step_samples": multi_step_samples,
        "total_decision_steps": total_decision_steps,
        "avg_steps_per_sample": total_decision_steps / max(1, total_samples - skipped),
        "total_conditioned_steps": total_conditioned_steps,
        "conditioned_tool_call": conditioned_tool_call_count,
        "conditioned_final_answer": conditioned_final_answer_count,
        "total_no_tool_steps": total_no_tool_steps,
        "no_tool_ask_clarification": no_tool_ask_clarification_count,
        "no_tool_no_needed": no_tool_no_needed_count,
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  统计: {stats_path}")


if __name__ == "__main__":
    main()
