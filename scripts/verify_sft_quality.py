"""验证 SFT 数据质量：检查 prior/target 不可见工具调用。"""
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

prior_unavailable_steps = 0
prior_unavailable_calls = 0
target_unavailable_steps = 0
total_samples = 0

with open(os.path.join(PROJECT_ROOT, "data", "sft", "sft_train.jsonl")) as f:
    for line in f:
        total_samples += 1
        sample = json.loads(line)
        msgs = sample["messages"]
        system_content = msgs[0].get("content", "") if msgs and msgs[0].get("role") == "system" else ""

        # 检查 completion
        completion = msgs[-1].get("content", "") if msgs[-1].get("role") == "assistant" else ""
        match = re.search(r"<tool_call>(.*?)</tool_call>", completion, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                tool_names = []
                if isinstance(parsed, dict):
                    tool_names = [parsed.get("name", "")]
                elif isinstance(parsed, list):
                    tool_names = [c.get("name", "") for c in parsed if isinstance(c, dict)]
                for tn in tool_names:
                    if tn and tn not in system_content:
                        target_unavailable_steps += 1
                        break
            except Exception:
                pass

        # 检查 prior history
        for msg in msgs[1:-1]:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            matches_found = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
            for m in matches_found:
                try:
                    parsed = json.loads(m)
                    tool_names = []
                    if isinstance(parsed, dict):
                        tool_names = [parsed.get("name", "")]
                    elif isinstance(parsed, list):
                        tool_names = [c.get("name", "") for c in parsed if isinstance(c, dict)]
                    for tn in tool_names:
                        if tn and tn not in system_content:
                            prior_unavailable_calls += 1
                    if any(tn and tn not in system_content for tn in tool_names):
                        prior_unavailable_steps += 1
                except Exception:
                    pass

print(f"Total samples: {total_samples}")
print(f"Target unavailable (completion): {target_unavailable_steps}")
print(f"Prior unavailable calls (history): {prior_unavailable_calls}")
print(f"Prior unavailable steps (history): {prior_unavailable_steps}")
