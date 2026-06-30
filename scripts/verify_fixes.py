"""Verify fixes by running _build_task_dict on the existing parquet data."""
import sys, json
sys.path.insert(0, '.')

import pandas as pd
from src.reward.oval_reward_fn import _build_task_dict

train = pd.read_parquet('data/train.parquet')
val = pd.read_parquet('data/val.parquet')
all_data = pd.concat([train, val])

print(f"Total samples: {len(all_data)}\n")

errors = {"p4a": 0, "p4b": 0, "p3a": 0}
clar_lock_errors = 0
correct_p4a = 0
correct_p4b = 0

for idx, row in all_data.iterrows():
    ei = dict(row['extra_info'])
    task_dict = _build_task_dict(ei)

    scenario = ei.get('scenario_type', '')
    has_mf = ei.get('has_missing_function', False)
    is_abstain = has_mf or scenario in ('missing_function', 'irrelevant')

    # P4a check: abstain tasks must have empty required_tool_calls
    rtc = task_dict.get('required_tool_calls', [])
    if is_abstain:
        if rtc:
            errors["p4a"] += 1
        else:
            correct_p4a += 1

    # P4b check: clarification oracle that is NOT last should NOT lock terminal
    oc_raw = ei.get('oracle_calls', [])
    if isinstance(oc_raw, str):
        oc_raw = json.loads(oc_raw)
    if isinstance(oc_raw, list) and oc_raw:
        has_any_clar = any(c.get('action') == 'clarification' for c in oc_raw if isinstance(c, dict))
        last_is_clar = oc_raw[-1].get('action') == 'clarification' if isinstance(oc_raw[-1], dict) else False
        allowed = task_dict.get('allowed_terminal_actions', [])

        if has_any_clar and not last_is_clar:
            if 'ask_clarification' not in allowed:
                correct_p4b += 1
            else:
                clar_lock_errors += 1

    # P3a check: for non-abstain tasks with oracle, required_tool_calls tools
    # should match oracle tool names (not required_tools field)
    if not is_abstain and rtc:
        oc_tool_names = set()
        for c in oc_raw if isinstance(oc_raw, list) else []:
            if isinstance(c, dict) and c.get('action') != 'clarification':
                oc_tool_names.add(c.get('tool_name', ''))
        rtc_names = set(c.get('tool_name', '') for c in rtc)
        if oc_tool_names and rtc_names != oc_tool_names:
            errors["p3a"] += 1

print(f"P4a (missing_function required_tool_calls):")
print(f"  Correct (empty): {correct_p4a}")
print(f"  Errors (non-empty): {errors['p4a']}")
print(f"\nP4b (clarification not locking terminal when not last):")
print(f"  Correct (final_answer): {correct_p4b}")
print(f"  Errors (locked to ask_clarification): {clar_lock_errors}")
print(f"\nP3a (required_tool_calls matching oracle):")
print(f"  Errors (mismatch still): {errors['p3a']}")

if any(errors.values()) or clar_lock_errors:
    print("\n*** SOME ERRORS REMAIN ***")
else:
    print("\n*** ALL FIXES VERIFIED PASSED ***")
