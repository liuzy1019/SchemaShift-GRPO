#!/usr/bin/env python3
"""Full pipeline adversarial review — runs after data generation completes."""
import sys, json
sys.path.insert(0, '.')
import pandas as pd
from collections import Counter, defaultdict

train = pd.read_parquet('data/train.parquet', columns=['uid','prompt','scenario_type','perturbation_level','extra_info','reward_model'])
val = pd.read_parquet('data/val.parquet', columns=['uid','prompt','scenario_type','perturbation_level','extra_info','reward_model'])
all_data = pd.concat([train, val])
print(f'TOTAL: {len(all_data)} (train={len(train)}, val={len(val)})')

from src.reward.oval_reward_fn import _build_task_dict

BAD = Counter()
for idx, row in all_data.iterrows():
    ei = dict(row['extra_info'])
    uid = row['uid']
    hmf = bool(ei.get('has_missing_function'))
    st = str(ei.get('scenario_type', ''))
    abst = hmf or st in ('missing_function', 'irrelevant')

    # Parse oracle
    oc = ei.get('oracle_calls', '')
    if isinstance(oc, str):
        try: oc = json.loads(oc)
        except: oc = []
    elif not isinstance(oc, list): oc = []
    real = [c for c in oc if isinstance(c, dict) and c.get('action') != 'clarification']
    is_clar_last = bool(oc and isinstance(oc[-1], dict) and oc[-1].get('action') == 'clarification')

    # Build task_dict
    td = _build_task_dict(ei)
    rtc = td['required_tool_calls']
    terminal = td['allowed_terminal_actions']

    # Core contracts
    if abst and real: BAD['abs_has_real'] += 1
    if abst and rtc: BAD['abs_has_rtc'] += 1
    if abst and terminal != ['report_error']: BAD['abs_wrong_terminal'] += 1
    if not abst and real and not is_clar_last and terminal != ['final_answer']:
        BAD['normal_wrong_terminal'] += 1
    if not abst and is_clar_last and terminal != ['ask_clarification']:
        BAD['clar_wrong_terminal'] += 1
    if not abst and is_clar_last and not real and rtc:
        BAD['clar_only_has_rtc'] += 1
    if not abst and real:
        oracle_ts = set(c.get('tool_name','') for c in real)
        rtc_ts = set(c['tool_name'] for c in rtc if c.get('tool_name'))
        if oracle_ts != rtc_ts: BAD['tool_set_mismatch'] += 1
        assertions = td['outcome_assertions']
        assert_ts = set(a['tool_name'] for a in assertions if a.get('tool_name'))
        if assert_ts != oracle_ts: BAD['assertion_tool_mismatch'] += 1

    # Success criteria
    sc = ei.get('success_criteria', '')
    if isinstance(sc, str):
        try: sc = json.loads(sc)
        except: sc = []
    for c in sc:
        if isinstance(c, dict):
            t = c.get('type', '')
            if t == 'state_exists' and not c.get('path', ''): BAD['sc_empty_path'] += 1
            if t == 'state_equals' and c.get('value') is None: BAD['sc_none_val'] += 1

    # Hidden tool leak (exact tool name match in system prompt)
    ht = ei.get('hidden_tools', [])
    p = json.loads(row['prompt']) if isinstance(row['prompt'], str) else row['prompt']
    sys_txt = p[0].get('content', '') if isinstance(p, list) and p else ''
    import re
    tool_names_in_prompt = set(re.findall(r'^- (\w+):', sys_txt, re.MULTILINE))
    for h in ht:
        if h in tool_names_in_prompt: BAD['hidden_leak'] += 1

    # Visible tools in prompt
    vt = ei.get('visible_tool_names', [])
    for t in vt:
        if t not in tool_names_in_prompt: BAD['visible_missing'] += 1

    # Budget
    if len(real) > ei.get('budget', 8): BAD['budget_exceed'] += 1
    # perturbation_level
    pl = ei.get('perturbation_level', '')
    if pl not in ('complete', 'missing', 'minimal'): BAD['bad_perturb'] += 1

# UID checks
tu, vu = set(train['uid']), set(val['uid'])
if tu & vu: BAD['uid_overlap'] += len(tu & vu)
if len(tu) != len(train): BAD['train_dup_uid'] += len(train) - len(tu)
if len(vu) != len(val): BAD['val_dup_uid'] += len(val) - len(vu)

# Dedup check
def _oracle_sig(row):
    ei = row['extra_info']; domain = ei.get('domain', '')
    oc_r = ei.get('oracle_calls', [])
    if isinstance(oc_r, str): oc_r = json.loads(oc_r)
    tools = tuple(sorted(c.get('tool_name','') for c in oc_r if isinstance(c,dict) and c.get('action')!='clarification'))
    return (domain, tools)
train_sigs = set(_oracle_sig(row) for _, row in train.iterrows())
val_cross = sum(1 for _, row in val.iterrows() if _oracle_sig(row) in train_sigs)
if val_cross: BAD['semantic_overlap'] = val_cross

print()
print('='*55)
print('FINAL ADVERSARIAL REVIEW')
print('='*55)
all_ok = True
for k in sorted(BAD):
    v = BAD[k]
    print(f'  {"✅" if v == 0 else "❌"} {k}: {v}')
    if v > 0: all_ok = False

print(f'\nScenario: {dict(all_data.scenario_type.value_counts())}')
print(f'Difficulty: {dict(all_data.perturbation_level.value_counts())}')
print(f'\n{"="*55}')
if all_ok:
    print('RESULT: ALL CHECKS PASSED')
else:
    print(f'RESULT: {sum(1 for v in BAD.values() if v > 0)} CATEGORIES HAVE ERRORS')
print('='*55)
