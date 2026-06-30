"""PROVE adversarial audit — smoke30c"""
import pandas as pd, json
from collections import Counter

t = pd.read_parquet('data/train.parquet')
v = pd.read_parquet('data/val.parquet')
combined = pd.concat([t, v])
total = len(combined)
print(f'Total rows: {total}')
print()

# D1
levels = Counter()
scenarios = Counter()
domains = Counter()
for i in range(total):
    ei = combined['extra_info'].iloc[i]
    if isinstance(ei, str): ei = eval(ei)
    levels[ei.get('perturbation_level', '?')] += 1
    scenarios[ei.get('scenario_type', '?')] += 1
    domains[ei.get('domain', '?')] += 1
print('D1: INFORMATION-LEVEL STRATIFICATION (expect 60/20/20)')
for k in ['complete','missing','minimal']:
    print(f'  {k}: {levels.get(k,0)}/{total} = {levels.get(k,0)/total*100:.1f}%')
print(f'  Scenarios: {dict(scenarios)}')
print(f'  Domains: {len(domains)} total, {dict(sorted(domains.items()))}')
print()

# D2
chain_lens = []
for i in range(total):
    ei = combined['extra_info'].iloc[i]
    if isinstance(ei, str): ei = eval(ei)
    oc_raw = ei.get('oracle_calls','[]')
    oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
    real = [c for c in oc if c.get('action','tool_call')!='clarification']
    chain_lens.append(len(real))
cl = Counter(chain_lens)
empty_oc = cl.get(0,0)
print('D2: ORACLE CHAIN LENGTH (PROVE: 1-5)')
print(f'  Distribution: {dict(sorted(cl.items()))}')
print(f'  Empty oracle: {empty_oc}/{total}')
print(f'  >5 violations: {sum(1 for l in chain_lens if l>5)}/{total}')
print()

# D3
conv_rounds = Counter()
for i in range(total):
    ei = combined['extra_info'].iloc[i]
    if isinstance(ei, str): ei = eval(ei)
    conv_rounds[int(ei.get('conversation_rounds',1))] += 1
print('D3: CONVERSATION STRUCTURE (PROVE: 2-3 rounds)')
print(f'  {dict(sorted(conv_rounds.items()))}')
print()

# D4
d_n = i_n = mf_n = n_n = 0
for i in range(total):
    ei = combined['extra_info'].iloc[i]
    if isinstance(ei, str): ei = eval(ei)
    s = ei.get('scenario_type','?')
    if s=='distractor': d_n += 1
    elif s=='irrelevant': i_n += 1
    elif s=='missing_function': mf_n += 1
    else: n_n += 1
print('D4: ROBUSTNESS KNOBS')
print(f'  Normal: {n_n} ({n_n/total*100:.1f}%)')
print(f'  Distractor: {d_n} ({d_n/total*100:.1f}%)')
print(f'  Missing function: {mf_n} ({mf_n/total*100:.1f}%)')
print(f'  Irrelevant: {i_n} ({i_n/total*100:.1f}%)')
print()

# D5
import re
id_pats = {'evt':r'evt_\d+','acc':r'acc_\w+','eml':r'eml_\d+','prd':r'prd_\d+','inv':r'inv_\d+','iss':r'iss_\d+'}
id_hits = Counter(); id_rows = 0
for i in range(total):
    p = combined['prompt'].iloc[i]
    msgs = json.loads(p) if isinstance(p, str) else list(p)
    user_text = ' '.join(m.get('content','') for m in msgs if m.get('role')=='user')
    found = False
    for t, pat in id_pats.items():
        if re.findall(pat, user_text): id_hits[t] += 1; found = True
    if found: id_rows += 1
print('D5: PROMPT GROUNDING (real entity IDs)')
print(f'  Rows with entity IDs: {id_rows}/{total}')
print(f'  Types: {dict(id_hits)}')
print()

# D6
orphan = xml0 = gt_err = 0
for i in range(total):
    ei = combined['extra_info'].iloc[i]
    if isinstance(ei, str): ei = eval(ei)
    oc_raw = ei.get('oracle_calls','[]')
    oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
    real = [c for c in oc if c.get('action','tool_call')!='clarification']
    p = combined['prompt'].iloc[i]; msgs = json.loads(p) if isinstance(p, str) else list(p)
    xml = sum((m.get('content','') or '').count('<tool_call>') for m in msgs if m.get('role')=='assistant')
    tool = sum(1 for m in msgs if m.get('role')=='tool')
    if tool > xml: orphan += 1
    if xml==0 and len(real)>0: xml0 += 1
    rm = combined['reward_model'].iloc[i]
    if isinstance(rm, str): rm = eval(rm)
    if rm.get('ground_truth',{}).get('oracle_calls','') != ei.get('oracle_calls',''): gt_err += 1
print('D6: PROMPT↔ORACLE CONSISTENCY')
print(f'  Orphan tool results: {orphan}/{total}')
print(f'  XML=0 but oracle>0: {xml0}/{total}')
print(f'  GT vs EI mismatch: {gt_err}/{total}')
print()

# D7 — check if hidden tool exists as a function *definition* in the prompt
import re as _re
mf = mf_oc = mf_hidden = 0
for i in range(total):
    ei = combined['extra_info'].iloc[i]
    if isinstance(ei, str): ei = eval(ei)
    if ei.get('scenario_type')!='missing_function':
        continue
    mf += 1
    oc_raw = ei.get('oracle_calls','[]')
    oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
    if [c for c in oc if c.get('action','tool_call')!='clarification']: mf_oc += 1
    hidden = list(ei.get('hidden_tools', []))
    if not hidden:
        continue
    p = combined['prompt'].iloc[i]; msgs = json.loads(p) if isinstance(p, str) else list(p)
    # Extract all tool names from system prompt — look for "name" fields in tool definitions
    tool_names_in_prompt = set()
    for m in msgs:
        if m.get('role') == 'system':
            content = m.get('content', '') or ''
            # Match JSON tool definitions: "name": "tool_name" or 'name': 'tool_name'
            names = _re.findall(r'''["']name["']\s*:\s*["']([^"']+)["']''', content)
            tool_names_in_prompt.update(names)
    for h in hidden:
        if h in tool_names_in_prompt:
            mf_hidden += 1
print('D7: MISSING_FUNCTION ABSTAIN')
print(f'  MF rows: {mf}')
print(f'  MF with oracle: {mf_oc} (expect 0)')
print(f'  MF hidden in prompt: {mf_hidden} (expect 0)')
print()

# D8
print('D8: REWARD MODEL STRUCTURE (§3.3)')
rm0 = combined['reward_model'].iloc[0]
if isinstance(rm0, str): rm0 = eval(rm0)
print(f'  reward_model keys: {list(rm0.keys())}')
gt_keys = rm0.get("ground_truth", {})
print(f'  ground_truth keys: {list(gt_keys.keys())}')
# required_tools match
req_err = 0
for i in range(total):
    rm = combined['reward_model'].iloc[i]
    if isinstance(rm, str): rm = eval(rm)
    req = set(rm.get('ground_truth',{}).get('required_tools',[]))
    ei = combined['extra_info'].iloc[i]
    if isinstance(ei, str): ei = eval(ei)
    oc_raw = ei.get('oracle_calls','[]')
    oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
    oc_tools = set(c.get('tool_name') for c in oc if c.get('action','tool_call')!='clarification')
    if req != oc_tools: req_err += 1
print(f'  required_tools ≠ oracle unique tools: {req_err}/{total}')
print()

# SUMMARY
p1 = sum(1 for l in chain_lens if l>5)==0
p2 = orphan==0
p3 = xml0==0
p4 = gt_err==0
p5 = mf_oc==0
p6 = mf_hidden==0
s1 = "PASS" if p1 else "FAIL"
s2 = "PASS" if p2 else "FAIL"
s3 = "PASS" if p3 else "FAIL"
s4 = "PASS" if p4 else "FAIL"
s5 = "PASS" if p5 else "FAIL"
s6 = "PASS" if p6 else "FAIL"
all_ok = all([p1,p2,p3,p4,p5,p6])
print('='*60)
print(f'  Chain≤5:              {s1}')
print(f'  No orphan results:     {s2}')
print(f'  XML↔Oracle:            {s3}')
print(f'  GT↔EI:                 {s4}')
print(f'  MF abstain:            {s5}')
print(f'  MF no hidden in prompt:{s6}')
final = "ALL PASS" if all_ok else "FAILURES"
print(f'  => {final}')
print('='*60)
