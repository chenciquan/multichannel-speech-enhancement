#!/usr/bin/env python3
"""Compact read-only status; no Torch import and no experiment mutation."""
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g3_overnight_v1'
answer={}
for name in ('queue','extension','finalize'):
    p=OUT/(name+'_status.json')
    if p.exists():
        try:r=json.loads(p.read_text())
        except json.JSONDecodeError:continue
        answer[name]={k:r[k] for k in ('status','variant','seed','stage','error') if k in r}
        if 'variant' in r:
            path=OUT/'train'/f'{r["variant"]}_s{r["seed"]}'/'status.json'
            if path.exists():
                try:s=json.loads(path.read_text())
                except json.JSONDecodeError:continue
                answer[name].update({k:s[k] for k in ('step','completed_steps','elapsed_seconds') if k in s})
p=OUT/'evaluation/status.json'
if p.exists():
    try:answer['evaluation']=json.loads(p.read_text())
    except json.JSONDecodeError:pass
answer['successful_training_runs']=sum(1 for p in (OUT/'train').glob('*/result.json') if json.loads(p.read_text()).get('status')=='ok')
p=OUT/'evaluation/signals.jsonl'
if p.exists():
    counts=Counter()
    with p.open() as stream:
        for line in stream:
            try:counts[json.loads(line)['status']]+=1
            except json.JSONDecodeError:counts['incomplete_observation_line']+=1
    answer['signal_row_status_counts']=dict(counts)
print(json.dumps(answer))
