#!/usr/bin/env python3
"""Compact read-only monitor for the current G5 stage."""
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g5_rotation_v1';result={}
for key,name in [('queue','queue_status.json'),('evaluation','status.json')]:
    p=OUT/name
    if p.exists():
        try:r=json.loads(p.read_text())
        except json.JSONDecodeError:continue
        result[key]=r
        if r.get('status')=='training':
            s=OUT/r['geometry']/'train'/f'{r["variant"]}_s{r["seed"]}'/'status.json'
            if s.exists():
                try:t=json.loads(s.read_text())
                except json.JSONDecodeError:continue
                result['current_run']={k:t[k] for k in ('status','step','completed_steps','elapsed_seconds') if k in t}
result['completed_runs']=len(list(OUT.glob('*/train/*/result.json')))
p=OUT/'evaluation/signals.jsonl'
if p.exists():
    c=Counter()
    for line in p.open():
        try:c[json.loads(line)['status']]+=1
        except json.JSONDecodeError:pass
    result['signal_status_counts']=dict(c)
print(json.dumps(result))
