#!/usr/bin/env python3
"""Compact read-only stage health; no Torch import or experiment mutation."""
import argparse,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();folder=(ROOT/'artifacts'/a.run).resolve();folder.relative_to((ROOT/'artifacts').resolve());out={}
for name in ('queue_status.json','status.json','power_helper.json'):
    path=folder/name
    if path.exists():
        try:out[name]=json.loads(path.read_text())
        except json.JSONDecodeError:out[name]={'observation':'partial JSON write; not a terminal job state'}
path=folder/'evaluation/signals.jsonl'
if path.exists():
    c=Counter()
    for line in path.open():
        try:c[json.loads(line)['status']]+=1
        except json.JSONDecodeError:pass
    out['signal_row_status_counts']=dict(c)
print(json.dumps(out))
