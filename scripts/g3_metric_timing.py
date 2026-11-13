#!/usr/bin/env python3
"""CPU backend timing on an OLD G2 waveform, never method/test tuning."""
import json,sys,time
from pathlib import Path
import soundfile as sf
from threadpoolctl import threadpool_limits,threadpool_info
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.evaluation import MetricEvaluator
r=json.loads((ROOT/'artifacts/g2_confirmation_v1/mcnet/cases.jsonl').read_text().splitlines()[0])
s,_=sf.read(r['reference_path'],dtype='float32');y,_=sf.read(r['enhanced_path'],dtype='float32');e=MetricEvaluator();records=[]
for limit in (None,1):
    with threadpool_limits(limits=limit,user_api='blas'):
        times=[];scores=None
        for _ in range(3):
            start=time.perf_counter();scores=e.evaluate(s,y);times.append(time.perf_counter()-start)
        records.append({'blas_limit':limit,'times_seconds':times,'scores':scores,'threadpools':threadpool_info()})
out=ROOT/'artifacts/g3_overnight_v1/metric_cpu_profile.json'
if out.exists():raise RuntimeError('Timing report exists')
out.write_text(json.dumps(records,indent=2)+'\n');print(json.dumps([{k:r[k] for k in ('blas_limit','times_seconds','scores')} for r in records]))
