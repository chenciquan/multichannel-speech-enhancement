#!/usr/bin/env python3
"""Await the existing serial training queue, then evaluate once, before deadline."""
import json,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g3_overnight_v1'
DEADLINE=datetime.fromisoformat('2026-11-11T21:54:15+00:00').timestamp()
def status(obj):(OUT/'finalize_status.json').write_text(json.dumps(obj,indent=2)+'\n')
def main():
    status({'status':'waiting_for_primary_and_registered_factorial_extension'})
    while not (OUT/'READY_FOR_TEST.json').exists():
        p=OUT/'queue_status.json'
        if p.exists():
            try:r=json.loads(p.read_text())
            except json.JSONDecodeError:r={}
            if r.get('status')=='failed':raise RuntimeError('Training queue failed; preserve logs, do not evaluate partial comparison')
        extension=OUT/'extension_status.json'
        if extension.exists():
            try:r=json.loads(extension.read_text())
            except json.JSONDecodeError:r={}
            if r.get('status')=='failed':raise RuntimeError('Registered factorial extension failed; test remains locked')
        if time.time()>DEADLINE-3000:raise TimeoutError('Evaluation reserve reached without complete comparison; no test opened')
        time.sleep(15)
    for stage in ('signal','asr'):
        status({'status':'evaluating','stage':stage})
        with (OUT/f'evaluate_{stage}.log').open('x') as log:
            p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/g3_evaluate.py'),'--stage',stage],cwd=ROOT,
                             stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()-120))
        if p.returncode:raise RuntimeError(f'Evaluation failed at {stage}; inspect log before retry')
        if stage=='signal':subprocess.run([sys.executable,str(ROOT/'scripts/g3_stage_archive.py'),'--stage','signal'],cwd=ROOT,check=True)
    status({'status':'evaluated','time_utc':datetime.now(timezone.utc).isoformat()})
if __name__=='__main__':
    try:main()
    except Exception as e:status({'status':'failed','error':str(e)});traceback.print_exc();sys.exit(1)
