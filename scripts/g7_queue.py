#!/usr/bin/env python3
"""Bounded G7 local queue. Never advances without stage reports."""
import json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g7_unseen_noise_v1'
def state(**kw):
    (OUT/'queue_status.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n')
def main():
    OUT.mkdir(parents=True,exist_ok=False);assert json.loads((ROOT/'artifacts/g6_local_runtime_v1/FINAL_COMPLETION_AUDIT.json').read_text())['status']=='ok'
    plan=json.loads((ROOT/'configs/g7_unseen_noise_plan.json').read_text());plan['frozen_utc']=datetime.now(timezone.utc).isoformat();(OUT/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n');state(status='preparing')
    with (OUT/'build.log').open('x') as log:p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/g7_build.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=plan['max_data_stage_seconds'])
    if p.returncode:raise RuntimeError('G7 data stage failed; inspect preserved log before retry')
    state(status='data_archived_waiting_evaluator');limit=time.time()+1800
    while not (OUT/'EVALUATOR_READY.json').exists():
        if time.time()>limit:raise TimeoutError('Evaluator not ready; saved data stage retained')
        time.sleep(5)
    state(status='evaluating')
    with (OUT/'evaluation.log').open('x') as log:p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/g7_evaluate.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=plan['max_evaluation_stage_seconds'])
    if p.returncode:raise RuntimeError('G7 evaluation failed; do not repeat data acquisition or training')
    state(status='complete')
if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():state(status='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('queue ended\n')
