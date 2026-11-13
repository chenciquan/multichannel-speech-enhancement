#!/usr/bin/env python3
"""Serial local job queue, no dependency installation and no final-test access."""
import json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g3_overnight_v1'
PLAN=json.loads((ROOT/'configs/g3_overnight_plan.json').read_text())
CUTOFF=datetime.fromisoformat(PLAN['training_cutoff_utc'].replace('Z','+00:00')).timestamp()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def main():
    # First job is already supervised by the calling terminal. Observe its result;
    # do not rerun it merely because an observation handle timed out.
    first=OUT/'train/ordinary_s20261112';start=time.time()
    while not (first/'result.json').exists():
        if time.time()-start>1800:raise RuntimeError('First run has not completed in 30 minutes; inspect existing process before any retry')
        time.sleep(10)
    jobs=[('complete',20261112),('without_stratification',20261112),('without_clean_identity',20261112),('ordinary',20261113),('complete',20261113)]
    results=[json.loads((first/'result.json').read_text())]
    assert results[0]['status']=='ok'
    for variant,seed in jobs:
        dump(OUT/'queue_status.json',{'status':'running','variant':variant,'seed':seed,'completed':len(results),'time_utc':datetime.now(timezone.utc).isoformat()})
        if time.time()+max(r['elapsed_seconds'] for r in results)+300>CUTOFF:raise RuntimeError('Insufficient cutoff margin; queue stops with completed checkpoints preserved')
        with (OUT/f'train_{variant}_s{seed}.log').open('x') as log:
            p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/g3_train.py'),'--variant',variant,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,CUTOFF-time.time()))
        if p.returncode:raise RuntimeError(f'Training failed: {variant} seed {seed}; inspect its log')
        r=json.loads((OUT/'train'/f'{variant}_s{seed}'/'result.json').read_text());assert r['status']=='ok';results.append(r)
    # The optional third seed is pre-registered; allocate both members together.
    estimated_pair=2*max(r['elapsed_seconds'] for r in results)
    if time.time()+estimated_pair+1800<CUTOFF:
        for variant in ('ordinary','complete'):
            seed=20261114
            dump(OUT/'queue_status.json',{'status':'running_optional_third_pair','variant':variant,'seed':seed,'completed':len(results)})
            with (OUT/f'train_{variant}_s{seed}.log').open('x') as log:
                p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/g3_train.py'),'--variant',variant,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,CUTOFF-time.time()))
            if p.returncode:raise RuntimeError(f'Third-seed run failed: {variant}')
            r=json.loads((OUT/'train'/f'{variant}_s{seed}'/'result.json').read_text());assert r['status']=='ok';results.append(r)
    selected=[{'system':f"{r['variant']}_s{r['seed']}",'variant':r['variant'],'seed':r['seed'],**r['selected'],
               'stoi_gate_passed':r['stoi_gate_passed'],'completed_steps':r['completed_steps']} for r in results]
    manifest={'status':'ok','selected_at_utc':datetime.now(timezone.utc).isoformat(),'final_test_opened':False,
              'selection_rule':'validation SI-SDR with initial STOI minus 0.005 gate; no ASR tuning','systems':selected}
    dump(OUT/'selected_checkpoints.json',manifest);dump(OUT/'TRAIN_COMPLETE.json',{'status':'ok','runs':len(results),'steps':sum(r['completed_steps'] for r in results),'selection_manifest':'selected_checkpoints.json'})
    dump(OUT/'queue_status.json',{'status':'complete','runs':len(results)});print(json.dumps(manifest),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(OUT/'queue_status.json',{'status':'failed','error':str(e)});traceback.print_exc();sys.exit(1)
