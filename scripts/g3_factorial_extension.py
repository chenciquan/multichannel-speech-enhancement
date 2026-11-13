#!/usr/bin/env python3
"""Transparent validation-triggered ablation replication, strictly before test."""
import hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g3_overnight_v1'
CUTOFF=datetime.fromisoformat('2026-11-11T21:19:35+00:00').timestamp()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def main():
    dump(OUT/'extension_status.json',{'status':'waiting_for_existing_primary_queue'})
    while not (OUT/'TRAIN_COMPLETE.json').exists():
        p=OUT/'queue_status.json'
        if p.exists():
            try:r=json.loads(p.read_text())
            except json.JSONDecodeError:r={}
            if r.get('status')=='failed':raise RuntimeError('Primary queue failed; no test access')
        if time.time()>CUTOFF-1800:raise TimeoutError('Insufficient extension margin')
        time.sleep(15)
    assert not (OUT/'evaluation').exists(),'No extra training after test access'
    original=OUT/'selected_checkpoints.json';primary=json.loads(original.read_text());selected=list(primary['systems'])
    assert any(s['seed']==20261114 for s in selected),'Third paired seed must finish for factorial extension'
    jobs=[('without_stratification',20261113),('without_clean_identity',20261113),('without_stratification',20261114),('without_clean_identity',20261114)]
    for variant,seed in jobs:
        dump(OUT/'extension_status.json',{'status':'training','variant':variant,'seed':seed,'completed_extra_runs':len(selected)-len(primary['systems'])})
        if time.time()>CUTOFF-600:raise TimeoutError('Extension training cutoff margin')
        with (OUT/f'train_{variant}_s{seed}.log').open('x') as log:
            p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/g3_train.py'),'--variant',variant,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,CUTOFF-time.time()))
        if p.returncode:raise RuntimeError(f'Extension run failed {variant} {seed}')
        r=json.loads((OUT/'train'/f'{variant}_s{seed}'/'result.json').read_text());assert r['status']=='ok'
        selected.append({'system':f'{variant}_s{seed}','variant':variant,'seed':seed,**r['selected'],'stoi_gate_passed':r['stoi_gate_passed'],'completed_steps':r['completed_steps'],'classification':'validation-triggered exploratory replication'})
    assert not (OUT/'evaluation').exists()
    final={'status':'ok','selected_at_utc':datetime.now(timezone.utc).isoformat(),'final_test_opened':False,
           'primary_selection_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
           'extension_plan_sha256':hashlib.sha256((ROOT/'configs/g3_factorial_extension.json').read_bytes()).hexdigest(),
           'classification':'Original primary plus explicitly registered exploratory factorial replication','systems':selected}
    dump(OUT/'selected_checkpoints_extended.json',final);dump(OUT/'EXTENSION_COMPLETE.json',{'extra_runs':4,'total_runs':len(selected),'steps':len(selected)*3000})
    subprocess.run([sys.executable,str(ROOT/'scripts/g3_stage_archive.py'),'--stage','training'],cwd=ROOT,check=True)
    dump(OUT/'READY_FOR_TEST.json',{'ready_utc':datetime.now(timezone.utc).isoformat(),'selection':'selected_checkpoints_extended.json'})
    dump(OUT/'extension_status.json',{'status':'complete','extra_runs':4})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(OUT/'extension_status.json',{'status':'failed','error':str(e)});traceback.print_exc();sys.exit(1)
