#!/usr/bin/env python3
import json,subprocess,sys,time,traceback
from g15_common import ROOT,OUT,now,sha,dump,status


def main():
    gate=ROOT/'artifacts/g14_corrected_geometry_v1';assert json.loads((gate/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok';assert (gate/'stages/next_decision/archive_manifest.json').exists()
    oldsystems=json.loads((ROOT/'artifacts/g5_rotation_v1/selected_checkpoints.json').read_text())['systems'];comparisons=[r for r in oldsystems if r['variant']=='without_clean_identity'];assert len(comparisons)==6 and all(r['step']==3000 and sha(ROOT/r['path'])==r['checkpoint_sha256'] for r in comparisons)
    OUT.mkdir(exist_ok=False);plan=json.loads((ROOT/'configs/g15_rigid_training_plan.json').read_text());plan.update(frozen_utc=now(),G14_decision_archive_sha256=sha(gate/'stages/next_decision/archive_manifest.json'))
    dump(OUT/'frozen_plan.json',plan,True);dump(OUT/'old_final_step_comparisons.json',comparisons,True)
    names=['configs/g15_rigid_training_plan.json','configs/g3_overnight_plan.json','scripts/g15_common.py','scripts/g15_queue.py','scripts/g15_prepare.py','scripts/g15_train.py','scripts/g15_finish.py',
        'scripts/g3_train.py','scripts/g5_prepare.py','src/mcse/geometry.py','src/mcse/adaptation.py','src/mcse/simulation.py','src/mcse/spatial_noise.py']
    ready={'frozen_utc':now(),'files':{p:sha(ROOT/p) for p in names},'last_full_test_suite':{'stage':'G14','passed':98,'log_sha256':sha(gate/'full98_unit_tests.log')}};dump(OUT/'PIPELINE_READY.json',ready,True)
    def run(script,args,limit,log):
        assert all(sha(ROOT/p)==h for p,h in ready['files'].items())
        with (OUT/log).open('x') as f:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
        if r.returncode:raise RuntimeError(script+' failed; keep incomplete run,no automaticretry')
    run('g15_prepare.py',[],plan['max_data_seconds'],'prepare.log');assert (OUT/'stages/preparation/archive_manifest.json').exists();started=time.monotonic()
    for seed in plan['seeds']:
        if time.monotonic()-started>plan['max_total_training_seconds']-plan['max_per_seed_seconds']:raise TimeoutError('Training reserve for complete seed')
        status(stage='training',seed=seed);run('g15_train.py',['--seed',str(seed)],plan['max_per_seed_seconds'],f'train_{seed}.log')
    run('g15_finish.py',[],300,'finish.log');assert (OUT/'stages/training/archive_manifest.json').exists();status(stage='complete_pending_acceptance')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('training queue ended\n')
