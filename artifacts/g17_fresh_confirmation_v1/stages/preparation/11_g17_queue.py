#!/usr/bin/env python3
import json,subprocess,sys,traceback
from g17_common import ROOT,OUT,now,sha,dump,status


def main():
    gate=ROOT/'artifacts/g16_rigid_dev_v1';assert json.loads((gate/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok';assert (gate/'stages/next_decision/archive_manifest.json').exists()
    candidate=ROOT/'configs/g17_fresh_confirmation_plan.json';decision=json.loads((gate/'NEXT_CONFIRMATION_DECISION.json').read_text());assert sha(candidate)==decision['G17_candidate_plan_sha256']
    OUT.mkdir(exist_ok=False);plan=json.loads(candidate.read_text());plan.update(status='frozen_before_fresh_source_fetch',frozen_utc=now(),previous_archive_sha256=sha(gate/'stages/next_decision/archive_manifest.json'),
       geometry_sha256=sha(ROOT/'configs/g1_geometry.json'),beamforming_sha256=sha(ROOT/'src/mcse/beamforming.py'),runtime_sha256=sha(ROOT/'src/mcse/runtime.py'),prior_registry_sha256=sha(gate/'systems.json'),source_endpoint_record_sha256=sha(ROOT/'artifacts/g17_source_endpoint_v1.json'))
    previous=json.loads((gate/'systems.json').read_text());direct=[{**r,'registered_id':'mcnet' if r['kind']=='mcnet' else ('g5/'+r['system'] if r['kind']=='adapter' else None)} for r in previous['baseline'] if r['kind']!='mask_mvdr']+previous['direct'];beam=[r for r in previous['baseline'] if r['kind']=='mask_mvdr']+previous['beam']
    assert len(direct)==11 and len(beam)==10 and len({r['system'] for r in direct+beam})==21
    for r in direct:
        if r['kind']=='adapter':assert r['step']==3000 and sha(ROOT/r['path'])==r['checkpoint_sha256']
    dump(OUT/'systems.json',{'direct':direct,'beam':beam,'baseline':[]},True);plan['systems_sha256']=sha(OUT/'systems.json');dump(OUT/'frozen_plan.json',plan,True)
    names=['configs/g17_fresh_confirmation_plan.json','configs/g1_geometry.json','scripts/g17_common.py','scripts/g17_queue.py','scripts/g17_fetch.py','scripts/g17_build.py','scripts/g17_signal.py','scripts/g17_asr.py','scripts/g17_finish.py',
       'src/mcse/runtime.py','src/mcse/geometry.py','src/mcse/beamforming.py','src/mcse/adaptation.py','src/mcse/simulation.py','src/mcse/spatial_noise.py']
    ready={'frozen_utc':now(),'files':{n:sha(ROOT/n) for n in names},'shared_numeric_QA_log_sha256':sha(gate/'full98_runtime_tests.log')};dump(OUT/'PIPELINE_READY.json',ready,True)
    def run(script,limit,log):
        assert all(sha(ROOT/n)==h for n,h in ready['files'].items())
        with (OUT/log).open('x') as f:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
        if r.returncode:raise RuntimeError(script+' failed;preserve evidence,no silent retry or source substitution')
    run('g17_fetch.py',plan['fetch_timeout_seconds']+60,'fetch.log');assert (OUT/'stages/sources/archive_manifest.json').exists()
    run('g17_build.py',plan['max_data_seconds'],'build.log');assert (OUT/'stages/preparation/archive_manifest.json').exists()
    run('g17_signal.py',plan['max_signal_seconds'],'signal.log');assert (OUT/'stages/beamforming/archive_manifest.json').exists()
    run('g17_asr.py',plan['max_ASR_seconds'],'asr.log');assert (OUT/'stages/completion/archive_manifest.json').exists();status(stage='complete_pending_acceptance')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('fresh-confirmation queue ended\n')
