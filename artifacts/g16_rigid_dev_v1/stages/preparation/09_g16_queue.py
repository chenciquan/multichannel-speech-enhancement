#!/usr/bin/env python3
import json,subprocess,sys,traceback
from g16_common import ROOT,OUT,now,sha,dump,status


def main():
    g15=ROOT/'artifacts/g15_rigid_training_v1';g14=ROOT/'artifacts/g14_corrected_geometry_v1';assert json.loads((g15/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    OUT.mkdir(exist_ok=False);plan=json.loads((ROOT/'configs/g16_rigid_dev_plan.json').read_text());plan.update(frozen_utc=now(),previous_archive_sha256=sha(g15/'stages/acceptance/archive_manifest.json'),
        G14_manifest_sha256=sha(ROOT/'data/g14_v1/manifest.jsonl'),G14_signals_sha256=sha(g14/'evaluation/combined_signals.jsonl'),G15_registry_sha256=sha(g15/'registered_checkpoints.json'),
        beamforming_sha256=sha(ROOT/'src/mcse/beamforming.py'),runtime_sha256=sha(ROOT/'src/mcse/runtime.py'))
    direct=[{**r,'registered_id':'g15/'+r['system']} for r in json.loads((g15/'registered_checkpoints.json').read_text())['systems']]
    beam=[{'system':r['system']+'_mask_mvdr','kind':'mask_mvdr','mask_system':r['system'],'geometry':'rigid','seed':r['seed']} for r in direct];old=json.loads((g14/'systems.json').read_text())
    dump(OUT/'systems.json',{'direct':direct,'beam':beam,'baseline':old['direct']+old['beam']},True);dump(OUT/'frozen_plan.json',plan,True)
    names=['configs/g16_rigid_dev_plan.json','scripts/g16_common.py','scripts/g16_queue.py','scripts/g16_preflight.py','scripts/g16_prepare.py','scripts/g16_signal.py','scripts/g16_asr.py','scripts/g16_finish.py',
        'src/mcse/runtime.py','src/mcse/beamforming.py','src/mcse/adaptation.py','src/mcse/geometry.py']
    ready={'frozen_utc':now(),'files':{name:sha(ROOT/name) for name in names}};dump(OUT/'PIPELINE_READY.json',ready,True)
    def run(script,limit,log):
        assert all(sha(ROOT/n)==h for n,h in ready['files'].items())
        with (OUT/log).open('x') as f:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
        if r.returncode:raise RuntimeError(script+' failed;preserve no silent retry')
    run('g16_preflight.py',plan['max_preflight_seconds'],'preflight.log');assert (OUT/'stages/runtime/archive_manifest.json').exists()
    run('g16_prepare.py',plan['max_data_seconds'],'prepare.log');assert (OUT/'stages/preparation/archive_manifest.json').exists()
    run('g16_signal.py',plan['max_signal_seconds'],'signal.log');assert (OUT/'stages/beamforming/archive_manifest.json').exists()
    run('g16_asr.py',plan['max_ASR_seconds'],'asr.log');assert (OUT/'stages/completion/archive_manifest.json').exists();status(stage='complete_pending_acceptance')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('development queue ended\n')
