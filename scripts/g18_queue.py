#!/usr/bin/env python3
import json,subprocess,sys,traceback
from g18_common import ROOT,OUT,now,sha,dump,status


def main():
    assert json.loads((OUT/'PHYSICS_JOB_COMPLETE.json').read_text())['status']=='ok';assert (OUT/'stages/physics/archive_manifest.json').exists();plan=json.loads((OUT/'frozen_plan.json').read_text())
    assert sha(ROOT/'artifacts/g17_fresh_confirmation_v1/systems.json')==plan['G17_registry_sha256'] and sha(OUT/'systems.json')==plan['G17_registry_sha256']
    names=['scripts/g18_common.py','scripts/g18_start_physics.py','scripts/g18_physics.py','scripts/g18_queue.py','scripts/g18_build.py','scripts/g18_signal.py','scripts/g18_asr.py','scripts/g18_finish.py',
        'src/mcse/reverberation.py','src/mcse/geometry.py','src/mcse/runtime.py','src/mcse/beamforming.py','src/mcse/adaptation.py','src/mcse/simulation.py','src/mcse/spatial_noise.py','configs/g18_room_candidate.json']
    ready={'frozen_utc':now(),'files':{n:sha(ROOT/n) for n in names},'physical_report_archive_sha256':sha(OUT/'stages/physics/archive_manifest.json')};dump(OUT/'PIPELINE_READY.json',ready,True)
    def run(script,limit,log):
        assert all(sha(ROOT/n)==h for n,h in ready['files'].items())
        with (OUT/log).open('x') as f:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
        if r.returncode:raise RuntimeError(script+' failed;preserve,no silentretry or conditiondrop')
    status(stage='data_generation');run('g18_build.py',plan['max_data_seconds'],'build.log');assert (OUT/'stages/preparation/archive_manifest.json').exists()
    run('g18_signal.py',plan['max_signal_seconds'],'signal.log');assert (OUT/'stages/beamforming/archive_manifest.json').exists()
    run('g18_asr.py',plan['max_ASR_seconds'],'asr.log');assert (OUT/'stages/completion/archive_manifest.json').exists();status(stage='complete_pending_readback')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('room stress queue ended\n')
