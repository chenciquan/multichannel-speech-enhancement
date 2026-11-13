#!/usr/bin/env python3
"""Continue after separately completed G12A; no unbounded readiness wait."""
import json,subprocess,sys,traceback
from g12_common import ROOT,OUT,now,sha,dump,status


def main():
    assert json.loads((OUT/'ASSET_JOB_COMPLETE.json').read_text())['status']=='ok'
    assert (OUT/'stages/asset/archive_manifest.json').exists()
    testlog=(OUT/'reverberation_unit_tests.log').read_text()
    assert 'Ran 91 tests' in testlog and '\nOK' in testlog
    plan=json.loads((OUT/'frozen_plan.json').read_text());source=ROOT/'artifacts/g10_fresh_voice_v1/systems.json'
    assert sha(source)==plan['G10_systems_sha256'];dump(OUT/'systems.json',json.loads(source.read_text()),True)
    names=['scripts/g12_common.py','scripts/g12_queue.py','scripts/g12_build.py','scripts/g12_signal.py','scripts/g12_asr.py','scripts/g12_finish.py',
        'src/mcse/reverberation.py','src/mcse/runtime.py','src/mcse/beamforming.py','tests/test_reverberation.py']
    ready={'frozen_utc':now(),'files':{name:sha(ROOT/name) for name in names},'full_tests_before_room_scores':91,'asset_report_archived':True}
    dump(OUT/'PIPELINE_READY.json',ready,True)
    def run(script,limit,log):
        assert all(sha(ROOT/name)==h for name,h in ready['files'].items())
        with (OUT/log).open('x') as f:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
        if r.returncode:raise RuntimeError(script+' failed; retain exact logs/no silent retries')
    run('g12_build.py',plan['max_data_seconds'],'build.log');assert (OUT/'stages/preparation/archive_manifest.json').exists()
    run('g12_signal.py',plan['max_signal_seconds'],'signal.log');assert (OUT/'stages/beamforming/archive_manifest.json').exists()
    run('g12_asr.py',plan['max_ASR_seconds'],'asr.log');assert (OUT/'stages/completion/archive_manifest.json').exists()
    status(stage='complete_pending_readback')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('room queue ended\n')
