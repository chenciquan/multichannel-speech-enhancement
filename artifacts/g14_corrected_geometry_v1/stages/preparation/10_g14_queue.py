#!/usr/bin/env python3
import json,subprocess,sys,traceback
from g14_common import ROOT,OUT,now,sha,dump,status


def main():
    gate=ROOT/'artifacts/g13_geometry_audit_v1';assert json.loads((gate/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok';assert (gate/'stages/acceptance/archive_manifest.json').exists()
    OUT.mkdir(exist_ok=False);plan=json.loads((ROOT/'configs/g14_corrected_geometry_plan.json').read_text());plan.update(frozen_utc=now(),previous_archive_sha256=sha(gate/'stages/acceptance/archive_manifest.json'),
       G10_manifest_sha256=sha(ROOT/'data/g10_v1/manifest.jsonl'),geometry_sha256=sha(ROOT/'configs/g1_geometry.json'),geometry_code_sha256=sha(ROOT/'src/mcse/geometry.py'),
       legacy_rotation_source_sha256=sha(ROOT/'scripts/g5_prepare.py'),beamforming_sha256=sha(ROOT/'src/mcse/beamforming.py'),negative_control_output_max_abs=1e-5)
    dump(OUT/'systems.json',json.loads((ROOT/'artifacts/g10_fresh_voice_v1/systems.json').read_text()),True)
    dump(OUT/'selected_sources.json',json.loads((ROOT/'artifacts/g12_room_stress_v2/selected_sources.json').read_text()),True);dump(OUT/'frozen_plan.json',plan,True)
    names=['configs/g14_corrected_geometry_plan.json','scripts/g14_common.py','scripts/g14_queue.py','scripts/g14_build.py','scripts/g14_signal.py','scripts/g14_asr.py','scripts/g14_finish.py',
        'src/mcse/geometry.py','src/mcse/runtime.py','src/mcse/beamforming.py','tests/test_g14_analysis.py','tests/test_azimuth_geometry.py']
    ready={'frozen_utc':now(),'files':{n:sha(ROOT/n) for n in names},'previous_G13_acceptance_sha256':sha(gate/'FINAL_ACCEPTANCE_AUDIT.json')};dump(OUT/'PIPELINE_READY.json',ready,True)
    def run(script,limit,log):
        assert all(sha(ROOT/n)==h for n,h in ready['files'].items())
        with (OUT/log).open('x') as f:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
        if r.returncode:raise RuntimeError(script+' failed; retain evidence,no silent retries')
    dump(OUT/'analysis_test_observation.json',{'status':'ok','tests':2,'observed_before_queue':True,'source':'Successful pre-queue exec_command unittest test_g14_analysis.py; not a second test execution',
        'test_code_sha256':sha(ROOT/'tests/test_g14_analysis.py'),'analysis_code_sha256':sha(ROOT/'scripts/g14_finish.py')},True)
    run('g14_build.py',plan['max_data_seconds'],'build.log');assert (OUT/'stages/preparation/archive_manifest.json').exists()
    run('g14_signal.py',plan['max_signal_seconds'],'signal.log');assert (OUT/'stages/beamforming/archive_manifest.json').exists()
    run('g14_asr.py',plan['max_ASR_seconds'],'asr.log');assert (OUT/'stages/completion/archive_manifest.json').exists();status(stage='complete_pending_readback')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('paired geometry queue ended\n')
