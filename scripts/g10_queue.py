#!/usr/bin/env python3
import json,subprocess,sys,time,traceback
from g10_common import ROOT,OUT,now,sha,dump,status


def main():
    gate=ROOT/'artifacts/g9_beamforming_v1';assert json.loads((gate/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    assert (gate/'stages/acceptance/archive_manifest.json').exists();OUT.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'configs/g10_fresh_speaker_plan.json').read_text());plan.update(frozen_utc=now(),previous_archive_sha256=sha(gate/'stages/acceptance/archive_manifest.json'),
      geometry_sha256=sha(ROOT/'configs/g1_geometry.json'),beamforming_code_sha256=sha(ROOT/'src/mcse/beamforming.py'))
    from mcse.runtime import archived_json
    selected=archived_json(ROOT/'artifacts/g5_rotation_v1/selected_checkpoints.json',ROOT/'artifacts/g5_rotation_v1/stages/training/archive_manifest.json')
    direct=[{'system':'noisy','kind':'noisy'},{'system':'mcnet','kind':'mcnet'}]+[r for r in selected['systems'] if r['variant']=='without_clean_identity']
    assert len(direct)==8
    for r in direct:
        if r['kind']=='adapter':assert sha(ROOT/r['path'])==r['checkpoint_sha256']
    beam=[{'system':r['system']+'_mask_mvdr','kind':'mask_mvdr','mask_system':r['system'],'geometry':r.get('geometry'),'seed':r.get('seed')} for r in direct if r['kind']!='noisy']
    dump(OUT/'systems.json',{'direct':direct,'beam':beam},True);plan['systems_sha256']=sha(OUT/'systems.json');dump(OUT/'frozen_plan.json',plan,True)
    def run(script,timeout,log):
        with (OUT/log).open('x') as f:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        if r.returncode:raise RuntimeError(script+' failed; inspect saved artifacts, no silent retry')
    run('g10_fetch.py',plan['fetch_timeout_seconds']+60,'source_fetch.log');status(stage='sources_archived_waiting_builder');limit=time.monotonic()+1800
    while not (OUT/'PIPELINE_READY.json').exists():
        if time.monotonic()>limit:raise TimeoutError('G10 pipeline readiness deadline')
        time.sleep(5)
    run('g10_build.py',plan['max_data_seconds'],'build.log');assert (OUT/'stages/preparation/archive_manifest.json').exists()
    run('g10_signal.py',plan['max_signal_seconds'],'signal.log');assert (OUT/'stages/beamforming/archive_manifest.json').exists()
    run('g10_asr.py',plan['max_ASR_seconds'],'asr.log');status(stage='complete')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('queue ended\n')
