#!/usr/bin/env python3
import json,subprocess,sys,traceback
from g18_common import ROOT,OUT,now,sha,dump,status


def main():
    gate=ROOT/'artifacts/g17_fresh_confirmation_v1';assert json.loads((gate/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok';assert (gate/'stages/next_decision/archive_manifest.json').exists()
    candidate=ROOT/'configs/g18_room_candidate.json';assert sha(candidate)==json.loads((gate/'NEXT_ROOM_DECISION.json').read_text())['candidate_plan_sha256'];OUT.mkdir(exist_ok=False)
    plan=json.loads(candidate.read_text());plan.update(status='frozen_before_G18physics_and_room_outcomes',frozen_utc=now(),G17_decision_archive_sha256=sha(gate/'stages/next_decision/archive_manifest.json'),
       G17_sources_manifest_sha256=sha(ROOT/'data/g17_v1/sources/speech_manifest.jsonl'),G17_registry_sha256=sha(gate/'systems.json'),
       geometry_sha256=sha(ROOT/'configs/g1_geometry.json'),beamforming_sha256=sha(ROOT/'src/mcse/beamforming.py'),runtime_sha256=sha(ROOT/'src/mcse/runtime.py'),
       SNR_audit_mask='Same unquantized reference activity mask used for mixing,frozen beforePCM24;check savedPCM SNR with that mask. RecomputedPCM activity masks separately audited,not silently substituted.',
       SNR_saved_audit_tolerance_db=.002)
    dump(OUT/'frozen_plan.json',plan,True);dump(OUT/'selected_sources.json',[json.loads(line) for line in (ROOT/'data/g17_v1/sources/speech_manifest.jsonl').read_text().splitlines()],True);dump(OUT/'systems.json',json.loads((gate/'systems.json').read_text()),True)
    with (OUT/'physics.log').open('x') as f:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/g18_physics.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=plan['max_physics_seconds'])
    if r.returncode:raise RuntimeError('G18physics failed;preserve before any retry')
    assert (OUT/'stages/physics/archive_manifest.json').exists();dump(OUT/'PHYSICS_JOB_COMPLETE.json',{'status':'ok','completed_utc':now(),'room_data_not_started':True},True);status(stage='physics_complete_data_not_started')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'physics_keep_awake_stop').write_text('physics job ended\n')
