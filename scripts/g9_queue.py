#!/usr/bin/env python3
import json,subprocess,sys,time,traceback
from g9_common import ROOT,OUT,now,sha,dump,status


def main():
    gate=ROOT/'artifacts/g8_cross_asr_v1'
    assert json.loads((gate/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok' and (gate/'stages/acceptance/archive_manifest.json').exists()
    OUT.mkdir(parents=True,exist_ok=False);plan=json.loads((ROOT/'configs/g9_beamforming_plan.json').read_text())
    plan.update(frozen_utc=now(),previous_archive_sha256=sha(gate/'stages/acceptance/archive_manifest.json'),geometry_sha256=sha(ROOT/'configs/g1_geometry.json'),
      datasets={ds:{'folder':folder,'expected_cases':count,'signals_sha256':sha(ROOT/folder/'signals.jsonl'),'whisper_sha256':sha(ROOT/folder/'asr_scores.jsonl')} for ds,folder,count in
       [('G5','artifacts/g5_rotation_v1/evaluation',252),('G7','artifacts/g7_unseen_noise_v1/evaluation',648)]})
    dump(OUT/'frozen_plan.json',plan,True)
    def run(script,limit,log):
        with (OUT/log).open('x') as f:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
        if r.returncode:raise RuntimeError(script+' failed; inspect saved log and stage, no silent retry')
    run('g9_preflight.py',plan['max_preflight_seconds']+30,'preflight.log')
    status(stage='preflight_archived_waiting_evaluator');deadline=time.monotonic()+1800
    while not (OUT/'EVALUATOR_READY.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('G9 evaluator readiness deadline')
        time.sleep(5)
    run('g9_signal.py',plan['max_signal_seconds'],'signal.log')
    assert (OUT/'stages/signal/archive_manifest.json').exists()
    run('g9_asr.py',plan['max_ASR_seconds'],'asr.log');status(stage='complete')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('queue ended\n')
