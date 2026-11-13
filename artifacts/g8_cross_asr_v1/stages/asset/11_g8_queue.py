#!/usr/bin/env python3
"""Bounded second-ASR queue; prior report and audit are mandatory entry gates."""
import json,os,subprocess,sys,traceback
from pathlib import Path
from g8_common import ROOT,OUT,now,sha,dump,status


def main():
    gate=ROOT/'artifacts/g7_unseen_noise_v1'
    assert json.loads((gate/'completion_audit.json').read_text())['status']=='ok'
    assert (gate/'MAJOR_STAGE_COMPLETE.json').is_file() and (gate/'stages/completion/archive_manifest.json').is_file()
    OUT.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'configs/g8_ctc_candidate.json').read_text());plan.update(status='frozen_before_asset_download_and_CTC_outputs',frozen_utc=now(),
      evaluation_timeout_seconds=5400,asset_stage_timeout_seconds=1500,previous_stage_archive_sha256=sha(gate/'stages/completion/archive_manifest.json'),
      datasets={ds:{'folder':folder,'expected_rows':n,'signals_sha256':sha(ROOT/folder/'signals.jsonl'),'whisper_sha256':sha(ROOT/folder/'asr_scores.jsonl')} for ds,folder,n in
        [('G5','artifacts/g5_rotation_v1/evaluation',3528),('G7','artifacts/g7_unseen_noise_v1/evaluation',9072)]})
    plan['predeclared_analysis']='Every system and seed, per noise/target/SNR and G5 clean controls; rotated-fixed WER source-cluster bootstrap2000 for both recognizers, fixed seed20261112; no winner selection.12 reused source utterances,3 speakers.'
    dump(OUT/'frozen_plan.json',plan,True);status(stage='asset')
    for script,timeout,logname in [('g8_asset.py',plan['asset_stage_timeout_seconds'],'asset.log'),('g8_evaluate.py',plan['evaluation_timeout_seconds'],'evaluation.log')]:
        with (OUT/logname).open('x') as log:
            result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:raise RuntimeError(script+' failed; inspect saved stage and log before any retry')
        if script=='g8_asset.py':assert (OUT/'stages/asset/archive_manifest.json').is_file()
    status(stage='complete')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('queue ended\n')
