#!/usr/bin/env python3
import json,subprocess,sys,time,traceback
from g11_common import ROOT,OUT,G10,now,sha,dump,status


def main():
    assert json.loads((G10/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok' and (G10/'stages/acceptance/archive_manifest.json').exists()
    OUT.mkdir(parents=True,exist_ok=False);plan=json.loads((ROOT/'configs/g11_component_plan.json').read_text());p10=json.loads((G10/'frozen_plan.json').read_text())
    plan.update(frozen_utc=now(),previous_archive_sha256=sha(G10/'stages/acceptance/archive_manifest.json'),
      source_evidence_sha256={name:sha(ROOT/name) for name in ('data/g10_v1/manifest.jsonl','data/g10_v1/sources/speech_manifest.jsonl','configs/g1_geometry.json','src/mcse/beamforming.py','src/mcse/simulation.py','src/mcse/spatial_noise.py','artifacts/g10_fresh_voice_v1/systems.json','artifacts/g10_fresh_voice_v1/evaluation/combined_signals.jsonl','artifacts/g10_fresh_voice_v1/evaluation/Whisper/asr_scores.jsonl','artifacts/g10_fresh_voice_v1/evaluation/CTC/asr_scores.jsonl')})
    assert plan['source_evidence_sha256']['configs/g1_geometry.json']==p10['geometry_sha256'] and plan['source_evidence_sha256']['src/mcse/beamforming.py']==p10['beamforming_code_sha256']
    dump(OUT/'frozen_plan.json',plan,True)
    def run(script,limit,log):
        with (OUT/log).open('x') as f:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
        if r.returncode:raise RuntimeError(script+' failed; preserve evidence and inspect before any revision')
    run('g11_preflight.py',plan['max_preflight_seconds'],'preflight.log');status(stage='preflight_archived_waiting_analyzer');deadline=time.monotonic()+1800
    while not (OUT/'ANALYZER_READY.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('G11 analyzer readiness timeout')
        time.sleep(5)
    run('g11_run.py',plan['max_analysis_seconds'],'analysis.log');status(stage='complete')


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('queue ended\n')
