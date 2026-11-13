#!/usr/bin/env python3
"""Reuse the frozen evaluator with explicit G5 paths; no historical overwrite."""
import json,sys,traceback
from pathlib import Path
import g4_run as runtime
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g5_rotation_v1';BASE=ROOT/'data/g5_v1'
def main():
    assert (OUT/'TRAIN_COMPLETE.json').exists() and (OUT/'stages/training/archive_manifest.json').exists()
    selected=json.loads((OUT/'selected_checkpoints.json').read_text());assert not selected['confirmation_opened']
    runtime.OUT=OUT;runtime.EVAL=OUT/'evaluation';runtime.BASE=BASE/'confirmation';runtime.EVAL.mkdir(exist_ok=False)
    cases=runtime.read(BASE/'confirmation/manifest.jsonl');assert len(cases)==252 and runtime.sha(BASE/'confirmation/manifest.jsonl')==selected['confirmation_manifest_sha256']
    systems=[{'system':'noisy','kind':'noisy'},{'system':'mcnet','kind':'mcnet'}]+selected['systems'];runtime.dump(runtime.EVAL/'systems.json',systems)
    runtime.status(stage='signal',systems=len(systems),cases_per_system=252)
    runtime.signal_stage(cases,systems,manifest_path=BASE/'confirmation/manifest.jsonl',protocol_path=OUT/'frozen_plan.json',supplement_path=None,
       previous_archive=OUT/'stages/training/archive_manifest.json',report_path=ROOT/'docs/G5C_新说话人信号指标报告.md',report_title='G5C 新说话人信号指标报告',
       selection_description='All complete four-cell seed blocks; G5 shared54-case validation SI-SDR with initial STOI gate; no confirmation or WER tuning')
    runtime.asr_stage()
    from g5_finish import finish_report
    finish_report(cases,systems,selected);runtime.status(stage='complete',report='docs/G5_整场景旋转与新说话人确认报告.md')
if __name__=='__main__':
    try:main()
    except Exception as e:
        runtime.OUT=OUT;runtime.status(stage='failed',error_type=type(e).__name__,error=str(e));traceback.print_exc();sys.exit(1)
