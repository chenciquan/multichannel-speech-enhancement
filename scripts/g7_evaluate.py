#!/usr/bin/env python3
"""Frozen G5 checkpoint transfer; bounded reused evaluator and new report paths."""
import json,sys,time,traceback
from pathlib import Path
import g4_run as rt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g7_unseen_noise_v1';BASE=ROOT/'data/g7_v1'
def main():
    assert (OUT/'stages/preparation/archive_manifest.json').exists();plan=json.loads((OUT/'frozen_plan.json').read_text());cases=rt.read(BASE/'manifest.jsonl');systems=json.loads((OUT/'systems.json').read_text())
    assert len(cases)==648 and len(systems)==14
    rt.OUT=OUT;rt.EVAL=OUT/'evaluation';rt.BASE=BASE;rt.STOP=time.time()+plan['max_evaluation_stage_seconds']-120;rt.EVAL.mkdir(exist_ok=False)
    rt.status(stage='signal',systems=len(systems),cases_per_system=len(cases))
    rt.signal_stage(cases,systems,manifest_path=BASE/'manifest.jsonl',protocol_path=OUT/'frozen_plan.json',supplement_path=None,
       previous_archive=OUT/'stages/preparation/archive_manifest.json',report_path=ROOT/'docs/G7B_新噪声信号指标报告.md',report_title='G7B 新噪声信号指标报告',
       selection_description='All G5 checkpoints frozen from prior validation,3 complete seed blocks; no new tuning or best-seed selection')
    rt.asr_stage()
    from g7_finish import finish_report
    finish_report(cases,systems);rt.status(stage='complete',report='docs/G7_新噪声迁移与方向权衡报告.md')
if __name__=='__main__':
    try:main()
    except Exception as e:
        rt.OUT=OUT;rt.status(stage='failed',error_type=type(e).__name__,error=str(e));traceback.print_exc();sys.exit(1)
