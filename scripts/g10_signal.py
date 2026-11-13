#!/usr/bin/env python3
"""Frozen neural estimates first, then seven standard MVDR pipelines."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import g4_run as rt
from g10_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive
from mcse.beamforming import mask_mvdr


def main():
    plan=json.loads((OUT/'frozen_plan.json').read_text());registry=json.loads((OUT/'systems.json').read_text());start=time.monotonic()
    prep=json.loads((OUT/'PREPARATION_COMPLETE.json').read_text());assert sha(BASE/'manifest.jsonl')==prep['manifest_sha256'] and sha(OUT/'systems.json')==plan['systems_sha256']
    cases=read(BASE/'manifest.jsonl');assert len(cases)==624
    for r in cases:assert sha(r['noisy_path'])==r['input_sha256'] and sha(r['reference_path'])==r['reference_sha256']
    assert sha(ROOT/'src/mcse/beamforming.py')==plan['beamforming_code_sha256'];torch.set_num_threads(2)
    rt.OUT=OUT;rt.BASE=BASE;rt.EVAL=OUT/'evaluation';rt.STOP=time.time()+plan['max_signal_seconds']-120;rt.EVAL.mkdir(exist_ok=False)
    rt.signal_stage(cases,registry['direct'],manifest_path=BASE/'manifest.jsonl',protocol_path=OUT/'frozen_plan.json',supplement_path=None,
       previous_archive=OUT/'stages/preparation/archive_manifest.json',report_path=ROOT/'docs/G10C_新说话人直接输出信号报告.md',report_title='G10C 新说话人直接输出信号报告',
       selection_description='All3 previously validation-selected G5 no-identity fixed/rotated seeds; no fresh-source tuning')
    direct=read(OUT/'evaluation/signals.jsonl');assert len(direct)==4992;assert (OUT/'stages/signal/archive_manifest.json').exists()
    idx={(r['system'],r['sample_id']):r for r in direct};complete=[]
    dump(OUT/'evaluation/beam_provenance.json',{'started_utc':now(),'previous_archive_sha256':sha(OUT/'stages/signal/archive_manifest.json'),
       'beamforming_code_sha256':sha(ROOT/'src/mcse/beamforming.py'),'no_oracle_reference_or_DOA':True,'no_new_neural_forward':True,'CPU_precision':'float64/complex128'},True)
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool,(OUT/'evaluation/beam_signals.jsonl').open('x') as log:
        pending=set()
        def collect(block):
            nonlocal pending
            if not pending:return
            ready,pending=wait(pending,timeout=None if block else 0,return_when=FIRST_COMPLETED)
            for f in ready:
                r=f.result();complete.append(r);log.write(json.dumps(r)+'\n');log.flush()
                if r['status']!='ok':raise RuntimeError('Beam metric failure retained, no fallback')
        for system in registry['beam']:
            for index,case in enumerate(cases):
                if time.monotonic()-start>plan['max_signal_seconds']-180:raise TimeoutError('Direct+MVDR signal reserve')
                r=idx[(system['mask_system'],case['sample_id'])];assert sha(r['output_path'])==r['output_sha256']
                x,sr=sf.read(r['noisy_path'],dtype='float32',always_2d=True);estimate,er=sf.read(r['output_path'],dtype='float32');assert sr==er==16000
                began=time.perf_counter();y,info=mask_mvdr(x,estimate);seconds=time.perf_counter()-began
                info.pop('oracle_information_inside_function',None);info.update(oracle_clean_reference=False,oracle_DOA=False)
                path=OUT/'evaluation/beam_audio'/system['system']/(r['sample_id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
                saved,yr=sf.read(path,dtype='float32');assert yr==16000 and np.array_equal(y,saved)
                record={**r,'system':system['system'],'source_system':system['mask_system'],'mask_estimate_path':r['output_path'],'mask_estimate_sha256':r['output_sha256'],
                  'output_path':str(path),'output_sha256':sha(path),'direct_inference_seconds':r['inference_seconds'],'inference_seconds':seconds,'beamforming_diagnostics':info}
                pending.add(pool.submit(rt.score_signal,record))
                if len(pending)>=8:collect(True)
                if (index+1)%100==0:status(stage='beamforming',system=system['system'],cases_done=index+1,cases_total=624,beam_rows_completed=len(complete))
            while pending:collect(True)
    assert len(complete)==4368;allrows=direct+complete;assert len(allrows)==9360 and len({(r['system'],r['sample_id']) for r in allrows})==9360
    assert len(Counter(r['system'] for r in allrows))==15 and all(n==624 for n in Counter(r['system'] for r in allrows).values())
    with (OUT/'evaluation/combined_signals.jsonl').open('x') as f:
        for r in allrows:f.write(json.dumps(r)+'\n')
    dump(OUT/'evaluation/BEAM_COMPLETE.json',{'status':'ok','completed_utc':now(),'beam_rows':4368,'combined_rows':9360,'combined_sha256':sha(OUT/'evaluation/combined_signals.jsonl'),'elapsed_direct_plus_beam_seconds':time.monotonic()-start},True)
    lines=['# G10D 新说话人MVDR输出信号报告','','直接输出报告归档后，以同一G9固定滤波器处理全部七种冻结估计，无真值掩码/方位。新说话人16句/4人；7×624=4,368条MVDR输出，与4,992条直接输出合并9,360条，15系统全部保留。','','|掩码前端|记录|SI-SDR dB|PESQ|STOI|','|---|---:|---:|---:|---:|']
    for s in registry['beam']:
        rows=[r for r in complete if r['system']==s['system']];values=[np.mean([r['metrics'][k] for r in rows]) for k in ('si_sdr_db','pesq_wb','stoi')]
        lines.append(f'|{s["mask_system"]}|{len(rows)}|{values[0]:.2f}|{values[1]:.3f}|{values[2]:.3f}|')
    lines+=['','此处混合干净/噪声/方向，仅作阶段覆盖核验，不用于主结论。全部FLOAT回读一致、有限值和指标成功，无错误回填。ASR前保存组合清单SHA与脚本；下一步Whisper报告归档后再运行CTC并联合分条件分析。','', 'UTC：'+now()]
    report=ROOT/'docs/G10D_新说话人MVDR输出信号报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('beamforming',report,[OUT/'evaluation/beam_provenance.json',OUT/'evaluation/BEAM_COMPLETE.json',ROOT/'scripts/g10_signal.py',ROOT/'scripts/g10_asr.py',ROOT/'scripts/g10_finish.py'])


if __name__=='__main__':main()
