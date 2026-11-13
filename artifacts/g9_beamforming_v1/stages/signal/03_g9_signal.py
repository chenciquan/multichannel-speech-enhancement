#!/usr/bin/env python3
"""Three fixed beamforming systems, no additional neural SE inference."""
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
from g9_common import ROOT,OUT,now,sha,read,dump,status,archive
from mcse.beamforming import mask_mvdr,oracle_delay_and_sum
from mcse.simulation import geometric_delays
from mcse.spatial_noise import direction
from g4_run import score_signal


def main():
    assert (OUT/'stages/preflight/archive_manifest.json').exists();plan=json.loads((OUT/'frozen_plan.json').read_text())
    geom=ROOT/'configs/g1_geometry.json';assert sha(geom)==plan['geometry_sha256'];pos=json.loads(geom.read_text())['microphone_positions_m']
    EVAL=OUT/'evaluation';EVAL.mkdir(exist_ok=False);torch.set_num_threads(2);started=time.monotonic();cases=[]
    for dataset,spec in plan['datasets'].items():
        path=ROOT/spec['folder']/'signals.jsonl';assert sha(path)==spec['signals_sha256']
        selected=[{**r,'dataset':dataset} for r in read(path) if r['system']=='mcnet'];assert len(selected)==spec['expected_cases'];cases+=selected
    assert len(cases)==900
    dump(EVAL/'signal_provenance.json',{'started_utc':now(),'preflight_archive_sha256':sha(OUT/'stages/preflight/archive_manifest.json'),
      'plan_sha256':sha(OUT/'frozen_plan.json'),'cases':900,'systems':plan['systems'],'new_SE_training_or_inference':False,
      'CPU_threads':2,'CPU_metric_workers':2,'precision':'float64/complex128 CPU; saved float32','no_peak_normalization_or_fallback':True},True)
    complete=[];done_count=Counter();checks={'all_input_and_saved_McNet_hashes':True,'exact_FLOAT_roundtrip':True}
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool,(EVAL/'signals.jsonl').open('x',encoding='utf-8') as log:
        pending=set()
        def collect(block):
            nonlocal pending
            if not pending:return
            ready,pending=wait(pending,timeout=None if block else 0,return_when=FIRST_COMPLETED)
            for future in ready:
                row=future.result();complete.append(row);done_count[row['system']]+=1;log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                if row['status']!='ok':raise RuntimeError('Metric failure retained, no fallback')
        for i,r in enumerate(cases):
            if time.monotonic()-started>plan['max_signal_seconds']-180:raise TimeoutError('Signal stage reserve')
            assert sha(r['noisy_path'])==r['input_sha256'] and sha(r['output_path'])==r['output_sha256'] and sha(r['reference_path'])==r['reference_sha256']
            x,sr=sf.read(r['noisy_path'],dtype='float32',always_2d=True);estimate,er=sf.read(r['output_path'],dtype='float32');reference,rr=sf.read(r['reference_path'],dtype='float32')
            assert sr==er==rr==16000 and x.shape==(len(reference),6) and estimate.shape==reference.shape
            for system in plan['systems']:
                began=time.perf_counter()
                if system=='oracle_doa_delay_sum':
                    delays=geometric_delays(pos,direction(r['target_azimuth'],10));y=oracle_delay_and_sum(x,delays);info={'oracle_DOA':True,'oracle_clean_reference':False}
                else:
                    oracle=system=='oracle_ref_mask_mvdr';y,info=mask_mvdr(x,reference if oracle else estimate)
                    info.pop('oracle_information_inside_function',None);info.update(oracle_DOA=False,oracle_clean_reference=oracle)
                seconds=time.perf_counter()-began;assert y.shape==reference.shape and np.isfinite(y).all()
                path=EVAL/'audio'/system/(r['sample_id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
                saved,ys=sf.read(path,dtype='float32');assert ys==16000 and np.array_equal(y,saved)
                row={**r,'system':system,'mask_estimate_path':r['output_path'] if system=='mcnet_mask_mvdr' else r['reference_path'] if system=='oracle_ref_mask_mvdr' else None,
                  'output_path':str(path),'output_sha256':sha(path),'original_mcnet_inference_seconds':r['inference_seconds'],
                  'inference_seconds':seconds,'beamforming_diagnostics':info}
                pending.add(pool.submit(score_signal,row))
                if len(pending)>=8:collect(True)
            if (i+1)%50==0:status(stage='signal',cases_done=i+1,cases_total=900,completed_rows=len(complete));print(json.dumps({'cases_done':i+1}),flush=True)
        while pending:collect(True)
    assert len(complete)==2700 and all(n==900 for n in done_count.values())
    completion={'completed_utc':now(),'rows':len(complete),'elapsed_seconds':time.monotonic()-started,'status':'ok','checks':checks};dump(EVAL/'SIGNAL_COMPLETE.json',completion,True)
    lines=['# G9B 波束形成信号与耗时报告','','协议与前置测试归档后评估。同900条件/系统，3个新系统共2,700条；G5包含36条干净对照。只有mcnet_mask_mvdr是估计掩码主对照，另两项使用oracle信息。未新增增强模型训练/推理，引用保存McNet估计。','','|系统|有效/总数|SI-SDR dB|PESQ|STOI|滤波平均秒|','|---|---:|---:|---:|---:|---:|']
    for system in plan['systems']:
        group=[r for r in complete if r['system']==system];vals=[float(np.mean([r['metrics'][k] for r in group])) for k in ('si_sdr_db','pesq_wb','stoi')]
        lines.append(f'|{system}|{len(group)}/{len(group)}|{vals[0]:.2f}|{vals[1]:.3f}|{vals[2]:.3f}|{np.mean([r["inference_seconds"] for r in group]):.4f}|')
    lines+=['','该表混合干净/不同SNR/方向，仅作完整覆盖检查；最终结果必须分条件讨论。滤波CPU计时包含本函数STFT/ISTFT，不含输入读取、指标、进程加载；主对照另外需要McNet估计，其原生成时间逐例保存在original_mcnet_inference_seconds。不能将本表当整条神经前端的端到端耗时。所有有限值、FLOAT回读和输入/原估计哈希检查通过，无数值失败回填。','', 'UTC：'+now()]
    report=ROOT/'docs/G9B_波束形成信号与耗时报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('signal',report,[EVAL/'signal_provenance.json',EVAL/'SIGNAL_COMPLETE.json',ROOT/'scripts/g9_signal.py',ROOT/'scripts/g9_asr.py',ROOT/'scripts/g9_finish.py'])


if __name__=='__main__':main()
