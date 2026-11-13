#!/usr/bin/env python3
"""Bounded matched-room direct, blind-filter and oracle-statistic phases."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import soundfile as sf
import torch
from g22_common import ROOT,OUT,BASE,sha,read,dump,now,archive,metadata,guard,status
from g22_filter_worker import score,filtered,initialize
from mcse.runtime import load_system,enhance_array

EV=OUT/'evaluation'


def report(phase,rows,marker):
    name={'direct':'直接输出','blind':'估计掩码滤波','truth':'真值协方差诊断'}[phase]
    lines=[f'# G22 {name}信号阶段','','固定两房间配对输入，全部种子和原有参数不变。下表只核对覆盖，混合条件的均值不用于宣称优越。','',
           '|系统|记录数|SI-SDR dB|PESQ|STOI|','|---|---:|---:|---:|---:|']
    for system in sorted({r['system'] for r in rows}):
        group=[r for r in rows if r['system']==system]
        lines.append(f'|{system}|{len(group)}|'+ '|'.join(f"{np.mean([r['metrics'][k] for r in group]):.4f}" for k in ('si_sdr_db','pesq_wb','stoi'))+'|')
    lines+=['','全部输出由FLOAT WAV回读后评分；无重采样、自动电平恢复、时间拟合或指标回填。干净例另存RMS与NMSE。真值系统明确含不可部署的真实分量信息，不作上界。ASR尚待后续固定设置评估。','',now()]
    path=ROOT/'docs'/f'G22_{name}信号阶段报告.md'
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('signal_'+phase,path,[marker,EV/'provenance.json',ROOT/'scripts/g22_signal.py',ROOT/'scripts/g22_filter_worker.py'])


def main(phase):
    ready=guard();start=time.monotonic();deadline=start+ready['max_phase_seconds']-60
    cases=read(BASE/'manifest.jsonl');assert len(cases)==504
    registry=metadata(OUT/'signal_systems.json');frontends=[s for s in registry['direct'] if s['kind']!='noisy']
    if phase=='direct':
        assert (OUT/'stages/next_signal_decision/archive_manifest.json').exists()
        EV.mkdir(exist_ok=False)
        for c in cases:
            assert sha(c['noisy_path'])==c['input_sha256'] and sha(c['reference_path'])==c['reference_sha256']
        dump(EV/'provenance.json',{'started_utc':now(),'ready_sha256':sha(OUT/'SIGNAL_PIPELINE_READY.json'),
            'manifest_sha256':sha(BASE/'manifest.jsonl'),'FP32':True,'batch':1,'metric_workers':2,
            'all_three_seeds':True,'new_training':False,'parameter_tuning':False,'reference_channel_zero_based':4},True)
    else:assert (OUT/'stages'/('signal_direct' if phase=='blind' else 'signal_blind')/'archive_manifest.json').exists()
    rows=[];resources=[];log_path=EV/(phase+'_signals.jsonl')
    torch.set_num_threads(2);torch.manual_seed(20261119)
    with log_path.open('x',encoding='utf-8') as log, ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        def keep(row):
            assert row['status']=='ok';log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
        if phase=='direct':
            for system in registry['direct']:
                runtime=None
                if system['kind']!='noisy':runtime=load_system(system['registered_id'],'cuda');torch.cuda.reset_peak_memory_stats()
                pending=[]
                for i,c in enumerate(cases,1):
                    assert time.monotonic()<deadline
                    x,sr=sf.read(c['noisy_path'],dtype='float32');assert sr==16000
                    if runtime is None:y=x[:,4].copy();seconds=0.
                    else:y,seconds=enhance_array(runtime,x)
                    path=EV/'direct_audio'/system['system']/(c['sample_id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
                    assert y.shape==(c['frames'],) and y.dtype==np.float32 and np.isfinite(y).all()
                    with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
                    saved,rate=sf.read(path,dtype='float32');assert rate==16000 and np.array_equal(saved,y)
                    r={**c,'system':system['system'],'inference_seconds':seconds,'audio_seconds':len(y)/16000,
                       'output_path':str(path),'output_sha256':sha(path),'oracle_clean_information':False,'deployable':True}
                    pending.append(pool.submit(score,r))
                    if len(pending)>=8:
                        for future in pending:keep(future.result())
                        pending=[];log.flush()
                    if i%84==0:status(stage='signal_direct',system=system['system'],cases=i,total=504,rows=len(rows))
                for future in pending:keep(future.result())
                log.flush()
                if runtime is not None:
                    resources.append({'system':system['system'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                      'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'metadata':runtime.metadata})
                    del runtime;gc.collect();torch.cuda.empty_cache()
                dump(EV/'runtime_resources.json',resources)
        else:
            direct={(r['system'],r['sample_id']):r for r in read(EV/'direct_signals.jsonl')}
            if phase=='blind':jobs=[(method,c,direct[s['system'],c['sample_id']]) for s in frontends for c in cases for method in ('mvdr','mwf')]
            else:jobs=[(method,c,None) for c in cases if c['condition']!='clean' for method in ('mvdr','mwf')]
            for offset in range(0,len(jobs),24):
                assert time.monotonic()<deadline
                for row in pool.map(filtered,jobs[offset:offset+24]):keep(row)
                log.flush()
                if len(rows)%240==0:status(stage='signal_'+phase,rows=len(rows),total=len(jobs))
    expected={'direct':5544,'blind':10080,'truth':864}[phase]
    assert len(rows)==len({(r['system'],r['sample_id']) for r in rows})==expected
    guard()
    marker=EV/(phase.upper()+'_COMPLETE.json')
    dump(marker,{'status':'ok','rows':len(rows),'completed_utc':now(),'signals_sha256':sha(log_path),
                 'elapsed_seconds':time.monotonic()-start,'training':False,'oracle':phase=='truth'},True)
    report(phase,rows,marker)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('direct','blind','truth'));main(parser.parse_args().phase)
