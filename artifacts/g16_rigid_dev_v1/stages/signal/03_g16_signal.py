#!/usr/bin/env python3
"""Serial registered runtimes, bounded metric workers, then frozen MVDR."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import gc,json,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from multiprocessing import get_context
import numpy as np
import soundfile as sf
import torch
import g4_run as metric
from g16_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive
from mcse.runtime import load_system,enhance_array
from mcse.beamforming import mask_mvdr


def stage_report(stage,title,path,rows,extra):
    lines=['# '+title,'',extra,'','|系统|记录|SI-SDR dB|PESQ|STOI|','|---|---:|---:|---:|---:|']
    for name in sorted({r['system'] for r in rows}):
        a=[r for r in rows if r['system']==name];v=[np.mean([r['metrics'][k] for r in a]) for k in ('si_sdr_db','pesq_wb','stoi')]
        lines.append(f'|{name}|{len(a)}|{v[0]:.2f}|{v[1]:.3f}|{v[2]:.3f}|')
    lines+=['','上表混合干净/噪声/正确训练开发/方向，仅为覆盖核验；不能据此宣称总体优越。最终按训练策略、方向、噪声、场型和全部种子分析。完整FLOAT回读一致，指标无回填，WER另评。','', 'UTC：'+now()]
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive(stage,path,[OUT/'evaluation'/('DIRECT_COMPLETE.json' if stage=='signal' else 'BEAM_COMPLETE.json'),OUT/'evaluation'/('provenance.json' if stage=='signal' else 'beam_provenance.json'),ROOT/'scripts/g16_signal.py'])


def main():
    plan=json.loads((OUT/'frozen_plan.json').read_text());reg=json.loads((OUT/'systems.json').read_text());start=time.monotonic()
    assert (OUT/'stages/preparation/archive_manifest.json').exists();prep=json.loads((OUT/'PREPARATION_COMPLETE.json').read_text())
    assert sha(OUT/'case_manifest.jsonl')==prep['manifest_sha256'];cases=read(OUT/'case_manifest.jsonl');assert len(cases)==168
    for r in cases:assert sha(r['noisy_path'])==r['input_sha256'] and sha(r['reference_path'])==r['reference_sha256']
    assert sha(ROOT/'src/mcse/beamforming.py')==plan['beamforming_sha256'];torch.set_num_threads(2);torch.manual_seed(20261119)
    ev=OUT/'evaluation';ev.mkdir(exist_ok=False);direct=[];resource=[]
    dump(ev/'provenance.json',{'started_utc':now(),'manifest_sha256':sha(OUT/'case_manifest.jsonl'),'systems_sha256':sha(OUT/'systems.json'),
       'previous_archive_sha256':sha(OUT/'stages/preparation/archive_manifest.json'),'training':False,'batch':1,'FP32':True,
       'runtime_sha256':sha(ROOT/'src/mcse/runtime.py'),'device':'cuda','metric_workers':2,'per_system_release':True},True)
    def guard():
        if time.monotonic()-start>plan['max_signal_seconds']-120:raise TimeoutError('G16 signal reserve')
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        for mode in ('direct','beam'):
            output_rows=direct if mode=='direct' else [];pending=set()
            log=(ev/('signals.jsonl' if mode=='direct' else 'beam_signals.jsonl')).open('x')
            def collect(block):
                nonlocal pending
                if not pending:return
                ready,pending=wait(pending,timeout=None if block else 0,return_when=FIRST_COMPLETED)
                for f in ready:
                    r=f.result();output_rows.append(r);log.write(json.dumps(r)+'\n');log.flush()
                    if r['status']!='ok':raise RuntimeError('Metric failure retained; no fallback')
            if mode=='beam':
                idx={(r['system'],r['sample_id']):r for r in direct}
                dump(ev/'beam_provenance.json',{'started_utc':now(),'previous_archive_sha256':sha(OUT/'stages/signal/archive_manifest.json'),
                    'beamforming_code_sha256':sha(ROOT/'src/mcse/beamforming.py'),'no_oracle_reference_or_DOA':True,'no_new_neural_forward':True,'CPU_precision':'float64/complex128'},True)
            for system in reg[mode]:
                name=system['system'];runtime=None;began_system=time.monotonic()
                if mode=='direct' and system['kind']!='noisy':
                    runtime=load_system(system['registered_id'],'cuda');torch.cuda.reset_peak_memory_stats()
                for ix,case in enumerate(cases):
                    guard();x,sr=sf.read(case['noisy_path'],dtype='float32',always_2d=True);assert sr==16000;began=time.perf_counter()
                    info={}
                    if mode=='direct':
                        if runtime is None:y=x[:,4].copy();seconds=time.perf_counter()-began
                        else:y,seconds=enhance_array(runtime,x)
                        record={**case,'system':name,'inference_seconds':seconds,'audio_seconds':len(y)/16000}
                    else:
                        r=idx[(system['mask_system'],case['sample_id'])];assert sha(r['output_path'])==r['output_sha256'];estimate,er=sf.read(r['output_path'],dtype='float32');assert er==16000
                        began=time.perf_counter();y,info=mask_mvdr(x,estimate);seconds=time.perf_counter()-began;info.pop('oracle_information_inside_function',None);info.update(oracle_clean_reference=False,oracle_DOA=False)
                        record={**case,'system':name,'source_system':system['mask_system'],'mask_estimate_path':r['output_path'],'mask_estimate_sha256':r['output_sha256'],
                            'direct_inference_seconds':r['inference_seconds'],'inference_seconds':seconds,'audio_seconds':len(y)/16000,'beamforming_diagnostics':info}
                    assert y.shape==(len(x),) and np.isfinite(y).all();path=ev/('audio' if mode=='direct' else 'beam_audio')/name/(case['sample_id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
                    saved,yr=sf.read(path,dtype='float32');assert yr==16000 and np.array_equal(y,saved)
                    record.update(output_path=str(path),output_sha256=sha(path));pending.add(pool.submit(metric.score_signal,record))
                    if len(pending)>=8:collect(True)
                    if (ix+1)%84==0:status(stage=mode,system=name,cases_done=ix+1,total=168,scored_rows=len(output_rows))
                while pending:collect(True)
                if runtime is not None:
                    resource.append({'system':name,'elapsed_seconds_with_metrics':time.monotonic()-began_system,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'metadata':runtime.metadata});del runtime;gc.collect();torch.cuda.empty_cache()
                dump(ev/'runtime_resources.json',resource)
            log.close()
            assert len(output_rows)==168*len(reg[mode]) and all(r['status']=='ok' for r in output_rows)
            if mode=='direct':
                dump(ev/'DIRECT_COMPLETE.json',{'status':'ok','rows':504,'completed_utc':now(),'signals_sha256':sha(ev/'signals.jsonl')},True)
                stage_report('signal','G16C 正确训练开发直接输出信号报告',ROOT/'docs/G16C_正确训练直接输出信号报告.md',direct,
                    '数据报告归档后，3个新直接系统各168条件，共504条。FP32/batch1，每个模型单独创建和释放；逐系统实际Torch分配/保留峰值另存，不以容量探测替代。无训练或新权重。')
            else:
                allrows=read(OUT/'baseline_signals.jsonl')+direct+output_rows;assert len(allrows)==3528 and all(n==168 for n in Counter(r['system'] for r in allrows).values())
                with (ev/'combined_signals.jsonl').open('x') as f:
                    for r in allrows:f.write(json.dumps(r)+'\n')
                dump(ev/'BEAM_COMPLETE.json',{'status':'ok','beam_rows':504,'combined_rows':3528,'completed_utc':now(),'combined_sha256':sha(ev/'combined_signals.jsonl'),
                    'elapsed_direct_plus_beam_seconds':time.monotonic()-start},True)
                stage_report('beamforming','G16D 正确训练开发估计掩码MVDR信号报告',ROOT/'docs/G16D_正确训练MVDR信号报告.md',output_rows,
                    '直接输出报告归档后，3个新对应神经前端使用冻结G9掩码与Souden MVDR，共504条。掩码仅来自带噪输入与对应神经估计；没有真值语音/方位。CPU float64/complex128，前端耗时与滤波耗时分别存档。')


if __name__=='__main__':main()
