#!/usr/bin/env python3
"""Bounded frozen-model stress evaluation, with stage reports before advancing."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import hashlib,json,runpy,shutil,sys,time,traceback
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.evaluation import MetricEvaluator,waveform_diagnostics
from mcse.asr_scoring import score_words,corpus_score
OUT=ROOT/'artifacts/g4_counterfactual_v1';EVAL=OUT/'evaluation';BASE=ROOT/'data/g4_v1'
STOP=datetime.fromisoformat('2026-11-11T21:44:15+00:00').timestamp()
def now():return datetime.now(timezone.utc).isoformat()
def guard():
    if time.time()>STOP:raise TimeoutError('Stage reserve deadline; preserve partial evidence')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def status(**kw):dump(OUT/'status.json',{'time_utc':now(),**kw})
def archive(stage,report,files):
    folder=OUT/'stages'/stage;folder.mkdir(parents=True,exist_ok=False);entries=[]
    for i,p in enumerate([report]+files):
        dst=folder/f'{i:02d}_{p.name}';shutil.copy2(p,dst);entries.append({'source':str(p.relative_to(ROOT)),'snapshot':dst.name,'sha256':sha(p)})
    dump(folder/'archive_manifest.json',{'saved_at_utc':now(),'files':entries})
_evaluator=None
def score_signal(record):
    global _evaluator
    try:
        if _evaluator is None:_evaluator=MetricEvaluator()
        s,sr=sf.read(record['reference_path'],dtype='float32');y,yr=sf.read(record['output_path'],dtype='float32');assert sr==yr==16000
        record['metrics']=_evaluator.evaluate(s,y);record['waveform_diagnostics']=waveform_diagnostics(s,y)
        record['status']='ok' if all(v['status']=='ok' for v in record['metrics']['status'].values()) else 'metric_error'
    except Exception as e:record.update(status='error',error_type=type(e).__name__,error=str(e))
    return record
def signal_stage(cases,systems,*,manifest_path=None,protocol_path=None,supplement_path='default',previous_archive=None,report_path=None,report_title='G4B 冻结模型信号评估报告',selection_description=None):
    import torch
    from mcse.adaptation import attach_lora,adapter_state,load_adapter
    torch.set_num_threads(2);torch.manual_seed(20261112)
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g4_inference');repo=ROOT/'third_party/McNet';cp=repo/rt['OFFLINE_CHECKPOINT']
    rt['verify_upstream'](repo,cp);model,icirm,_=rt['load_model'](repo,cp,torch.device('cuda'));attach_lora(model);zero=adapter_state(model)
    manifest_path=manifest_path or BASE/'manifest_extended.jsonl';protocol_path=protocol_path or OUT/'frozen_protocol.json'
    if supplement_path=='default':supplement_path=OUT/'rotation_supplement.json'
    previous_archive=previous_archive or ROOT/'artifacts/g3_overnight_v1/stages/completion/archive_manifest.json'
    provenance={'started_utc':now(),'training_performed':False,'manifest_sha256':sha(manifest_path),
                'protocol_sha256':sha(protocol_path),'supplement_sha256':sha(supplement_path) if supplement_path else None,
                'previous_stage_archive_sha256':sha(previous_archive),
                'fp32_inference':True,'blas_threads':1,'cpu_metric_workers':2,'metric_source':'saved FLOAT WAV, no resampling/normalization/alignment fitting',
                'checkpoint_selection':selection_description or 'G3 validation-selected first seed for every variant; no G4 tuning'}
    dump(EVAL/'provenance.json',provenance)
    start=time.perf_counter();completed=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool,(EVAL/'signals.jsonl').open('x',encoding='utf-8') as log:
        pending=set()
        def collect(block):
            nonlocal pending
            if not pending:return
            done,remaining=wait(pending,return_when=FIRST_COMPLETED,timeout=None if block else 0)
            pending=remaining
            for future in done:
                row=future.result();completed.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
        for system in systems:
            name=system['system'];folder=EVAL/'audio'/name;folder.mkdir(parents=True)
            if system['kind']=='mcnet':load_adapter(model,zero)
            elif system['kind']=='adapter':
                path=ROOT/system['path'];assert sha(path)==system['checkpoint_sha256'];saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['step']==system['step']
                if 'seed' in system:assert saved['seed']==system['seed']
                if 'variant' in system:assert saved['variant']==system['variant']
                load_adapter(model,saved['adapter'])
            model.eval()
            for index,r in enumerate(cases):
                guard();x,sr=sf.read(r['noisy_path'],dtype='float32',always_2d=True);assert sr==16000 and x.shape[1]==6
                began=time.perf_counter()
                if system['kind']=='noisy':y=x[:,4].copy()
                else:
                    tx=torch.from_numpy(np.ascontiguousarray(x.T))[None].cuda()
                    with torch.inference_mode():
                        pred,data=model(tx);y=rt['reconstruct_waveform'](model,icirm,tx,pred.to(torch.complex64),x.shape[0])[0].cpu().numpy()
                assert np.isfinite(y).all();seconds=time.perf_counter()-began;path=folder/(r['sample_id']+'.wav');sf.write(path,y,16000,subtype='FLOAT')
                saved_y,_=sf.read(path,dtype='float32');assert np.array_equal(y,saved_y)
                row={**r,'system':name,'output_path':str(path),'output_sha256':sha(path),'inference_seconds':seconds,'audio_seconds':len(y)/16000}
                pending.add(pool.submit(score_signal,row))
                if len(pending)>=8:collect(True)
                if (index+1)%80==0:status(stage='signal',system=name,processed_in_system=index+1,total_per_system=len(cases));print(json.dumps({'stage':'signal','system':name,'processed':index+1}),flush=True)
            while pending:collect(True)
    rt['verify_upstream'](repo,cp)
    assert len(completed)==len(cases)*len(systems);dump(EVAL/'SIGNAL_COMPLETE.json',{'rows':len(completed),'statuses':dict(Counter(r['status'] for r in completed)),'finished_utc':now(),'elapsed_seconds':time.perf_counter()-start})
    report=report_path or ROOT/'docs/G4B_冻结模型信号评估报告.md';lines=['# '+report_title,'',f'数据与前一阶段快照保存后推理。本评估阶段没有训练或调参。每系统{len(cases)}例，全部使用同一清单、FP32推理，保存FLOAT WAV重读计分。','','|系统|条件|有效/总数|SI-SDR dB|PESQ|STOI|','|---|---|---:|---:|---:|---:|']
    for s in systems:
        for condition in sorted({r['condition'] for r in cases}):
            items=[r for r in completed if r['system']==s['system'] and r['condition']==condition];good=[r for r in items if r['status']=='ok']
            vals=[np.mean([r['metrics'][k] for r in good]) for k in ('si_sdr_db','pesq_wb','stoi')]
            lines.append(f'|{s["system"]}|{condition}|{len(good)}/{len(items)}|{vals[0]:.3f}|{vals[1]:.3f}|{vals[2]:.4f}|')
    lines+=['','对应空间反事实组的参考一致性由数据协议和最终审计核查；这些指标不能证明词语或音素完整。下面进入固定识别器阶段，不返回修改模型。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8');archive('signal',report,[EVAL/'SIGNAL_COMPLETE.json',EVAL/'provenance.json',Path(__file__)])
    if any(r['status']!='ok' for r in completed):raise RuntimeError('Signal errors preserved; do not advance silently')

def asr_stage():
    import torch,whisper
    from importlib.metadata import version
    torch.set_num_threads(2);torch.manual_seed(20261106);np.random.seed(20261106);torch.cuda.empty_cache()
    old=ROOT/'artifacts/g3_overnight_v1/evaluation';previous=json.loads((old/'asr_provenance.json').read_text());checkpoint=ROOT/'data/models/whisper/tiny.en.pt'
    assert sha(checkpoint)==previous['checkpoint_sha256']=='d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03';assert version('openai-whisper')=='20250625'
    cache={r['sample_sha256']:r for r in read(old/'asr_cache.jsonl')};options=previous['options'];model=whisper.load_model(str(checkpoint),device='cuda').eval().requires_grad_(False)
    dump(EVAL/'asr_provenance.json',{'started_utc':now(),'model':'Whisper tiny.en','checkpoint_sha256':sha(checkpoint),'options':options,
         'previous_cache_sha256':sha(old/'asr_cache.jsonl'),'cache_initial_entries':len(cache),'reference_text_used_as_prompt':False})
    signals=read(EVAL/'signals.jsonl');paths={p:None for r in signals for p in (r['reference_path'],r['output_path'])}
    with (EVAL/'asr_cache_new.jsonl').open('x',encoding='utf-8') as log:
        for index,path in enumerate(paths):
            guard();a,sr=sf.read(path,dtype='float32');assert sr==16000 and a.ndim==1;key=hashlib.sha256(a.tobytes()).hexdigest();paths[path]=key
            if key not in cache:
                began=time.perf_counter()
                with torch.inference_mode():decoded=model.transcribe(a,**options)
                r={'sample_sha256':key,'hypothesis':decoded['text'],'inference_seconds':time.perf_counter()-began};cache[key]=r;log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
            if (index+1)%100==0:status(stage='asr',paths_completed=index+1,paths_total=len(paths),cache_entries=len(cache))
    with (EVAL/'asr_scores.jsonl').open('x',encoding='utf-8') as log:
        for r in signals:
            h=cache[paths[r['output_path']]]['hypothesis'];rh=cache[paths[r['reference_path']]]['hypothesis']
            record={k:r.get(k) for k in ('system','sample_id','source_id','speaker_id','condition','target_azimuth','noise_azimuth','relative_angle','noise_id','snr_db','transcript')}
            record.update(hypothesis=h,score=score_words(r['transcript'],h),reference_hypothesis=rh,reference_score=score_words(r['transcript'],rh));log.write(json.dumps(record,ensure_ascii=False)+'\n')
    dump(EVAL/'ASR_COMPLETE.json',{'finished_utc':now(),'paths':len(paths),'total_cache_entries':len(cache)})

def aggregate(items,ai):
    return {'cases':len(items),'sources':len({r['source_id'] for r in items}),
        'metrics':{k:float(np.mean([r['metrics'][k] for r in items])) for k in ('si_sdr_db','pesq_wb','stoi')},
        'asr':corpus_score(ai[(r['system'],r['sample_id'])]['score'] for r in items)}

def finish(cases,systems):
    from g4_finish import finish_report
    finish_report(cases,systems)

def main():
    assert (OUT/'stages/preparation/archive_manifest.json').exists();EVAL.mkdir(exist_ok=False)
    cases=read(BASE/'manifest_extended.jsonl');assert len(cases)==472
    plan=json.loads((OUT/'frozen_protocol.json').read_text());systems=plan['systems'];status(stage='starting',training_performed=False)
    assert sha(BASE/'manifest_extended.jsonl')==json.loads((OUT/'rotation_supplement.json').read_text())['extended_manifest_sha256']
    signal_stage(cases,systems);asr_stage();finish(cases,systems);status(stage='complete',report='docs/G4_空间泛化压力测试报告.md')
if __name__=='__main__':
    try:main()
    except Exception as e:status(stage='failed',error_type=type(e).__name__,error=str(e));traceback.print_exc();sys.exit(1)
