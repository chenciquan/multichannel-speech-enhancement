#!/usr/bin/env python3
"""Locked held-out evaluation after checkpoint selection; no training or tuning."""
import argparse,hashlib,importlib.metadata,json,runpy,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.adaptation import attach_lora,adapter_state,load_adapter
from mcse.evaluation import MetricEvaluator,waveform_diagnostics
from mcse.spatial_noise import highpass40
from mcse.simulation import render_plane_wave,geometric_delays
from mcse.asr_scoring import score_words,corpus_score,NORMALIZATION,ALIGNMENT_TIE_POLICY
OUT=ROOT/'artifacts/g3_overnight_v1';BASE=ROOT/'data/g3_v1/dataset'
DEADLINE=datetime.fromisoformat('2026-11-11T21:54:15+00:00').timestamp()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def guard():
    if time.time()>DEADLINE-120:raise TimeoutError('Hard deadline reserve reached; preserve partial outputs')
def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def signal_stage():
    assert (OUT/'READY_FOR_TEST.json').exists(),'Primary and registered extension selection must be complete before test opens'
    selection_path=OUT/'selected_checkpoints_extended.json'
    selected=json.loads(selection_path.read_text());assert not selected['final_test_opened']
    target=OUT/'evaluation';target.mkdir(exist_ok=False);start=time.perf_counter()
    dump(target/'protocol.json',{'started_utc':now(),'selection_manifest_sha256':sha(selection_path),'selection_manifest':selection_path.name,
         'selection_before_test':True,'inference_dtype':'float32','no_waveform_peak_normalization':True,'no_resampling':True,
         'scoring':'Read saved FLOAT WAVs; SI-SDR, wideband PESQ, standard STOI; no implicit alignment',
         'independent_test_sources':30,'test_mixtures':360,'test_sources_used_to_tune':False,'test_speakers':4})
    torch.set_num_threads(2);torch.manual_seed(20261112)
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g3_eval_runtime');repo=ROOT/'third_party/McNet';cp=repo/rt['OFFLINE_CHECKPOINT']
    rt['verify_upstream'](repo,cp);model,icirm,_=rt['load_model'](repo,cp,torch.device('cuda'));attach_lora(model);zero=adapter_state(model)
    cases=rows(BASE/'test/manifest.jsonl');assert len(cases)==360
    evaluator=MetricEvaluator();assert all(r['status']=='available' for r in evaluator.backend_info.values())
    systems=[{'system':'noisy'},{'system':'hpf40'},{'system':'mcnet'}]+selected['systems']
    dump(target/'systems.json',systems);dump(target/'metric_backends.json',evaluator.backend_info)
    # Independent clean-input controls use exactly the same nominal speech field,
    # not the noisy mixture and not mono replicated across microphones.
    source={r['source_id']:r for r in rows(ROOT/'data/g3_v1/sources/speech_manifest.jsonl') if r['split']=='test'}
    geometry=json.loads((ROOT/'configs/g1_geometry.json').read_text());cleans=[]
    for sid,r in source.items():
        mono,sr=sf.read(ROOT/'data/g3_v1/sources'/r['path'],dtype='float64');assert sr==16000
        x=render_plane_wave(mono,geometric_delays(geometry['microphone_positions_m'],geometry['target_direction_unit'])).astype('float32')
        folder=target/'clean_input';folder.mkdir(exist_ok=True);xp=folder/(sid+'_6ch.wav');sp=folder/(sid+'_ref.wav')
        sf.write(xp,x,16000,subtype='FLOAT');sf.write(sp,x[:,4],16000,subtype='FLOAT')
        cleans.append({'sample_id':'clean_'+sid,'source_id':sid,'speaker_id':r['speaker_id'],'split':'clean_test',
                       'noisy_path':str(xp),'reference_path':str(sp),'transcript':r['transcript'],
                       'noise_id':'none','snr_db':None,'noise_field':'clean','noise_azimuth':None})
    evalcases=[]
    for r in cases:
        r=dict(r)
        for key in ('noisy_path','reference_path'):
            path=BASE/'test'/r[key];assert sha(path)==r[key.replace('_path','_sha256')];r[key]=str(path)
        evalcases.append(r)
    evalcases+=cleans
    with (target/'signals.jsonl').open('x',encoding='utf-8') as output:
        for system in systems:
            name=system['system'];folder=target/'audio'/name;folder.mkdir(parents=True)
            if name=='mcnet':load_adapter(model,zero)
            elif 'path' in system:
                path=ROOT/system['path'];saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['step']==system['step'];load_adapter(model,saved['adapter']);system['checkpoint_sha256']=sha(path)
            model.eval();system_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            for i,r in enumerate(evalcases):
                guard();rec={k:r.get(k) for k in ('sample_id','source_id','speaker_id','split','noise_id','snr_db','noise_field','noise_azimuth','transcript')}
                rec.update(system=name,status='ok',reference_path=r['reference_path'],input_path=r['noisy_path'])
                try:
                    x,sr=sf.read(r['noisy_path'],dtype='float32',always_2d=True);s,ssr=sf.read(r['reference_path'],dtype='float32');assert sr==ssr==16000
                    began=time.perf_counter()
                    if name=='noisy':y=x[:,4].copy()
                    elif name=='hpf40':y=highpass40(x[:,4]).astype('float32')
                    else:
                        tx=torch.from_numpy(np.ascontiguousarray(x.T))[None].cuda()
                        with torch.inference_mode():
                            prediction,data=model(tx);y=rt['reconstruct_waveform'](model,icirm,tx,prediction.to(torch.complex64),x.shape[0])[0].cpu().numpy()
                    assert y.shape==s.shape and np.isfinite(y).all()
                    inference=time.perf_counter()-began;path=folder/(r['sample_id']+'.wav');sf.write(path,y,16000,subtype='FLOAT')
                    saved_audio,sr=sf.read(path,dtype='float32');assert np.array_equal(saved_audio,y)
                    rec.update(output_path=str(path),output_sha256=sha(path),output_sample_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                               inference_seconds=inference,audio_seconds=len(y)/16000,metrics=evaluator.evaluate(s,saved_audio),
                               waveform_diagnostics=waveform_diagnostics(s,saved_audio))
                    if any(v['status']!='ok' for v in rec['metrics']['status'].values()):rec['status']='metric_error'
                except Exception as e:rec.update(status='error',error_type=type(e).__name__,error=str(e))
                output.write(json.dumps(rec,ensure_ascii=False)+'\n');output.flush()
                if (i+1)%60==0:
                    dump(target/'status.json',{'stage':'signal','system':name,'completed_in_system':i+1,'total_per_system':len(evalcases),'time_utc':now()})
                    print(json.dumps({'stage':'signal','system':name,'completed':i+1}),flush=True)
            system.update(signal_elapsed_seconds=time.perf_counter()-system_start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    rt['verify_upstream'](repo,cp)
    dump(target/'systems.json',systems);dump(target/'SIGNAL_COMPLETE.json',{'finished_utc':now(),'elapsed_seconds':time.perf_counter()-start,'systems':len(systems),'cases_per_system':len(evalcases)})

def asr_stage():
    import whisper
    target=OUT/'evaluation';assert (target/'SIGNAL_COMPLETE.json').exists()
    checkpoint=ROOT/'data/models/whisper/tiny.en.pt';assert sha(checkpoint)=='d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03'
    assert importlib.metadata.version('openai-whisper')=='20250625'
    torch.set_num_threads(2);torch.manual_seed(20261106);np.random.seed(20261106)
    model=whisper.load_model(str(checkpoint),device='cuda').eval().requires_grad_(False)
    options={'language':'en','task':'transcribe','temperature':0.,'condition_on_previous_text':False,'initial_prompt':None,
             'beam_size':1,'word_timestamps':False,'without_timestamps':True,'compression_ratio_threshold':2.4,
             'logprob_threshold':-1.,'no_speech_threshold':.6,'fp16':True,'verbose':None}
    dump(target/'asr_provenance.json',{'started_utc':now(),'model':'Whisper tiny.en','version':importlib.metadata.version('openai-whisper'),
         'checkpoint_sha256':sha(checkpoint),'options':options,'references_used_as_prompts':False,
         'normalization':NORMALIZATION,'alignment_tie_policy':ALIGNMENT_TIE_POLICY,'cache_key':'SHA256 of exact float32 mono samples at 16kHz',
         'limitations':['one recognizer, not a human word-deletion diagnosis','possible unknown pretraining overlap','test not used for checkpoint selection']})
    cache={};cp=target/'asr_cache.jsonl'
    if cp.exists():cache={r['sample_sha256']:r for r in rows(cp)}
    source_rows=rows(target/'signals.jsonl');tasks={}
    for r in source_rows:
        if r['status']!='ok':continue
        for path in (r['reference_path'],r['output_path']):tasks[path]=None
    with cp.open('a',encoding='utf-8') as log:
        for i,path in enumerate(tasks):
            guard();audio,sr=sf.read(path,dtype='float32');assert sr==16000 and audio.ndim==1
            key=hashlib.sha256(audio.tobytes()).hexdigest();tasks[path]=key
            if key not in cache:
                began=time.perf_counter()
                with torch.inference_mode():decoded=model.transcribe(audio,**options)
                rec={'sample_sha256':key,'hypothesis':decoded['text'],'inference_seconds':time.perf_counter()-began,
                     'segments':[{k:s.get(k) for k in ('text','avg_logprob','no_speech_prob','compression_ratio')} for s in decoded.get('segments',[])]}
                cache[key]=rec;log.write(json.dumps(rec,ensure_ascii=False)+'\n');log.flush()
            if (i+1)%100==0:
                dump(target/'status.json',{'stage':'asr','paths_completed':i+1,'paths_total':len(tasks),'unique_decodes':len(cache),'time_utc':now()})
                print(json.dumps({'stage':'asr','paths_completed':i+1,'paths_total':len(tasks),'unique_decodes':len(cache)}),flush=True)
    with (target/'asr_scores.jsonl').open('x',encoding='utf-8') as output:
        for r in source_rows:
            if r['status']!='ok':continue
            prediction=cache[tasks[r['output_path']]];reference=cache[tasks[r['reference_path']]]
            record={k:r[k] for k in ('system','sample_id','source_id','speaker_id','split','noise_id','snr_db','noise_field','noise_azimuth','transcript')}
            record.update(hypothesis=prediction['hypothesis'],score=score_words(r['transcript'],prediction['hypothesis']),
                          reference_hypothesis=reference['hypothesis'],reference_score=score_words(r['transcript'],reference['hypothesis']))
            output.write(json.dumps(record,ensure_ascii=False)+'\n')
    dump(target/'ASR_COMPLETE.json',{'finished_utc':now(),'unique_decodes':len(cache),'paths':len(tasks)})
    for script in ('g3_summarize.py','g3_audit.py','g3_plot.py','g3_report.py'):
        guard();subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/g3_stage_archive.py'),'--stage','completion'],cwd=ROOT,check=True)
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'stage':'G3','completed_utc':now(),'report':'docs/G3_五小时实验验收报告.md'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['signal','asr'],required=True);a=p.parse_args()
    try:signal_stage() if a.stage=='signal' else asr_stage()
    except Exception:
        traceback.print_exc();sys.exit(1)
