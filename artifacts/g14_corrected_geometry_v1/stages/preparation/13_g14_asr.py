#!/usr/bin/env python3
"""Two fixed ASR backends with validated sample caches and stage archives."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from g14_common import ROOT,OUT,now,sha,read,dump,status,archive
from mcse.asr_scoring import score_words,corpus_score,NORMALIZATION
from mcse.ctc import greedy_ctc_tokens


def main():
    assert (OUT/'stages/beamforming/archive_manifest.json').exists();plan=json.loads((OUT/'frozen_plan.json').read_text())
    signals=read(OUT/'evaluation/combined_signals.jsonl');assert len(signals)==5040;torch.set_num_threads(2);started=time.monotonic()
    for engine in ('Whisper','CTC'):
        folder=OUT/'evaluation'/engine;folder.mkdir(exist_ok=False);cache={};conflicts=[];sources=[]
        if engine=='Whisper':
            import whisper
            from importlib.metadata import version
            torch.manual_seed(20261106);np.random.seed(20261106)
            assert version('openai-whisper')=='20250625';base=ROOT/'artifacts/g3_overnight_v1/evaluation'
            original=json.loads((base/'asr_provenance.json').read_text());options=original['options'];checkpoint=ROOT/'data/models/whisper/tiny.en.pt'
            assert sha(checkpoint)==original['checkpoint_sha256']=='d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03'
            for directory,filename in [(base,'asr_cache.jsonl'),(ROOT/'artifacts/g5_rotation_v1/evaluation','asr_cache_new.jsonl'),(ROOT/'artifacts/g7_unseen_noise_v1/evaluation','asr_cache_new.jsonl')]:
                prov=json.loads((directory/'asr_provenance.json').read_text());assert prov['checkpoint_sha256']==original['checkpoint_sha256'] and prov['options']==options
                file=directory/filename;sources.append({'path':str(file.relative_to(ROOT)),'sha256':sha(file)})
                for r in read(file):
                    key=r['sample_sha256']
                    if key in cache and cache[key]['hypothesis']!=r['hypothesis']:conflicts.append({'sample_sha256':key,'kept':cache[key]['hypothesis'],'later':r['hypothesis']})
                    cache.setdefault(key,r)
            for directory in (ROOT/'artifacts/g9_beamforming_v1/evaluation/Whisper',ROOT/'artifacts/g10_fresh_voice_v1/evaluation/Whisper'):
                prior=json.loads((directory/'provenance.json').read_text())
                assert prior['model_id']==sha(checkpoint) and prior['decoding']['options']==options
                file=directory/'cache_new.jsonl';sources.append({'path':str(file.relative_to(ROOT)),'sha256':sha(file)})
                for r in read(file):
                    key=r['sample_sha256']
                    if key in cache and cache[key]['hypothesis']!=r['hypothesis']:conflicts.append({'sample_sha256':key,'kept':cache[key]['hypothesis'],'later':r['hypothesis']})
                    cache.setdefault(key,r)
            model=whisper.load_model(str(checkpoint),device='cuda').eval().requires_grad_(False)
            def decode(a):
                with torch.inference_mode():return model.transcribe(a,**options)['text']
            model_id=sha(checkpoint);decode_config={'options':options,'model_sha256':model_id}
        else:
            import g8_common
            torch.manual_seed(20261112);np.random.seed(20261112)
            previous=ROOT/'artifacts/g8_cross_asr_v1';model_id=sha(previous/'asr_provenance.json')
            manifest=json.loads((previous/'stages/asset/archive_manifest.json').read_text());entry=next(r for r in manifest['files'] if r['source']=='src/mcse/ctc.py')
            assert sha(ROOT/'src/mcse/ctc.py')==entry['sha256']
            file=previous/'evaluation/ctc_cache.jsonl';sources.append({'path':str(file.relative_to(ROOT)),'sha256':sha(file)})
            for r in read(file):
                assert r['recognizer_id']==model_id
                key=r['sample_sha256']
                if key in cache:assert cache[key]['hypothesis']==r['hypothesis']
                cache[key]=r
            for directory in (ROOT/'artifacts/g9_beamforming_v1/evaluation/CTC',ROOT/'artifacts/g10_fresh_voice_v1/evaluation/CTC'):
                assert json.loads((directory/'provenance.json').read_text())['model_id']==model_id
                file=directory/'cache_new.jsonl';sources.append({'path':str(file.relative_to(ROOT)),'sha256':sha(file)})
                for r in read(file):
                    assert r['model_id']==model_id
                    key=r['sample_sha256']
                    if key in cache:assert cache[key]['hypothesis']==r['hypothesis']
                    cache[key]=r
            model,labels=g8_common.model_load()
            def decode(a):
                with torch.inference_mode():emission,_=model(torch.from_numpy(np.ascontiguousarray(a))[None].cuda())
                assert torch.isfinite(emission).all() and emission.shape[-1]==29
                return greedy_ctc_tokens(emission[0].argmax(-1).cpu().tolist(),labels)
            decode_config={'previous_recognizer_id':model_id,'batch':1,'FP32':True,'normalization':False,'resampling':False,'external_LM':False,'greedy_CTC':True}
        provenance={'started_utc':now(),'engine':engine,'model_id':model_id,'decoding':decode_config,'source_caches':sources,
          'source_precedence':'Whisper G3 then G5 then G7 then G9 then G10; first observation retained regardless of score; CTC G8 then G9 then G10 exact recognizer identity',
          'cache_hypothesis_conflicts':conflicts,'normalization':NORMALIZATION,'reference_prompt':False,'signal_archive_sha256':sha(OUT/'stages/beamforming/archive_manifest.json')}
        dump(folder/'provenance.json',provenance,True);paths={p:None for r in signals for p in (r['reference_path'],r['output_path'])};new=0
        with (folder/'cache_new.jsonl').open('x') as log:
            for index,path in enumerate(paths):
                if time.monotonic()-started>plan['max_ASR_seconds']-240:raise TimeoutError('Dual-ASR reserve')
                a,sr=sf.read(path,dtype='float32');assert sr==16000 and a.ndim==1 and np.isfinite(a).all()
                key=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest();paths[path]=key
                if key not in cache:
                    began=time.perf_counter();h=decode(a);r={'sample_sha256':key,'hypothesis':h,'seconds':time.perf_counter()-began,'model_id':model_id};cache[key]=r;new+=1
                    log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
                if (index+1)%100==0:status(stage='asr',engine=engine,paths_done=index+1,paths_total=len(paths),new_decodes=new)
        scored=[]
        with (folder/'asr_scores.jsonl').open('x',encoding='utf-8') as log:
            for r in signals:
                h=cache[paths[r['output_path']]]['hypothesis'];rh=cache[paths[r['reference_path']]]['hypothesis']
                row={k:r.get(k) for k in ('dataset','system','sample_id','source_id','speaker_id','condition','target_azimuth','relative_angle','noise_id','snr_db','transcript','geometry_protocol','geometry_pair','actual_relative_azimuth','noise_field')}
                row.update(hypothesis=h,score=score_words(r['transcript'],h),reference_hypothesis=rh,reference_score=score_words(r['transcript'],rh),
                  output_samples_sha256=paths[r['output_path']],reference_samples_sha256=paths[r['reference_path']]);scored.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n')
        dump(folder/'COMPLETE.json',{'completed_utc':now(),'rows':len(scored),'unique_paths':len(paths),'new_decodes':new,'cache_entries':len(cache)},True)
        if engine=='Whisper':
            lines=['# G14D Whisper正确几何配对初评','','信号报告归档后才识别，沿用固定模型/选项/英语归一化，无参考提示。缓存按G3→G5→G7→G9→G10固定优先级使用，冲突全部保留，不按WER选择。下表仅完整覆盖汇总，混有干净/两种几何协议/方位，不用于宣称全条件优势。','','|系统|记录|微平均WER|','|---|---:|---:|']
            for system in sorted({r['system'] for r in signals}):
                a=[r for r in scored if r['system']==system];lines.append(f'|{system}|{len(a)}|{corpus_score(r["score"] for r in a)["wer"]:.1%}|')
            lines+=['',f'新增识别{new}条，缓存冲突{len(conflicts)}条；保存全部逐例S/D/I，WER不截断。本轮所有系统均不使用真值掩码或方位。下一步在独立CTC后端交叉核查，然后按几何协议、方向、噪声及全部适配器种子分条件讨论。','', 'UTC：'+now()]
            report=ROOT/'docs/G14D_Whisper正确几何配对初评.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');archive('whisper',report,[folder/'provenance.json',folder/'COMPLETE.json',ROOT/'scripts/g14_asr.py'])
        del model;gc.collect();torch.cuda.empty_cache()
    from g14_finish import finish
    finish()


if __name__=='__main__':main()
