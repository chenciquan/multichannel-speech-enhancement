#!/usr/bin/env python3
"""Second-recognizer scoring of fixed saved waves, with no SE forward/training."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,time
import numpy as np
import soundfile as sf
import torch
from g8_common import ROOT,OUT,now,sha,read,dump,status,archive,model_load
from mcse.ctc import greedy_ctc_tokens
from mcse.asr_scoring import score_words,NORMALIZATION,ALIGNMENT_TIE_POLICY


def main():
    assert (OUT/'ASSET_STAGE_COMPLETE.json').exists();plan=json.loads((OUT/'frozen_plan.json').read_text())
    asset_archive=OUT/'stages/asset/archive_manifest.json';assert asset_archive.exists()
    start=time.monotonic();records=[];paths={};torch.set_num_threads(2);torch.manual_seed(20261112)
    for dataset,base in plan['datasets'].items():
        folder=ROOT/base['folder'];assert sha(folder/'signals.jsonl')==base['signals_sha256'] and sha(folder/'asr_scores.jsonl')==base['whisper_sha256']
        rows=read(folder/'signals.jsonl');assert len(rows)==base['expected_rows']
        for r in rows:
            r={**r,'dataset':dataset};records.append(r)
            for p,h in ((r['output_path'],r['output_sha256']),(r['reference_path'],r['reference_sha256'])):
                if p in paths and paths[p]['file_sha256']!=h:raise ValueError('Conflicting frozen file hashes')
                paths[p]={'file_sha256':h}
    provenance={'started_utc':now(),'asset_archive_sha256':sha(asset_archive),'asset_provenance_sha256':sha(OUT/'asset_provenance.json'),
       'frozen_plan_sha256':sha(OUT/'frozen_plan.json'),'normalization':NORMALIZATION,'alignment_tie_policy':ALIGNMENT_TIE_POLICY,
       'CTC':'greedy argmax -> consecutive collapse -> remove blank0 -> | to space','batch':1,'dtype':'float32',
       'SE_training_or_inference_performed':False,'reference_text_used_as_prompt':False,'no_external_LM':True,'sample_rate':16000,
       'waveform_normalization':False,'resampling':False,'cache_key':'SHA256 of recognizer-provenance ID + raw contiguous mono16k float32 sample bytes hash'}
    dump(OUT/'asr_provenance.json',provenance,True);recognizer_id=sha(OUT/'asr_provenance.json')
    model,labels=model_load();cache={};OUT.joinpath('evaluation').mkdir(exist_ok=False)
    with (OUT/'evaluation/ctc_cache.jsonl').open('x',encoding='utf-8') as log:
        for index,(path,info) in enumerate(paths.items()):
            if time.monotonic()-start>plan['evaluation_timeout_seconds']-180:raise TimeoutError('CTC scoring reserve')
            if sha(path)!=info['file_sha256']:raise ValueError('Saved waveform hash mismatch')
            a,sr=sf.read(path,dtype='float32');assert sr==16000 and a.ndim==1 and np.isfinite(a).all() and 1<=len(a)<=64000
            sample_sha=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest();key=hashlib.sha256((recognizer_id+sample_sha).encode()).hexdigest()
            info.update(sample_sha256=sample_sha,cache_key=key)
            if key not in cache:
                x=torch.from_numpy(np.ascontiguousarray(a))[None].cuda();began=time.perf_counter()
                with torch.inference_mode():emission,_=model(x)
                assert emission.shape[-1]==len(labels)==29 and torch.isfinite(emission).all()
                ids=emission[0].argmax(-1).cpu().tolist();text=greedy_ctc_tokens(ids,labels)
                cache[key]={'cache_key':key,'recognizer_id':recognizer_id,'sample_sha256':sample_sha,'samples':len(a),'hypothesis':text,'seconds':time.perf_counter()-began}
                log.write(json.dumps(cache[key],ensure_ascii=False)+'\n');log.flush()
            if (index+1)%200==0:status(stage='CTC',paths_done=index+1,paths_total=len(paths),unique_decodes=len(cache))
    dump(OUT/'evaluation/path_hash_index.json',paths,True)
    with (OUT/'evaluation/asr_scores.jsonl').open('x',encoding='utf-8') as log:
        for r in records:
            h=cache[paths[r['output_path']]['cache_key']]['hypothesis'];rh=cache[paths[r['reference_path']]['cache_key']]['hypothesis']
            row={k:r.get(k) for k in ('dataset','system','sample_id','source_id','speaker_id','condition','target_azimuth','relative_angle','noise_id','snr_db','transcript')}
            row.update(hypothesis=h,score=score_words(r['transcript'],h),reference_hypothesis=rh,reference_score=score_words(r['transcript'],rh),
              output_samples_sha256=paths[r['output_path']]['sample_sha256'],reference_samples_sha256=paths[r['reference_path']]['sample_sha256'])
            log.write(json.dumps(row,ensure_ascii=False)+'\n')
    completion={'completed_utc':now(),'records':len(records),'unique_paths':len(paths),'unique_decodes':len(cache),'elapsed_seconds':time.monotonic()-start,
      'all_file_hashes_verified':True,'max_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'max_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
    dump(OUT/'evaluation/CTC_COMPLETE.json',completion,True)
    from g8_finish import finish
    finish()


if __name__=='__main__':main()
