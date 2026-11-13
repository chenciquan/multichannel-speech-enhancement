#!/usr/bin/env python3
"""Full-room auxiliary ASR under the already fixed level-control rule."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import gc
import hashlib
import json
import time
import traceback
from collections import Counter
import torch
from g26_prepare import ROOT,OUT,SOURCE,CONTROL,sha,read,dump,now,archive
from g19_scores import fixed_decoder,waveform
from mcse.asr_scoring import NORMALIZATION,score_words
from mcse.asr_level import standardize_level

REC=OUT/'recognition'
FIELDS=('system','sample_id','source_id','speaker_id','transcript','condition','room_id','requested_rt60','target_azimuth',
        'noise_id','noise_field','relative_angle','matched_room_pair','oracle_clean_information','deployable','ASR_regime','normalization_id')


def meta(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})


def normalized_audio(path,digest,expected,config):
    raw,raw_key=waveform(path,digest)
    result,info=standardize_level(raw,config['target_rms'],config['peak_limit'])
    key=hashlib.sha256(result.tobytes()).hexdigest();assert key==expected
    return result,key,raw_key


def seed_cache(engine,plan):
    folder=CONTROL/'recognition'/engine;prior=meta(folder/'provenance.json')
    assert sha(folder/'provenance.json')==plan['seed'][engine]['provenance_sha256']
    assert sha(folder/'observations.jsonl')==plan['seed'][engine]['observations_sha256']
    config=meta(OUT/'frozen_plan.json')['normalization']
    assert prior['input_preprocessing']==config and prior['ASR_regime']=='level_control_v1'
    cache={};observations=read(folder/'observations.jsonl');assert len(observations)==960
    for r in observations:
        assert r['model_id']==prior['model_id'] and r['normalization_id']==config['preprocessing_id']
        key=r['output_samples_sha256']
        if key in cache:assert cache[key]==r['hypothesis'],'Seed observation conflict'
        else:cache[key]=r['hypothesis']
    assert len(cache)==480
    return prior,cache


def backend(engine,rows,plan,deadline):
    prior,cache=seed_cache(engine,plan);folder=REC/engine;folder.mkdir(exist_ok=False)
    previous='recognition_protocol' if engine=='Whisper' else 'whisper'
    assert (OUT/'stages'/previous/'archive_manifest.json').exists()
    config=meta(OUT/'frozen_plan.json')['normalization']
    dump(folder/'provenance.json',{'engine':engine,'model_id':prior['model_id'],'decoding':prior['decoding'],
          'normalization':NORMALIZATION,'reference_prompt':False,'ASR_regime':'level_control_v1',
          'input_preprocessing':config,'seed':plan['seed'][engine],'initial_compatible_cache_entries':480,
          'cache_policy':'Exact normalized PCM bytes, same preprocessing/model/options; G25 seed only, then new observations',
          'plan_sha256':sha(REC/'frozen_plan.json'),'previous_archive_sha256':sha(OUT/'stages'/previous/'archive_manifest.json'),
          'started_utc':now()},True)
    decode=fixed_decoder(engine,prior);new=Counter();ref_paths={}
    with (folder/'cache_new.jsonl').open('x',encoding='utf-8') as clog,(folder/'asr_scores.jsonl').open('x',encoding='utf-8') as log:
        def obtain(a,key,kind):
            if key not in cache:
                assert time.monotonic()<deadline,'G26 decoder deadline exceeded'
                started=time.perf_counter();cache[key]=decode(a)
                clog.write(json.dumps({'sample_sha256':key,'hypothesis':cache[key],'model_id':prior['model_id'],
                    'normalization_id':config['preprocessing_id'],'first_kind':kind,'seconds':time.perf_counter()-started})+'\n')
                clog.flush();new[kind]+=1
            return cache[key]
        for i,r in enumerate(rows,1):
            assert time.monotonic()<deadline
            reference=r['reference_path'];rk=r['normalized_reference_samples_sha256']
            if reference not in ref_paths:
                a,key,_=normalized_audio(reference,r['reference_sha256'],rk,config)
                obtain(a,key,'reference');ref_paths[reference]=key
            assert ref_paths[reference]==rk
            a,key,raw_key=normalized_audio(r['output_path'],r['output_sha256'],r['normalized_output_samples_sha256'],config)
            assert raw_key==r['raw_output_samples_sha256']
            h=obtain(a,key,'output');rh=cache[rk]
            item={f:r[f] for f in FIELDS}
            item.update(hypothesis=h,score=score_words(r['transcript'],h),reference_hypothesis=rh,
                        reference_score=score_words(r['transcript'],rh),output_samples_sha256=key,
                        reference_samples_sha256=rk,raw_output_samples_sha256=raw_key)
            log.write(json.dumps(item,allow_nan=False)+'\n')
            if i%168==0:status(stage='normalized_full_ASR',engine=engine,done=i,total=16488,new_decodes=dict(new))
    marker={'status':'ok','utc':now(),'rows':16488,'reference_paths':len(ref_paths),'new_decodes':dict(new),
            'final_cache_entries':len(cache),'score_sha256':sha(folder/'asr_scores.jsonl'),'cache_sha256':sha(folder/'cache_new.jsonl')}
    dump(folder/'COMPLETE.json',marker,True)
    report=ROOT/'docs'/f'G26_{engine}完整电平辅评分覆盖.md'
    with report.open('x',encoding='utf-8') as f:
        f.write(f'# G26 {engine} 完整电平辅评分覆盖\n\n全部16488条评分完成。仅复用G25相同规范化PCM与模型配置的480个种子缓存，新增解码{dict(new)}。504个参考条件按规范化输入哈希处理，未缓存参考实际解码。原波形与raw-ASR保持不变；尚未根据本表选择方法。\n')
    archive(engine.lower(),report,[folder/'COMPLETE.json',folder/'provenance.json',ROOT/'scripts/g26_asr.py'])
    del decode;gc.collect();torch.cuda.empty_cache()


def main():
    done=meta(OUT/'PREPARATION_COMPLETE.json');data_plan=meta(OUT/'frozen_plan.json')
    assert done['status']=='ok' and sha(OUT/'manifest.jsonl')==done['manifest_sha256']
    assert sha(ROOT/'src/mcse/asr_level.py')==data_plan['normalization']['code_sha256']
    for n,h in data_plan['hashes'].items():assert sha(ROOT/n)==h
    for directory in (OUT/'stages').iterdir():
        for r in meta(directory/'archive_manifest.json')['files']:
            assert sha(directory/r['snapshot'])==r['sha256'] and sha(ROOT/r['source'])==r['sha256']
    rows=read(OUT/'manifest.jsonl');assert len(rows)==len({(r['system'],r['sample_id']) for r in rows})==16488
    REC.mkdir(exist_ok=False)
    names=['scripts/g26_asr.py','scripts/g26_finish.py','scripts/g26_accept.py','scripts/g26_prepare.py',
           'src/mcse/asr_level.py','scripts/g19_scores.py','scripts/g8_common.py','src/mcse/asr_scoring.py','src/mcse/ctc.py']
    plan={'frozen_utc':now(),'max_seconds':7200,'manifest_sha256':sha(OUT/'manifest.jsonl'),
          'hashes':{n:sha(ROOT/n) for n in names},
          'seed':{e:{'provenance_sha256':sha(CONTROL/'recognition'/e/'provenance.json'),
                     'observations_sha256':sha(CONTROL/'recognition'/e/'observations.jsonl')} for e in ('Whisper','CTC')},
          'raw_comparators':{e:sha(SOURCE/'recognition'/e/'asr_scores.jsonl') for e in ('Whisper','CTC')}}
    dump(REC/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G26C_完整电平双识别器执行协议.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G26C 完整电平双识别器执行协议\n\n两个固定后端各16488记录。原波形在内存重建规范化输入，逐例核对预先冻结PCM哈希；原评分不改。只使用G25相同预处理/模型的规范化种子，之后按精确规范化PCM哈希缓存；不把raw缓存或raw成绩混入此条件。\n\n两个解码后端共用7200秒上限。先完成Whisper归档，再CTC归档，最后比较原始/规范化模型排序、房间差异和完整S/D/I。全部原种子、场型、噪声和干净条件保留，不挑最优分数或调电平。\n')
    archive('recognition_protocol',report,[REC/'frozen_plan.json']+[ROOT/n for n in names])
    torch.set_num_threads(2);deadline=time.monotonic()+plan['max_seconds']-120
    for e in ('Whisper','CTC'):backend(e,rows,plan,deadline)
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    from g26_finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception as e:status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:(OUT/'keep_awake_stop').write_text('G26 normalized full-room ASR ended\n')
    try:
        for _ in range(24):
            if (OUT/'power_helper.json').exists() and meta(OUT/'power_helper.json')['status']=='released':break
            time.sleep(2)
        assert meta(OUT/'power_helper.json')['status']=='released'
        from g26_accept import main as accept
        accept();print('G26 full auxiliary ASR stage accepted',flush=True)
    except Exception as e:status(stage='acceptance_failed',error=repr(e));traceback.print_exc();raise
