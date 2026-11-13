#!/usr/bin/env python3
"""Evaluate predeclared scalar-gain controls using fixed cached recognizers."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import gc
import json
import time
import traceback
from functools import lru_cache
import numpy as np
import torch
from g23_prepare import ROOT,OUT,G22,sha,dump,read,now,archive
from g19_scores import fixed_decoder,load_cache,waveform
from mcse.asr_scoring import score_words,NORMALIZATION

REC=OUT/'recognition'
FIELDS=('base_system','sample_id','gain_db','gain','source_id','speaker_id','transcript','room_id',
        'requested_rt60','target_azimuth','noise_id','noise_field','relative_angle','control_only','oracle_clean_information')


def meta(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})


def validate_inputs():
    preparation=meta(OUT/'PREPARATION_COMPLETE.json');plan=meta(OUT/'frozen_plan.json')
    assert preparation['status']=='ok' and sha(OUT/'manifest.jsonl')==preparation['manifest_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    for directory in (OUT/'stages').iterdir():
        for r in meta(directory/'archive_manifest.json')['files']:
            assert sha(directory/r['snapshot'])==r['sha256'] and sha(ROOT/r['source'])==r['sha256']
    rows=read(OUT/'manifest.jsonl');cases={r['sample_id'] for r in rows}
    keys={(r['base_system'],r['sample_id'],r['gain_db']) for r in rows}
    assert len(rows)==len(keys)==1440 and len(cases)==48
    assert keys=={(s,c,g) for s in plan['base_systems'] for c in cases for g in plan['gain_dB']}
    @lru_cache(maxsize=4)
    def original(path,h):return waveform(path,h)[0]
    pcm={};ref_pcm={};maximum=0.
    for r in rows:
        base=original(r['base_path'],r['base_sha256'])
        wave,key=waveform(r['output_path'],r['output_sha256']);pcm[r['base_system'],r['sample_id'],r['gain_db']]=key
        assert np.array_equal(wave,(base.astype(float)*10**(r['gain_db']/20)).astype('float32'))
        if r['gain_db']==0:assert r['output_path']==r['base_path'] and r['output_sha256']==r['base_sha256']
        error=float(np.linalg.norm(wave.astype(float)/r['gain']-base)/np.linalg.norm(base.astype(float)))
        assert error<=plan['relative_roundtrip_error_max'];maximum=max(maximum,error)
        if r['sample_id'] not in ref_pcm:
            _,ref_pcm[r['sample_id']]=waveform(r['reference_path'],r['reference_sha256'])
    result={'status':'ok','utc':now(),'records':1440,'cases':48,'all_scaled_samples_exact_reconstruction':True,
            'max_relative_roundtrip_error':maximum,'manifest_sha256':sha(OUT/'manifest.jsonl')}
    dump(OUT/'INPUT_ACCEPTANCE_AUDIT.json',result,True)
    report=ROOT/'docs/G23_增益控制输入独立核验.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G23 增益控制输入独立核验\n\n1440个预定键完整，所有文件哈希通过。960条缩放波形逐样本等于原波形乘固定增益并转float32；480个0dB记录直接引用原文件。独立复算缩放往返相对误差通过预定上限。未运行ASR或模型训练。\n')
    archive('input_acceptance',report,[OUT/'INPUT_ACCEPTANCE_AUDIT.json'])
    return rows,pcm,ref_pcm


def backend(engine,rows,pcm,ref_pcm,plan,deadline):
    prior_folder=G22/'recognition'/engine;prior=meta(prior_folder/'provenance.json')
    assert sha(prior_folder/'provenance.json')==plan['parent'][engine]['provenance_sha256']
    assert sha(prior_folder/'cache_new.jsonl')==plan['parent'][engine]['cache_sha256']
    assert sha(prior_folder/'asr_scores.jsonl')==plan['parent'][engine]['score_sha256']
    cache,conflicts,sources=load_cache(engine,prior,prior_folder);assert not conflicts
    parent={(r['system'],r['sample_id']):r for r in read(prior_folder/'asr_scores.jsonl')}
    preceding='recognition_protocol' if engine=='Whisper' else 'whisper'
    assert (OUT/'stages'/preceding/'archive_manifest.json').exists()
    folder=REC/engine;folder.mkdir(exist_ok=False)
    dump(folder/'provenance.json',{'engine':engine,'model_id':prior['model_id'],'decoding':prior['decoding'],
          'normalization':NORMALIZATION,'reference_prompt':False,'source_caches':sources,
          'previous_archive_sha256':sha(OUT/'stages'/preceding/'archive_manifest.json'),
          'plan_sha256':sha(REC/'frozen_plan.json'),'control_only':True,'started_utc':now()},True)
    # Prove every 0dB control was already observed before any new inference.
    zero=0
    for r in rows:
        if r['gain_db']!=0:continue
        key=pcm[r['base_system'],r['sample_id'],0];p=parent[('noisy' if r['base_system']=='clean_reference' else r['base_system'],r['sample_id'])]
        expected_key=p['reference_samples_sha256'] if r['base_system']=='clean_reference' else p['output_samples_sha256']
        expected_hyp=p['reference_hypothesis'] if r['base_system']=='clean_reference' else p['hypothesis']
        assert key==expected_key and cache[key]==expected_hyp
        assert ref_pcm[r['sample_id']]==p['reference_samples_sha256'] and ref_pcm[r['sample_id']] in cache
        zero+=1
    assert zero==480
    decode=fixed_decoder(engine,prior);new=0
    with (folder/'cache_new.jsonl').open('x',encoding='utf-8') as clog,(folder/'asr_scores.jsonl').open('x',encoding='utf-8') as log:
        for i,r in enumerate(rows,1):
            assert time.monotonic()<deadline,'G23 ASR deadline exceeded'
            key=pcm[r['base_system'],r['sample_id'],r['gain_db']];rk=ref_pcm[r['sample_id']]
            if key not in cache:
                assert r['gain_db']!=0,'Unexpected baseline cache miss'
                samples,actual=waveform(r['output_path'],r['output_sha256']);assert actual==key
                started=time.perf_counter();cache[key]=decode(samples)
                clog.write(json.dumps({'sample_sha256':key,'hypothesis':cache[key],'model_id':prior['model_id'],
                                       'gain_db_first_seen':r['gain_db'],'seconds':time.perf_counter()-started})+'\n');clog.flush();new+=1
            item={k:r[k] for k in FIELDS}
            item.update(output_samples_sha256=key,reference_samples_sha256=rk,hypothesis=cache[key],
                        score=score_words(r['transcript'],cache[key]),reference_hypothesis=cache[rk],
                        reference_score=score_words(r['transcript'],cache[rk]),reference_waveform_control=r['base_system']=='clean_reference')
            log.write(json.dumps(item,allow_nan=False)+'\n')
            if i%120==0:status(stage='asr',engine=engine,done=i,total=1440,new_decodes=new)
    marker={'status':'ok','utc':now(),'rows':1440,'zero_db_cache_hits':zero,'new_decodes':new,
            'scores_sha256':sha(folder/'asr_scores.jsonl'),'cache_sha256':sha(folder/'cache_new.jsonl')}
    dump(folder/'COMPLETE.json',marker,True)
    report=ROOT/'docs'/f'G23_{engine}控制识别覆盖.md'
    with report.open('x',encoding='utf-8') as f:
        f.write(f'# G23 {engine} 控制识别覆盖\n\n1440条预定增益控制评分完成。480个0dB基准的PCM哈希和假设文本均与G22原记录一致，全部命中缓存；新增解码{new}条，仅来自非零增益。参考识别复用相同PCM哈希缓存，未提供文本提示。该记录只证明覆盖，增益差值另行配对汇总。\n')
    archive(engine.lower(),report,[folder/'COMPLETE.json',folder/'provenance.json',ROOT/'scripts/g23_asr.py'])
    del decode;gc.collect();torch.cuda.empty_cache()


def main():
    rows,pcm,refs=validate_inputs();REC.mkdir(exist_ok=False)
    names=['scripts/g23_asr.py','scripts/g23_finish.py','scripts/g23_accept.py','scripts/g23_prepare.py',
           'scripts/g19_scores.py','scripts/g8_common.py','src/mcse/asr_scoring.py','src/mcse/ctc.py']
    plan={'frozen_utc':now(),'max_seconds':1800,'manifest_sha256':sha(OUT/'manifest.jsonl'),
          'hashes':{n:sha(ROOT/n) for n in names},'parent':{e:{'provenance_sha256':sha(G22/'recognition'/e/'provenance.json'),
                   'cache_sha256':sha(G22/'recognition'/e/'cache_new.jsonl'),'score_sha256':sha(G22/'recognition'/e/'asr_scores.jsonl')} for e in ('Whisper','CTC')}}
    dump(REC/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G23C_固定双识别器执行协议.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G23C 固定双识别器执行协议\n\n两个固定后端各1440个控制。所有480个0dB波形必须在任何新解码前与G22的同模型缓存及原假设文本匹配；不得重新解码0dB来替换原基准。仅对未缓存的非零增益样本运行识别。\n\n记录规范化文本变化及S/D/I的配对差，不选最优增益。两个解码后端共享1800秒上限；完整代码、来源、模型/配置身份和清单已冻结。每个后端归档后再继续。\n')
    archive('recognition_protocol',report,[REC/'frozen_plan.json']+[ROOT/n for n in names])
    torch.set_num_threads(2);deadline=time.monotonic()+plan['max_seconds']-60
    for e in ('Whisper','CTC'):backend(e,rows,pcm,refs,plan,deadline)
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    from g23_finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception as e:status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:(OUT/'keep_awake_stop').write_text('G23 gain-control job ended\n')
    try:
        for _ in range(24):
            if (OUT/'power_helper.json').exists() and meta(OUT/'power_helper.json')['status']=='released':break
            time.sleep(2)
        assert meta(OUT/'power_helper.json')['status']=='released'
        from g23_accept import main as accept
        accept()
        print('G23 gain-control stage independently accepted',flush=True)
    except Exception as e:status(stage='acceptance_failed',error=repr(e));traceback.print_exc();raise
