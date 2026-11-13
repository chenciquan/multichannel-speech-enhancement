#!/usr/bin/env python3
"""Fresh independent forwards for repeated baselines and exact binary gains."""
import os
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='1'
import gc
import json
import time
import traceback
import numpy as np
import torch
from g24_prepare import ROOT,OUT,PARENT,sha,read,dump,now,archive
from g19_scores import fixed_decoder,load_cache,waveform
from mcse.asr_scoring import NORMALIZATION,score_words

REC=OUT/'recognition'
FIELDS=('base_system','sample_id','observation','gain_exact','gain_db','source_id','speaker_id','transcript',
        'room_id','requested_rt60','target_azimuth','noise_id','noise_field','relative_angle','control_only','oracle_clean_information')


def meta(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def status(**fields):dump(OUT/'status.json',{'utc':now(),**fields})


def input_audit():
    done=meta(OUT/'PREPARATION_COMPLETE.json');plan=meta(OUT/'frozen_plan.json')
    assert done['status']=='ok' and sha(OUT/'manifest.jsonl')==done['manifest_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    rows=read(OUT/'manifest.jsonl');index={(r['base_system'],r['sample_id'],r['observation']):r for r in rows}
    bases={(r['base_system'],r['sample_id']) for r in rows}
    assert len(rows)==len(index)==1920 and len(bases)==480
    for offset in range(0,1920,4):
        block=rows[offset:offset+4]
        assert [r['observation'] for r in block]==plan['observation_order']
        assert len({(r['base_system'],r['sample_id']) for r in block})==1
        original,key=waveform(block[0]['output_path'],block[0]['output_sha256'])
        assert block[0]['output_path']==block[3]['output_path'] and block[0]['output_sha256']==block[3]['output_sha256']
        for r in block:
            a,k=waveform(r['output_path'],r['output_sha256'])
            assert k==r['output_samples_sha256'] and key==r['baseline_samples_sha256']
            assert np.array_equal(a/r['gain_exact'],original)
    report=ROOT/'docs/G24_重复观察输入独立核验.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G24 重复观察输入独立核验\n\n1920观察键完整，480个基准均按原幅度第一次、半幅、四分之一幅、原幅度第二次排列。所有文件与PCM哈希再次核对，缩放音频逐样本精确反向还原，两个原幅度观察使用同一文件。未运行解码。\n')
    dump(OUT/'INPUT_ACCEPTANCE_AUDIT.json',{'status':'ok','utc':now(),'observations':1920,'bases':480,
          'all_binary_inverse_exact':True,'manifest_sha256':sha(OUT/'manifest.jsonl')},True)
    archive('input_acceptance',report,[OUT/'INPUT_ACCEPTANCE_AUDIT.json'])
    return rows


def backend(engine,rows,plan,deadline):
    folder=REC/engine;prior_dir=PARENT/'recognition'/engine
    prior=meta(prior_dir/'provenance.json')
    for name,field in (('provenance.json','provenance_sha256'),('asr_scores.jsonl','scores_sha256'),('cache_new.jsonl','cache_sha256')):
        assert sha(prior_dir/name)==plan['parent'][engine][field]
    cache,conflicts,sources=load_cache(engine,prior,prior_dir);assert not conflicts
    parent={(r['base_system'],r['sample_id']):r for r in read(prior_dir/'asr_scores.jsonl') if r['gain_db']==0};assert len(parent)==480
    for r in rows:
        p=parent[r['base_system'],r['sample_id']]
        assert r['baseline_samples_sha256']==p['output_samples_sha256'] and cache[p['output_samples_sha256']]==p['hypothesis']
    del cache
    preceding='recognition_protocol' if engine=='Whisper' else 'whisper'
    assert (OUT/'stages'/preceding/'archive_manifest.json').exists();folder.mkdir(exist_ok=False)
    provenance={'engine':engine,'model_id':prior['model_id'],'decoding':prior['decoding'],'normalization':NORMALIZATION,
                'reference_prompt':False,'history_for_comparison_only':True,'cache_skipping':False,
                'source_caches':sources,'plan_sha256':sha(REC/'frozen_plan.json'),
                'previous_archive_sha256':sha(OUT/'stages'/preceding/'archive_manifest.json'),'started_utc':now()}
    dump(folder/'provenance.json',provenance,True)
    decode=fixed_decoder(engine,prior);calls=0
    with (folder/'observations.jsonl').open('x',encoding='utf-8') as log:
        for i,r in enumerate(rows):
            assert time.monotonic()<deadline,'G24 fresh ASR deadline exceeded'
            a,key=waveform(r['output_path'],r['output_sha256']);assert key==r['output_samples_sha256']
            started=time.perf_counter()
            # Deliberately unconditional: even identical original_first/second inputs run the network.
            hypothesis=decode(a);calls+=1
            record={field:r[field] for field in FIELDS}
            p=parent[r['base_system'],r['sample_id']]
            record.update(decode_index=i,output_samples_sha256=key,hypothesis=hypothesis,
                          score=score_words(r['transcript'],hypothesis),historical_baseline_hypothesis=p['hypothesis'],
                          historical_baseline_score=p['score'],seconds=time.perf_counter()-started,
                          actual_forward=True,model_id=prior['model_id'])
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            if calls%120==0:status(stage='fresh_asr',engine=engine,forwards=calls,total=1920)
    assert calls==1920
    dump(folder/'COMPLETE.json',{'status':'ok','utc':now(),'actual_forwards':calls,'cache_skips':0,
          'observations_sha256':sha(folder/'observations.jsonl')},True)
    report=ROOT/'docs'/f'G24_{engine}独立推理覆盖.md'
    with report.open('x',encoding='utf-8') as f:
        f.write(f'# G24 {engine} 独立推理覆盖\n\n全部1920观察均实际调用识别网络，包括480个基准各自两次原幅度，缓存跳过为0。历史缓存仅用于定位旧文本，未作为新输出，也未被新观察覆盖。模型身份与解码配置沿用G23；重复与增益差异将在后续配对汇总。\n')
    archive(engine.lower(),report,[folder/'COMPLETE.json',folder/'provenance.json',ROOT/'scripts/g24_asr.py'])
    del decode;gc.collect();torch.cuda.empty_cache()


def main():
    rows=input_audit();REC.mkdir(exist_ok=False)
    names=['scripts/g24_asr.py','scripts/g24_finish.py','scripts/g24_accept.py','scripts/g24_prepare.py',
           'scripts/g19_scores.py','scripts/g8_common.py','src/mcse/asr_scoring.py','src/mcse/ctc.py']
    plan={'frozen_utc':now(),'max_seconds':1800,'fresh_forwards_each':1920,'manifest_sha256':sha(OUT/'manifest.jsonl'),
          'hashes':{n:sha(ROOT/n) for n in names},'parent':{e:{'provenance_sha256':sha(PARENT/'recognition'/e/'provenance.json'),
                   'scores_sha256':sha(PARENT/'recognition'/e/'asr_scores.jsonl'),'cache_sha256':sha(PARENT/'recognition'/e/'cache_new.jsonl')} for e in ('Whisper','CTC')}}
    dump(REC/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G24C_全观察真实解码执行协议.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G24C 全观察真实解码执行协议\n\n两个后端各1920次真实推理，不按PCM哈希跳过任何重复观察；观察日志单独保存，不加入历史首观测缓存。每个基准按既定四观察顺序执行。两个后端解码共用1800秒上限，配置/来源/代码冻结。\n\n主分析先报告跨次与同次原幅度重复性，再解释二进制增益差；任何不稳定基准均保留并单列，不静默剔除。波形缩放精确可逆只排除缩放步骤的舍入，不排除识别器内部数值行为。无训练、增强推理或最优增益选择。\n')
    archive('recognition_protocol',report,[REC/'frozen_plan.json']+[ROOT/n for n in names])
    torch.set_num_threads(2);deadline=time.monotonic()+plan['max_seconds']-60
    for e in ('Whisper','CTC'):backend(e,rows,plan,deadline)
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    from g24_finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception as e:status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:(OUT/'keep_awake_stop').write_text('G24 fresh-repeat job ended\n')
    try:
        for _ in range(24):
            if (OUT/'power_helper.json').exists() and meta(OUT/'power_helper.json')['status']=='released':break
            time.sleep(2)
        assert meta(OUT/'power_helper.json')['status']=='released'
        from g24_accept import main as accept
        accept();print('G24 independent-repeat stage accepted',flush=True)
    except Exception as e:status(stage='acceptance_failed',error=repr(e));traceback.print_exc();raise
