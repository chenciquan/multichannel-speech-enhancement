#!/usr/bin/env python3
"""Two fresh recognitions of each canonical level-controlled diagnostic input."""
import os
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='1'
import gc
import json
import time
import traceback
import torch
from g25_prepare import ROOT,OUT,PARENT,sha,read,dump,now,archive
from g19_scores import fixed_decoder,waveform
from mcse.asr_scoring import NORMALIZATION,score_words
from mcse.asr_level import standardize_level

REC=OUT/'recognition'
FIELDS=('base_system','sample_id','observation','normalization_id','ASR_regime','source_id','speaker_id','transcript',
        'room_id','requested_rt60','target_azimuth','noise_id','noise_field','relative_angle','control_only','oracle_clean_information')


def meta(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def status(**fields):dump(OUT/'status.json',{'utc':now(),**fields})


def inputs():
    plan=meta(OUT/'frozen_plan.json');done=meta(OUT/'PREPARATION_COMPLETE.json')
    assert done['status']=='ok' and sha(OUT/'manifest.jsonl')==done['manifest_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    rows=read(OUT/'manifest.jsonl');index={(r['base_system'],r['sample_id'],r['observation']):r for r in rows}
    assert len(rows)==len(index)==960
    canonical={}
    for offset in range(0,960,2):
        a,b=rows[offset:offset+2]
        assert a['observation']=='normalized_first' and b['observation']=='normalized_second'
        assert (a['base_system'],a['sample_id'])==(b['base_system'],b['sample_id'])
        assert a['output_path']==b['output_path'] and a['output_sha256']==b['output_sha256']
        wave,key=waveform(a['output_path'],a['output_sha256']);assert key==a['output_samples_sha256']==b['output_samples_sha256']
        canonical[a['base_system'],a['sample_id']]=wave.tobytes()
    config=plan['normalization']
    for r in read(PARENT/'manifest.jsonl'):
        wave,key=waveform(r['output_path'],r['output_sha256']);assert key==r['output_samples_sha256']
        standardized,_=standardize_level(wave,config['target_rms'],config['peak_limit'])
        assert standardized.tobytes()==canonical[r['base_system'],r['sample_id']]
    dump(OUT/'INPUT_ACCEPTANCE_AUDIT.json',{'status':'ok','utc':now(),'canonical_inputs':480,
          'binary_variant_inputs_checked':1920,'all_standardized_bytes_equal':True,'normalization_id':config['preprocessing_id'],
          'manifest_sha256':sha(OUT/'manifest.jsonl')},True)
    report=ROOT/'docs/G25_输入标准化独立核验.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G25 输入标准化独立核验\n\n960观察键及480个规范化输入已回读；重新读取G24全部1920波形并计算固定标准化，与G25规范化输入逐字节一致。两次规范化观察使用同一文件。尚未解码，不预判识别结果。\n')
    archive('input_acceptance',report,[OUT/'INPUT_ACCEPTANCE_AUDIT.json'])
    return rows


def backend(engine,rows,plan,deadline):
    prior_dir=PARENT/'recognition'/engine;prior=meta(prior_dir/'provenance.json')
    assert sha(prior_dir/'provenance.json')==plan['parent'][engine]['provenance_sha256']
    assert sha(prior_dir/'observations.jsonl')==plan['parent'][engine]['observations_sha256']
    raw={(r['base_system'],r['sample_id']):r for r in read(prior_dir/'observations.jsonl') if r['observation']=='original_first'}
    assert len(raw)==480
    for r in rows:assert raw[r['base_system'],r['sample_id']]['output_samples_sha256']==r['raw_source_samples_sha256']
    folder=REC/engine;folder.mkdir(exist_ok=False)
    preceding='recognition_protocol' if engine=='Whisper' else 'whisper'
    assert (OUT/'stages'/preceding/'archive_manifest.json').exists()
    dump(folder/'provenance.json',{'engine':engine,'model_id':prior['model_id'],'decoding':prior['decoding'],
          'normalization':NORMALIZATION,'reference_prompt':False,'cache_skipping':False,'ASR_regime':'level_control_v1',
          'input_preprocessing':meta(OUT/'frozen_plan.json')['normalization'],
          'plan_sha256':sha(REC/'frozen_plan.json'),'previous_archive_sha256':sha(OUT/'stages'/preceding/'archive_manifest.json'),
          'parent_raw_observations_sha256':plan['parent'][engine]['observations_sha256'],'started_utc':now()},True)
    decode=fixed_decoder(engine,prior);calls=0
    with (folder/'observations.jsonl').open('x',encoding='utf-8') as log:
        for i,r in enumerate(rows):
            assert time.monotonic()<deadline,'G25 decoder deadline exceeded'
            a,key=waveform(r['output_path'],r['output_sha256']);assert key==r['output_samples_sha256']
            started=time.perf_counter();h=decode(a);calls+=1
            parent=raw[r['base_system'],r['sample_id']]
            item={field:r[field] for field in FIELDS}
            item.update(decode_index=i,actual_forward=True,model_id=prior['model_id'],output_samples_sha256=key,
                        hypothesis=h,score=score_words(r['transcript'],h),raw_hypothesis=parent['hypothesis'],
                        raw_score=parent['score'],raw_source_samples_sha256=r['raw_source_samples_sha256'],
                        seconds=time.perf_counter()-started)
            log.write(json.dumps(item,allow_nan=False)+'\n');log.flush()
            if calls%120==0:status(stage='fresh_ASR_level_control',engine=engine,forwards=calls,total=960)
    assert calls==960
    dump(folder/'COMPLETE.json',{'status':'ok','utc':now(),'fresh_forwards':calls,'cache_skips':0,
          'observations_sha256':sha(folder/'observations.jsonl')},True)
    report=ROOT/'docs'/f'G25_{engine}规范化独立识别覆盖.md'
    with report.open('x',encoding='utf-8') as f:
        f.write(f'# G25 {engine} 规范化独立识别覆盖\n\n480个规范化基准各两次真实解码，共960前向，没有缓存跳过。raw对照引用已验收G24的原幅度第一次结果，原始波形/文本保持不变。模型和解码设置不变，新增电平预处理及其ID另行记录。先保存覆盖，后作配对读回。\n')
    archive(engine.lower(),report,[folder/'COMPLETE.json',folder/'provenance.json',ROOT/'scripts/g25_asr.py'])
    del decode;gc.collect();torch.cuda.empty_cache()


def main():
    rows=inputs();REC.mkdir(exist_ok=False)
    names=['scripts/g25_asr.py','scripts/g25_finish.py','scripts/g25_accept.py','scripts/g25_prepare.py',
           'src/mcse/asr_level.py','scripts/g19_scores.py','scripts/g8_common.py','src/mcse/asr_scoring.py','src/mcse/ctc.py']
    plan={'frozen_utc':now(),'max_seconds':1800,'manifest_sha256':sha(OUT/'manifest.jsonl'),
          'hashes':{n:sha(ROOT/n) for n in names},'parent':{e:{'provenance_sha256':sha(PARENT/'recognition'/e/'provenance.json'),
                   'observations_sha256':sha(PARENT/'recognition'/e/'observations.jsonl')} for e in ('Whisper','CTC')}}
    dump(REC/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G25C_规范化双识别器执行协议.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G25C 规范化双识别器执行协议\n\n两个后端各960次真实前向，480个规范化输入各解码两遍。旧raw文本仅作对照，不覆盖。模型/解码与文字计分标准化保持不变；新波形电平预处理以单独ID和参数追踪，不能混同raw-ASR条件。两个后端解码共用1800秒上限。\n\n主分析先检验规范化重复性，再报告raw与规范化S/D/I及文本变化。比较480个第一次，不将重复解码翻倍为独立样本。不以WER选电平或声称增强模型改善；不读取参考来选择输出增益。\n')
    archive('recognition_protocol',report,[REC/'frozen_plan.json']+[ROOT/n for n in names])
    torch.set_num_threads(2);deadline=time.monotonic()+plan['max_seconds']-60
    for e in ('Whisper','CTC'):backend(e,rows,plan,deadline)
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    from g25_finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception as e:status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:(OUT/'keep_awake_stop').write_text('G25 ASR-only level-control job ended\n')
    try:
        for _ in range(24):
            if (OUT/'power_helper.json').exists() and meta(OUT/'power_helper.json')['status']=='released':break
            time.sleep(2)
        assert meta(OUT/'power_helper.json')['status']=='released'
        from g25_accept import main as accept
        accept();print('G25 level-control stage accepted',flush=True)
    except Exception as e:status(stage='acceptance_failed',error=repr(e));traceback.print_exc();raise
