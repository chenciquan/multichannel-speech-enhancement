#!/usr/bin/env python3
"""Fixed raw/level ASR of oracle power-mask diagnostics, separately cached."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import gc
import hashlib
import time
import traceback
import torch
from g27_signal import ROOT,OUT,SOURCE,sha,read,dump,now,meta,archive
from g19_scores import waveform,fixed_decoder
from mcse.asr_level import standardize_level
from mcse.asr_scoring import NORMALIZATION,score_words

LEVEL=ROOT/'artifacts/g26_room_ASR_level_v1';REC=OUT/'recognition'
FIELDS=('system','sample_id','source_id','speaker_id','transcript','condition','room_id','matched_room_pair',
        'requested_rt60','target_azimuth','noise_field','noise_id','relative_angle','oracle_clean_information','deployable')


def seed(engine,regime,plan):
    base=SOURCE if regime=='raw' else LEVEL;folder=base/'recognition'/engine
    source=plan['seeds'][engine][regime]
    assert sha(folder/'provenance.json')==source['provenance_sha256'] and sha(folder/'asr_scores.jsonl')==source['scores_sha256']
    p=meta(folder/'provenance.json');assert p['normalization']==NORMALIZATION and not p['reference_prompt']
    if regime=='level_control':assert p['input_preprocessing']==plan['normalization'] and p['ASR_regime']=='level_control_v1'
    cache={}
    for r in read(folder/'asr_scores.jsonl'):
        for key,hyp in ((r['output_samples_sha256'],r['hypothesis']),(r['reference_samples_sha256'],r['reference_hypothesis'])):
            if key in cache:assert cache[key]==hyp,'Conflicting canonical seed hypotheses'
            else:cache[key]=hyp
    return p,cache


def pcm(path,digest,regime,config):
    a,key=waveform(path,digest)
    if regime=='level_control':
        a,info=standardize_level(a,config['target_rms'],config['peak_limit']);key=hashlib.sha256(a.tobytes()).hexdigest()
    else:info=None
    return a,key,info


def main():
    assert meta(OUT/'SIGNAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    assert meta(LEVEL/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    config=meta(LEVEL/'frozen_plan.json')['normalization'];assert sha(ROOT/'src/mcse/asr_level.py')==config['code_sha256']
    REC.mkdir(exist_ok=False)
    names=['scripts/g27_asr.py','scripts/g27_asr_finish.py','scripts/g19_scores.py','src/mcse/asr_level.py',
           'src/mcse/asr_scoring.py','scripts/g8_common.py','src/mcse/ctc.py','artifacts/g27_oracle_mask_v1/signals.jsonl',
           'artifacts/g27_oracle_mask_v1/SIGNAL_ACCEPTANCE_AUDIT.json','artifacts/g22_matched_rooms_v1/FINAL_ACCEPTANCE_AUDIT.json',
           'artifacts/g26_room_ASR_level_v1/FINAL_ACCEPTANCE_AUDIT.json',
           'artifacts/g22_matched_rooms_v1/evaluation/combined_signals.jsonl',
           'artifacts/g22_matched_rooms_v1/signal_systems.json']
    plan={'utc':now(),'rows_each_engine_regime':864,'engines':['Whisper','CTC'],'regimes':['raw','level_control'],
          'max_seconds':3600,'normalization':config,'hashes':{n:sha(ROOT/n) for n in names},'seeds':{}}
    for e in plan['engines']:
        plan['seeds'][e]={}
        for regime,base in (('raw',SOURCE),('level_control',LEVEL)):
            folder=base/'recognition'/e;digest=sha(folder/'asr_scores.jsonl')
            accepted=(meta(SOURCE/'FINAL_ACCEPTANCE_AUDIT.json')['ASR'][e]['score_sha256'] if regime=='raw' else meta(LEVEL/'recognition/completion_audit.json')['ASR'][e]['score_sha256'])
            assert digest==accepted
            plan['seeds'][e][regime]={'provenance_sha256':sha(folder/'provenance.json'),'scores_sha256':digest,'accepted_score_sha256':accepted}
    dump(REC/'frozen_plan.json',plan,True)
    rows=read(OUT/'signals.jsonl');assert len(rows)==864
    manifest=[];references={}
    for r in rows:
        record={k:r[k] for k in FIELDS};record.update(output_path=r['output_path'],output_sha256=r['output_sha256'],reference_path=r['reference_path'],reference_sha256=r['reference_sha256'],inputs={})
        for regime in plan['regimes']:
            _,key,diag=pcm(r['output_path'],r['output_sha256'],regime,config)
            pk=(regime,r['reference_path'])
            if pk not in references:
                _,rk,rd=pcm(r['reference_path'],r['reference_sha256'],regime,config);references[pk]=(rk,rd)
            rk,rd=references[pk];record['inputs'][regime]={'output_key':key,'reference_key':rk,'output_level':diag,'reference_level':rd}
        manifest.append(record)
    with (REC/'input_manifest.jsonl').open('x',encoding='utf-8') as f:
        import json
        for r in manifest:f.write(json.dumps(r,allow_nan=False)+'\n')
    protocol=ROOT/'docs/G27C_真值掩码双识别器原始与电平评分协议.md'
    with protocol.open('x',encoding='utf-8') as f:
        f.write('# G27C 真值掩码双识别器原始与电平评分协议\n\n对864条新增诊断输出分别做raw和固定G25/G26电平控制评分，两个固定识别器各1728行。两种输入条件的缓存分开，分别由已验收G22/G26的完整评分表按精确PCM建立；同PCM、同模型、同解码与同预处理才复用。参考音频逐例核对真实输入哈希，不按相同文本或房间标签复用。所有新规范化输入键在解码前冻结，未缓存波形实际前向。\n\n比较McNet掩码、真值功率比掩码和真值协方差。后两者含真值，均不可部署；不选更低的ASR条件替换另一条件。识别S/D/I与人声声学遗漏分开解释。共享3600秒上限，原增强/滤波输出与质量指标不变，无新训练或超参选择。\n')
    archive('recognition_protocol',protocol,[REC/'frozen_plan.json',REC/'input_manifest.jsonl']+[ROOT/n for n in names[:7]])
    torch.set_num_threads(2);deadline=time.monotonic()+plan['max_seconds']-120
    import json
    for engine in plan['engines']:
        priors={};caches={}
        for regime in plan['regimes']:priors[regime],caches[regime]=seed(engine,regime,plan)
        assert all(priors['raw'][k]==priors['level_control'][k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
        decode=fixed_decoder(engine,priors['raw'])
        for regime in plan['regimes']:
            folder=REC/engine/regime;folder.mkdir(parents=True,exist_ok=False)
            cache=caches[regime];fresh=0;reference_new=0;config_keys={};start=time.monotonic()
            dump(folder/'provenance.json',{'engine':engine,'model_id':priors[regime]['model_id'],'decoding':priors[regime]['decoding'],
                'normalization':NORMALIZATION,'reference_prompt':False,'ASR_regime':regime,
                'input_preprocessing':config if regime=='level_control' else None,'seed':plan['seeds'][engine][regime],
                'plan_sha256':sha(REC/'frozen_plan.json'),'input_manifest_sha256':sha(REC/'input_manifest.jsonl')},True)
            with (folder/'cache_new.jsonl').open('x',encoding='utf-8') as clog,(folder/'asr_scores.jsonl').open('x',encoding='utf-8') as log:
                for i,r in enumerate(manifest,1):
                    assert time.monotonic()<deadline
                    observed={}
                    for kind in ('reference','output'):
                        a,key,_=pcm(r[kind+'_path'],r[kind+'_sha256'],regime,config)
                        assert key==r['inputs'][regime][kind+'_key']
                        if key not in cache:
                            before=time.monotonic();cache[key]=decode(a);fresh+=1;reference_new+=int(kind=='reference')
                            clog.write(json.dumps({'sample_sha256':key,'hypothesis':cache[key],'model_id':priors[regime]['model_id'],
                                'ASR_regime':regime,'normalization_id':config['preprocessing_id'] if regime=='level_control' else None,
                                'first_kind':kind,'seconds':time.monotonic()-before})+'\n');clog.flush()
                        observed[kind]=(key,cache[key])
                    rk,rh=observed['reference'];key,h=observed['output']
                    row={k:r[k] for k in FIELDS};row.update(ASR_regime=regime,hypothesis=h,reference_hypothesis=rh,
                        score=score_words(r['transcript'],h),reference_score=score_words(r['transcript'],rh),
                        output_samples_sha256=key,reference_samples_sha256=rk)
                    log.write(json.dumps(row,allow_nan=False)+'\n')
                    if i%144==0:dump(OUT/'status.json',{'utc':now(),'stage':'ASR','engine':engine,'regime':regime,'done':i,'total':864,'fresh':fresh})
            assert reference_new==0,'All true reference waveforms must match corresponding accepted cache'
            dump(folder/'COMPLETE.json',{'status':'ok','utc':now(),'rows':864,'fresh_decodes':fresh,'reference_new':reference_new,
                'score_sha256':sha(folder/'asr_scores.jsonl'),'cache_sha256':sha(folder/'cache_new.jsonl'),'elapsed_seconds':time.monotonic()-start},True)
        report=ROOT/'docs'/f'G27_{engine}双电平识别覆盖.md'
        with report.open('x',encoding='utf-8') as f:f.write(f'# G27 {engine} 双电平识别覆盖\n\nraw和固定电平控制各864行完整覆盖，未混用两种输入缓存；参考波形均匹配相应已验收输入。实际新增前向数量及缓存哈希分别保存。所有诊断均含不可部署真值，不作新方法成果。\n')
        archive(engine.lower(),report,[REC/engine/regime/'COMPLETE.json' for regime in plan['regimes']]+[REC/engine/regime/'provenance.json' for regime in plan['regimes']])
        del decode;gc.collect();torch.cuda.empty_cache()
    from g27_asr_finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception as e:dump(OUT/'ASR_failure.json',{'utc':now(),'error':repr(e)});traceback.print_exc();raise
    finally:(OUT/'asr_keep_awake_stop').write_text('G27 ASR ended\n')
    for _ in range(24):
        if (OUT/'power_asr_helper.json').exists() and meta(OUT/'power_asr_helper.json')['status']=='released':break
        time.sleep(2)
    assert meta(OUT/'power_asr_helper.json')['status']=='released'
    from g27_asr_finish import accept
    accept();print('G27 full oracle-mask diagnostic accepted',flush=True)
