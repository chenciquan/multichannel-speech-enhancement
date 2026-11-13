#!/usr/bin/env python3
"""Complete old development utterances, two fixed decoders and separate input regimes."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import gc
import hashlib
import json
import time
import traceback
from collections import Counter
import torch
from g32_common import ROOT,OUT,sha,read,dump,now,meta,archive,status,data_guard
from g19_scores import waveform,fixed_decoder
from mcse.asr_level import standardize_level
from mcse.asr_scoring import NORMALIZATION,score_words

REC=OUT/'recognition';RAW=ROOT/'artifacts/g22_matched_rooms_v1';LEVEL=ROOT/'artifacts/g26_room_ASR_level_v1'
FIELDS=('system','source_system','method','sample_id','source_id','speaker_id','transcript','condition','noise_field',
        'noise_id','snr_db','relative_angle','target_azimuth','requested_rt60','room_id','frames','full_source','ASR_scoring_allowed')


def pcm(path,digest,regime,config):
    a,key=waveform(path,digest)
    if regime=='level_control':
        a,diagnostics=standardize_level(a,config['target_rms'],config['peak_limit']);key=hashlib.sha256(a.tobytes()).hexdigest()
    else:diagnostics=None
    return a,key,diagnostics


def seed(engine,regime,plan):
    base=RAW if regime=='raw' else LEVEL;folder=base/'recognition'/engine
    expected=plan['seeds'][engine][regime]
    assert sha(folder/'asr_scores.jsonl')==expected['scores_sha256'] and sha(folder/'provenance.json')==expected['provenance_sha256']
    provenance=meta(folder/'provenance.json');assert provenance['normalization']==NORMALIZATION and provenance['reference_prompt'] is False
    if regime=='level_control':assert provenance['input_preprocessing']==plan['normalization'] and provenance['ASR_regime']=='level_control_v1'
    cache={}
    for r in read(folder/'asr_scores.jsonl'):
        for key,hyp in ((r['output_samples_sha256'],r['hypothesis']),(r['reference_samples_sha256'],r['reference_hypothesis'])):
            if key in cache:assert cache[key]==hyp,'Canonical seed conflict'
            else:cache[key]=hyp
    return provenance,cache


def main():
    assert meta(OUT/'SIGNAL_ACCEPTANCE_AUDIT.json')['status']=='ok';data_guard()
    config=meta(LEVEL/'frozen_plan.json')['normalization'];assert sha(ROOT/'src/mcse/asr_level.py')==config['code_sha256']
    REC.mkdir(exist_ok=False)
    names=['scripts/g32_asr.py','scripts/g32_asr_finish.py','scripts/g32_common.py','scripts/g19_scores.py',
        'scripts/g8_common.py','src/mcse/ctc.py','src/mcse/asr_scoring.py','src/mcse/asr_level.py',
        'artifacts/g32_complete_dev_v1/signals.jsonl','artifacts/g32_complete_dev_v1/SIGNAL_ACCEPTANCE_AUDIT.json',
        'artifacts/g31_development_signals_v1/primary_decision.json']
    plan={'utc':now(),'max_seconds':3600,'rows_each_regime':468,'anchors_each_regime':4,'engines':['Whisper','CTC'],
        'regimes':['raw','level_control'],'normalization':config,'reference_prompt':False,
        'hashes':{n:sha(ROOT/n) for n in names},'seeds':{},'G31_extension_gate_passed':False}
    assert meta(ROOT/'artifacts/g31_development_signals_v1/primary_decision.json')['extension_gate_passed'] is False
    for engine in plan['engines']:
        plan['seeds'][engine]={}
        for regime,base in (('raw',RAW),('level_control',LEVEL)):
            folder=base/'recognition'/engine;digest=sha(folder/'asr_scores.jsonl')
            anchor=(meta(RAW/'FINAL_ACCEPTANCE_AUDIT.json')['ASR'][engine]['score_sha256'] if regime=='raw' else meta(LEVEL/'recognition/completion_audit.json')['ASR'][engine]['score_sha256'])
            assert digest==anchor
            plan['seeds'][engine][regime]={'scores_sha256':digest,'accepted_score_sha256':anchor,'provenance_sha256':sha(folder/'provenance.json')}
    dump(REC/'frozen_plan.json',plan,True)
    manifest=[];refs={};peak_limited=0
    for r in read(OUT/'signals.jsonl'):
        assert r['full_source'] and r['ASR_scoring_allowed'] and r['frames']<=64000
        row={k:r[k] for k in FIELDS};row.update(output_path=r['output_path'],output_sha256=r['output_sha256'],
            reference_path=r['reference_path'],reference_sha256=r['reference_sha256'],inputs={})
        for regime in plan['regimes']:
            _,key,diagnostics=pcm(r['output_path'],r['output_sha256'],regime,config)
            refkey=regime,r['reference_path']
            if refkey not in refs:
                _,rk,rd=pcm(r['reference_path'],r['reference_sha256'],regime,config);refs[refkey]=(rk,rd)
            rk,rd=refs[refkey];row['inputs'][regime]={'output_key':key,'reference_key':rk,'output_level':diagnostics,'reference_level':rd}
            if diagnostics is not None:peak_limited+=int(diagnostics['peak_limited'])
        manifest.append(row)
    assert len(manifest)==468
    with (REC/'input_manifest.jsonl').open('x',encoding='utf-8') as f:
        for r in manifest:f.write(json.dumps(r,allow_nan=False)+'\n')
    anchors=[]
    for s in meta(OUT/'data_plan.json')['sources']:
        path=ROOT/'data/g3_v1/sources'/s['path'];row={'source_id':s['source_id'],'speaker_id':s['speaker_id'],
            'transcript':s['transcript'],'path':str(path),'sha256':s['sha256'],'inputs':{}}
        for regime in plan['regimes']:
            _,key,diag=pcm(str(path),s['sha256'],regime,config);row['inputs'][regime]={'key':key,'level':diag}
        anchors.append(row)
    dump(REC/'dry_anchors.json',anchors,True)
    protocol=ROOT/'docs/G32E_完整开发语句固定双识别器协议.md'
    with protocol.open('x',encoding='utf-8') as f:f.write('# G32E 完整开发语句固定双识别器协议\n\n每后端raw468行、电平控制468行，另有4干声×2条件参考。使用既有Whisper tiny.en与greedyCTC，相同模型/解码/文本规范化，无参考提示。raw及电平缓存分开，分别只复用G22/G26验收的同模型同预处理同PCM；未缓存参考、输出和干声必须实际解码。\n\n全部输入原文件SHA与两条件PCM键在解码前冻结，固定目标RMS=.05、峰值≤.95，不调参数或选更低条件覆盖。实际峰值受限计数保存，不能声称都达到同RMS。两后端共3600秒上限。原信号/质量及G31未过门槛不变。完整原句只合计16词，干声、混响参考及系统分数分开，不以S/D/I代替声学删除。\n')
    archive('recognition_protocol',protocol,[REC/'frozen_plan.json',REC/'input_manifest.jsonl',REC/'dry_anchors.json']+[ROOT/n for n in names[:8]])
    dump(REC/'input_audit.json',{'rows':468,'dry_anchors':4,'peak_limited_outputs':peak_limited,'input_manifest_sha256':sha(REC/'input_manifest.jsonl'),'dry_anchors_sha256':sha(REC/'dry_anchors.json')},True)
    torch.set_num_threads(2);deadline=time.monotonic()+plan['max_seconds']-120
    for engine in plan['engines']:
        priors={};caches={}
        for regime in plan['regimes']:priors[regime],caches[regime]=seed(engine,regime,plan)
        assert all(priors['raw'][k]==priors['level_control'][k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
        decode=fixed_decoder(engine,priors['raw'])
        for regime in plan['regimes']:
            folder=REC/engine/regime;folder.mkdir(parents=True,exist_ok=False);cache=caches[regime];fresh=Counter()
            dump(folder/'provenance.json',{'engine':engine,'model_id':priors[regime]['model_id'],'decoding':priors[regime]['decoding'],
                'normalization':NORMALIZATION,'reference_prompt':False,'ASR_regime':regime,
                'input_preprocessing':config if regime=='level_control' else None,'seed':plan['seeds'][engine][regime],
                'plan_sha256':sha(REC/'frozen_plan.json'),'input_manifest_sha256':sha(REC/'input_manifest.jsonl')},True)
            with (folder/'cache_new.jsonl').open('x',encoding='utf-8') as clog:
                def recognize(path,digest,expected,kind):
                    assert time.monotonic()<deadline
                    a,key,_=pcm(path,digest,regime,config);assert key==expected
                    if key not in cache:
                        started=time.perf_counter();cache[key]=decode(a);fresh[kind]+=1
                        clog.write(json.dumps({'sample_sha256':key,'hypothesis':cache[key],'model_id':priors[regime]['model_id'],
                            'ASR_regime':regime,'normalization_id':config['preprocessing_id'] if regime=='level_control' else None,
                            'first_kind':kind,'seconds':time.perf_counter()-started})+'\n');clog.flush()
                    return key,cache[key]
                with (folder/'dry_anchor_scores.jsonl').open('x',encoding='utf-8') as log:
                    for r in anchors:
                        key,h=recognize(r['path'],r['sha256'],r['inputs'][regime]['key'],'dry_anchor')
                        log.write(json.dumps({'source_id':r['source_id'],'speaker_id':r['speaker_id'],'transcript':r['transcript'],
                            'ASR_regime':regime,'sample_sha256':key,'hypothesis':h,'score':score_words(r['transcript'],h)})+'\n')
                with (folder/'asr_scores.jsonl').open('x',encoding='utf-8') as log:
                    for i,r in enumerate(manifest,1):
                        rk,rh=recognize(r['reference_path'],r['reference_sha256'],r['inputs'][regime]['reference_key'],'reference')
                        key,h=recognize(r['output_path'],r['output_sha256'],r['inputs'][regime]['output_key'],'output')
                        row={k:r[k] for k in FIELDS};row.update(ASR_regime=regime,hypothesis=h,reference_hypothesis=rh,
                            score=score_words(r['transcript'],h),reference_score=score_words(r['transcript'],rh),
                            output_samples_sha256=key,reference_samples_sha256=rk)
                        log.write(json.dumps(row,allow_nan=False)+'\n')
                        if i%78==0:status(stage='ASR',engine=engine,regime=regime,done=i,total=468,fresh=dict(fresh))
            dump(folder/'COMPLETE.json',{'status':'ok','utc':now(),'rows':468,'dry_anchors':4,'fresh':dict(fresh),
                'score_sha256':sha(folder/'asr_scores.jsonl'),'cache_sha256':sha(folder/'cache_new.jsonl'),
                'anchors_sha256':sha(folder/'dry_anchor_scores.jsonl')},True)
        report=ROOT/'docs'/f'G32_{engine}_双条件识别覆盖.md'
        with report.open('x',encoding='utf-8') as f:f.write(f'# G32 {engine} 双条件识别覆盖\n\nraw/电平控制各468行及各4个原干声参考已完成。缓存按实际PCM与模型/预处理严格区分，真实新增前向数分别保存；四条原句共16词，不能把重复条件当新增独立文本。\n')
        archive(engine.lower(),report,[REC/engine/regime/'COMPLETE.json' for regime in plan['regimes']]+[REC/engine/regime/'provenance.json' for regime in plan['regimes']])
        del decode;gc.collect();torch.cuda.empty_cache()
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    from g32_asr_finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception as e:dump(OUT/'ASR_failure.json',{'utc':now(),'error':repr(e)});traceback.print_exc();raise
    finally:(OUT/'asr_keep_awake_stop').write_text('G32 fixed dual ASR ended\n')
    for _ in range(30):
        if (OUT/'power_asr_helper.json').exists() and meta(OUT/'power_asr_helper.json')['status']=='released':break
        time.sleep(2)
    from g32_asr_finish import accept
    accept();print('G32 complete development signal/ASR diagnostic accepted',flush=True)
