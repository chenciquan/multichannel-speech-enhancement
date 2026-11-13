#!/usr/bin/env python3
"""Fixed G22 recognition, including uncached room-dependent references."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import gc
import json
import time
import traceback
from collections import defaultdict,Counter
import torch
from g22_common import ROOT,OUT,BASE,sha,dump,read,now,archive,metadata,guard,status
from g19_scores import fixed_decoder,load_cache,waveform
from mcse.asr_scoring import NORMALIZATION,score_words

REC=OUT/'recognition';PRIOR=ROOT/'artifacts/g21_truth_covariance_v1/recognition'
FIELDS=('system','sample_id','source_id','speaker_id','transcript','condition','room_id','matched_room_pair',
        'requested_rt60','target_azimuth','noise_field','noise_id','relative_angle','common_scale','oracle_clean_information','deployable')


def backend(engine,rows,plan,deadline):
    previous='recognition_protocol' if engine=='Whisper' else 'whisper'
    assert (OUT/'stages'/previous/'archive_manifest.json').exists()
    prior=metadata(PRIOR/engine/'provenance.json')
    assert sha(PRIOR/engine/'provenance.json')==plan['prior'][engine]['provenance_sha256']
    assert sha(PRIOR/engine/'cache_new.jsonl')==plan['prior'][engine]['cache_sha256']
    cache,conflicts,sources=load_cache(engine,prior,PRIOR/engine);assert not conflicts
    folder=REC/engine;folder.mkdir(exist_ok=False)
    dump(folder/'provenance.json',{'started_utc':now(),'engine':engine,'model_id':prior['model_id'],
          'decoding':prior['decoding'],'normalization':NORMALIZATION,'reference_prompt':False,
          'source_caches':sources,'frozen_plan_sha256':sha(REC/'frozen_plan.json'),
          'previous_stage_sha256':sha(OUT/'stages'/previous/'archive_manifest.json'),
          'signal_sha256':plan['signal_sha256'],'references':'Decode each uncached PCM hash; never reuse by transcript, room label, or old sample ID'},True)
    decode=fixed_decoder(engine,prior);path_keys={};new=Counter();totals=defaultdict(Counter)
    with (folder/'cache_new.jsonl').open('x',encoding='utf-8') as clog, (folder/'asr_scores.jsonl').open('x',encoding='utf-8') as log:
        def recognize(path,expected,kind):
            if path not in path_keys:
                samples,key=waveform(path,expected);path_keys[path]=key
                if key not in cache:
                    assert time.monotonic()<deadline,'G22 fixed ASR deadline exceeded'
                    started=time.perf_counter();cache[key]=decode(samples)
                    record={'sample_sha256':key,'hypothesis':cache[key],'model_id':prior['model_id'],
                            'first_kind':kind,'seconds':time.perf_counter()-started}
                    clog.write(json.dumps(record)+'\n');clog.flush();new[kind]+=1
            key=path_keys[path];return key,cache[key]
        for i,r in enumerate(rows,1):
            assert time.monotonic()<deadline
            rk,rh=recognize(r['reference_path'],r['reference_sha256'],'reference')
            key,h=recognize(r['output_path'],r['output_sha256'],'output')
            score=score_words(r['transcript'],h);refscore=score_words(r['transcript'],rh)
            item={k:r[k] for k in FIELDS}
            item.update(hypothesis=h,score=score,reference_hypothesis=rh,reference_score=refscore,
                        output_samples_sha256=key,reference_samples_sha256=rk)
            log.write(json.dumps(item,allow_nan=False)+'\n')
            totals[r['system']].update({k:score[k] for k in ('N','S','D','I')});totals[r['system']]['rows']+=1
            if i%168==0:status(stage='asr',engine=engine,done=i,total=16488,new_reference_decodes=new['reference'],new_output_decodes=new['output'])
    assert sum(t['rows'] for t in totals.values())==16488
    marker={'status':'ok','rows':16488,'completed_utc':now(),'new_decodes':dict(new),'unique_paths':len(path_keys),
            'cache_entries':len(cache),'asr_scores_sha256':sha(folder/'asr_scores.jsonl'),'cache_new_sha256':sha(folder/'cache_new.jsonl')}
    dump(folder/'COMPLETE.json',marker,True)
    lines=[f'# G22 {engine} 识别覆盖','','固定G21解码身份和选项，仅按PCM样本哈希复用缓存。新房间及共同尺度下未缓存的参考实际解码，不按转写文本或旧条件ID套用参考成绩。','',
           '|系统|记录数|参考词数|替换|删除|插入|微平均WER|','|---|---:|---:|---:|---:|---:|---:|']
    for name,t in sorted(totals.items()):lines.append(f"|{name}|{t['rows']}|{t['N']}|{t['S']}|{t['D']}|{t['I']}|{(t['S']+t['D']+t['I'])/t['N']:.2%}|")
    lines+=['','覆盖表混合两房间、方向、RT和干净条件，不用来选择方法。真值统计只有432条带噪，其他系统各504条，不直接比较表中合并均值。',
            '',f"新增解码参考 {new['reference']} 条、输出 {new['output']} 条。",now()]
    report=ROOT/'docs'/f'G22_{engine}识别覆盖.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive(engine.lower(),report,[folder/'COMPLETE.json',folder/'provenance.json',ROOT/'scripts/g22_asr.py'])
    del decode;gc.collect();torch.cuda.empty_cache()


def main():
    guard();assert metadata(OUT/'SIGNAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    REC.mkdir(exist_ok=False)
    pins=['scripts/g22_asr.py','scripts/g22_asr_finish.py','scripts/g22_final_accept.py','scripts/g19_scores.py','scripts/g8_common.py',
          'src/mcse/asr_scoring.py','src/mcse/ctc.py','artifacts/g22_matched_rooms_v1/SIGNAL_ACCEPTANCE_AUDIT.json']
    plan={'frozen_utc':now(),'max_seconds':7200,'rows':16488,'signal_sha256':sha(OUT/'evaluation/combined_signals.jsonl'),
          'hashes':{n:sha(ROOT/n) for n in pins},'prior':{e:{'provenance_sha256':sha(PRIOR/e/'provenance.json'),
                  'cache_sha256':sha(PRIOR/e/'cache_new.jsonl')} for e in ('Whisper','CTC')},
          'new_references':'Decode only if PCM sample SHA absent in same recognizer cache; no transcript prompt'}
    dump(REC/'frozen_plan.json',plan,True)
    protocol=ROOT/'docs/G22D_新房间固定双识别器协议.md'
    with protocol.open('x',encoding='utf-8') as f:
        f.write('# G22D 新房间固定双识别器协议\n\n16488条信号/后端。使用与G21完全相同的Whisper和CTC解码设置；既有缓存按同一模型与PCM样本SHA复用，未见过的参考音频也必须实际识别。新房间和电平导致的参考变化不能按旧句子ID复用。\n\n真值统计只参与432带噪条件对照；盲系统另有72干净条件。WER微平均，参考每条件计一次。按房间、RT、方向、场型、噪声、说话人及全部三种子汇总，检查RT0假设/波形对应，不选择最佳种子或调参。两个后端合计7200秒上限。\n')
    archive('recognition_protocol',protocol,[REC/'frozen_plan.json']+[ROOT/n for n in pins])
    rows=read(OUT/'evaluation/combined_signals.jsonl');assert len(rows)==16488
    torch.set_num_threads(2);deadline=time.monotonic()+plan['max_seconds']-120
    for e in ('Whisper','CTC'):backend(e,rows,plan,deadline)
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    from g22_asr_finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception as e:status(stage='ASR_failed',error=repr(e));traceback.print_exc();raise
    finally:(OUT/'ASR_keep_awake_stop').write_text('G22 ASR queue ended\n')
