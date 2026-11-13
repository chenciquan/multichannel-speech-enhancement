#!/usr/bin/env python3
"""Fixed two-ASR evaluation of saved oracle-covariance diagnostic outputs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import gc
import json
import time
import traceback
from collections import Counter,defaultdict
import torch
from g21_truth_covariance import ROOT,OUT,sha,dump,read,now,archive
from g19_scores import fixed_decoder,load_cache,baseline_references,waveform,FIELDS
from mcse.asr_scoring import NORMALIZATION,score_words

REC=OUT/'recognition'
G18=ROOT/'artifacts/g18_room_stress_v1/evaluation'
G19=ROOT/'artifacts/g19_room_mwf_v1/evaluation'


def metadata(p):return json.loads(p.read_text())
def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})


def run(engine,rows,plan,deadline):
    previous='recognition_protocol' if engine=='Whisper' else 'whisper'
    assert (OUT/'stages'/previous/'archive_manifest.json').exists()
    old=G19/engine; prior=metadata(old/'provenance.json')
    for name in ('provenance.json','asr_scores.jsonl','cache_new.jsonl'):
        assert sha(old/name)==plan['history_sha256'][engine]['g19_'+name]
    cache,conflicts,sources=load_cache(engine,prior,old);assert not conflicts
    refs=baseline_references(G18/engine/'asr_scores.jsonl',cache)
    folder=REC/engine;folder.mkdir(exist_ok=False)
    provenance={'engine':engine,'model_id':prior['model_id'],'decoding':prior['decoding'],
                'normalization':NORMALIZATION,'reference_prompt':False,'oracle_signals':True,
                'source_caches':sources,'frozen_plan_sha256':sha(REC/'frozen_plan.json'),
                'previous_stage_sha256':sha(OUT/'stages'/previous/'archive_manifest.json'),
                'signal_sha256':plan['signal_sha256'],'started_utc':now()}
    dump(folder/'provenance.json',provenance,True)
    decode=fixed_decoder(engine,prior);totals=defaultdict(Counter);new=0;ref_hashes={}
    with (folder/'cache_new.jsonl').open('x',encoding='utf-8') as clog, (folder/'asr_scores.jsonl').open('x',encoding='utf-8') as log:
        for i,r in enumerate(rows,1):
            assert time.monotonic()<deadline,'G21 ASR bounded deadline exceeded'
            if r['reference_path'] not in ref_hashes:
                _,ref_hashes[r['reference_path']]=waveform(r['reference_path'],r['reference_sha256'])
            rk=ref_hashes[r['reference_path']]
            assert (rk,r['transcript'])==refs[r['sample_id']]
            audio,key=waveform(r['output_path'],r['output_sha256'])
            if key not in cache:
                started=time.perf_counter();cache[key]=decode(audio)
                cr={'sample_sha256':key,'model_id':prior['model_id'],'hypothesis':cache[key],
                    'seconds':time.perf_counter()-started}
                clog.write(json.dumps(cr)+'\n');clog.flush();new+=1
            score=score_words(r['transcript'],cache[key])
            item={field:r.get(field) for field in FIELDS}
            item.update(hypothesis=cache[key],score=score,reference_hypothesis=cache[rk],
                        reference_score=score_words(r['transcript'],cache[rk]),
                        reference_samples_sha256=rk,output_samples_sha256=key,
                        oracle_clean_information=True,deployable=False)
            log.write(json.dumps(item,allow_nan=False)+'\n')
            totals[r['system']].update({k:score[k] for k in ('N','S','D','I')});totals[r['system']]['rows']+=1
            if i%72==0:status(stage='oracle_asr',engine=engine,done=i,total=864)
    assert sum(v['rows'] for v in totals.values())==864
    marker={'rows':864,'new_decodes':new,'completed_utc':now(),
            'asr_scores_sha256':sha(folder/'asr_scores.jsonl'),'cache_new_sha256':sha(folder/'cache_new.jsonl')}
    dump(folder/'COMPLETE.json',marker,True)
    lines=[f'# G21 {engine} 真值统计诊断识别覆盖','','固定原解码配置；参考文本不作为提示，参考音频识别复用已校验缓存。系统使用真值统计，仅供诊断。','',
           '|系统|例数|参考词数|替换|删除|插入|微平均WER|','|---|---:|---:|---:|---:|---:|---:|']
    for name,t in sorted(totals.items()):lines.append(f"|{name}|{t['rows']}|{t['N']}|{t['S']}|{t['D']}|{t['I']}|{(t['S']+t['D']+t['I'])/t['N']:.2%}|")
    lines+=['','不同房间条件复用8句/4人，不作为独立句子样本。',now()]
    report=ROOT/'docs'/f'G21_{engine}识别覆盖.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive(engine.lower(),report,[folder/'COMPLETE.json',folder/'provenance.json',ROOT/'scripts/g21_asr.py'])
    del decode;gc.collect();torch.cuda.empty_cache()


def main():
    assert metadata(OUT/'SIGNAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    assert (OUT/'stages/next_ASR_decision/archive_manifest.json').exists()
    REC.mkdir(exist_ok=False)
    pins=['scripts/g21_asr.py','scripts/g21_asr_finish.py','scripts/g19_scores.py','scripts/g8_common.py',
          'src/mcse/asr_scoring.py','src/mcse/ctc.py','artifacts/g21_truth_covariance_v1/SIGNAL_ACCEPTANCE_AUDIT.json']
    plan={'frozen_utc':now(),'max_seconds':1800,'rows':864,'cases':432,'new_decoding_options':False,
          'signal_sha256':sha(OUT/'signals.jsonl'),'hashes':{n:sha(ROOT/n) for n in pins},
          'history_sha256':{e:{prefix+name:sha(base/e/name) for prefix,base in (('g18_',G18),('g19_',G19))
                              for name in ('provenance.json','asr_scores.jsonl','cache_new.jsonl')} for e in ('Whisper','CTC')}}
    dump(REC/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G21C_固定双识别器补查协议.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G21C 固定双识别器补查协议\n\n864个已保存真值协方差输出，两个后端各864条评分。Whisper tiny.en和贪心CTC完全沿用G19配置，复用相同参考缓存，不提供转写提示。固定1800秒共享作业上限。\n\n合并对照仅取同一432个带噪条件：历史31系统加两个真值统计系统，共33系统、14256配对/后端。微WER=总S+D+I除以总N；参考语音识别只每条件计一次。按RT、方向、噪声、场型、说话人和全部三训练种子记录，不选最佳种子。真值统计明确只作诊断，不视为上界或部署方法。\n')
    archive('recognition_protocol',report,[REC/'frozen_plan.json']+[ROOT/n for n in pins])
    rows=read(OUT/'signals.jsonl');assert len(rows)==864
    torch.set_num_threads(2);deadline=time.monotonic()+plan['max_seconds']-60
    for engine in ('Whisper','CTC'):run(engine,rows,plan,deadline)
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    from g21_asr_finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception as e:
        status(stage='asr_failed',error=repr(e));traceback.print_exc();raise
    finally:(OUT/'keep_awake_ASR_stop').write_text('G21 ASR job ended\n')
