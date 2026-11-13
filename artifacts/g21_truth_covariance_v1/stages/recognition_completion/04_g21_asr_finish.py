"""Join frozen old/new diagnostic scores and audit the paired comparison."""
import json
from collections import defaultdict
import numpy as np
from g21_truth_covariance import ROOT,OUT,sha,dump,read,now,archive
from g19_scores import load_cache,waveform
from mcse.asr_scoring import score_words,corpus_score

REC=OUT/'recognition'
G18=ROOT/'artifacts/g18_room_stress_v1/evaluation'
G19=ROOT/'artifacts/g19_room_mwf_v1/evaluation'


def metadata(p):return json.loads(p.read_text())


def finish():
    plan=metadata(REC/'frozen_plan.json');assert sha(OUT/'signals.jsonl')==plan['signal_sha256']
    assert (OUT/'stages/ctc/archive_manifest.json').exists()
    cases={r['sample_id']:r for r in read(ROOT/'data/g18_v1/manifest.jsonl') if r['condition']!='clean'}
    signals={}
    for file in (G19/'combined_signals.jsonl',OUT/'signals.jsonl'):
        for r in read(file):
            if r['sample_id'] in cases:
                key=r['system'],r['sample_id'];assert key not in signals;signals[key]=r
    assert len(signals)==14256 and len({s for s,_ in signals})==33
    keys={};refkeys={}
    for key,r in signals.items():
        if r['system'].startswith('truth_covariance_'):
            _,keys[key]=waveform(r['output_path'],r['output_sha256'])
            if r['sample_id'] not in refkeys:_,refkeys[r['sample_id']]=waveform(r['reference_path'],r['reference_sha256'])
    scores={};audit={}
    for engine in ('Whisper','CTC'):
        folder=REC/engine;complete=metadata(folder/'COMPLETE.json')
        assert sha(folder/'asr_scores.jsonl')==complete['asr_scores_sha256']
        assert sha(folder/'cache_new.jsonl')==complete['cache_new_sha256']
        prior=metadata(G19/engine/'provenance.json')
        cache,conflicts,sources=load_cache(engine,prior,G19/engine);assert not conflicts
        assert sources==metadata(folder/'provenance.json')['source_caches']
        for r in read(folder/'cache_new.jsonl'):
            assert r['sample_sha256'] not in cache and r['model_id']==prior['model_id']
            cache[r['sample_sha256']]=r['hypothesis']
        joined={};refs={}
        for tag,base in (('g18_',G18),('g19_',G19),('new_',REC)):
            file=base/engine/'asr_scores.jsonl'
            if tag!='new_':assert sha(file)==plan['history_sha256'][engine][tag+'asr_scores.jsonl']
            for r in read(file):
                if r['sample_id'] not in cases:continue
                k=r['system'],r['sample_id'];assert k in signals and k not in joined
                assert r['transcript']==cases[r['sample_id']]['transcript']
                if tag=='new_':
                    assert r['output_samples_sha256']==keys[k] and r['reference_samples_sha256']==refkeys[r['sample_id']]
                    assert r['hypothesis']==cache[keys[k]] and r['reference_hypothesis']==cache[refkeys[r['sample_id']]]
                    assert r['oracle_clean_information'] and not r['deployable']
                    assert r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis'])
                if r['sample_id'] in refs:assert refs[r['sample_id']]==r['reference_score']
                else:refs[r['sample_id']]=r['reference_score']
                joined[k]=r['score']
        assert set(joined)==set(signals) and len(refs)==432
        scores[engine]=joined
        audit[engine]={'combined_rows':len(joined),'new_rows':864,'new_scores_and_cache_verified':True,
                       'cache_sha256':complete['cache_new_sha256'],'score_sha256':complete['asr_scores_sha256']}
        dump(folder/'reference_summary.json',{str(rt):corpus_score(v for sid,v in refs.items() if cases[sid]['requested_rt60']==rt) for rt in (0,.15,.35)},True)
    registry=metadata(ROOT/'artifacts/g18_room_stress_v1/systems.json')
    system_groups={r['system']:r.get('geometry',r['kind']) for r in registry['direct']}
    system_groups.update({r['system']:system_groups[r['mask_system']] for r in registry['beam']})
    for name in list(system_groups):
        if name!='noisy' and not name.endswith('_mask_mvdr'):system_groups[name+'_mask_mwf']=system_groups[name]
    system_groups.update(truth_covariance_mvdr='truth_covariance',truth_covariance_mwf='truth_covariance')
    def method(name):
        if name.endswith('_mvdr'):return 'mvdr'
        if name.endswith('_mwf'):return 'mwf'
        return 'direct'
    def aggregate(pairs):
        return {'rows':len(pairs),**{m:float(np.mean([signals[k]['metrics'][m] for k in pairs])) for m in ('si_sdr_db','pesq_wb','stoi')},
                **{e:corpus_score(scores[e][k] for k in pairs) for e in scores}}
    dimensions={'by_rt':('rt','group','method'),'by_system':('rt','system'),
                'by_direction':('rt','group','method','angle'),'by_speaker':('rt','group','method','speaker'),
                'by_field':('rt','group','method','field','relative_angle'),'by_noise':('rt','group','method','noise')}
    result={}
    for label,fields in dimensions.items():
        buckets=defaultdict(list)
        for k,r in signals.items():
            attrs={'rt':r['requested_rt60'],'group':system_groups[r['system']],'method':method(r['system']),
                   'angle':r['target_azimuth'],'speaker':r['speaker_id'],'field':r['noise_field'],
                   'relative_angle':r['relative_angle'],'noise':r['noise_id'],'system':r['system']}
            buckets[tuple(attrs[f] for f in fields)].append(k)
        result[label]=[{**dict(zip(fields,key)),**aggregate(pairs)} for key,pairs in buckets.items()]
    dump(REC/'summary.json',result,True)
    report=ROOT/'docs/G21_真值统计与识别保真综合报告.md'
    lines=['# G21 真值统计与识别保真综合报告','','全部33系统在相同432个带噪条件中配对，共14256记录/识别器；仅新增864条真值统计输出及各后端864条评分。历史评分原样复用。真值统计仅供诊断，不能部署或作为理论上界。','',
           '|请求RT|前端组|输出方式|记录数|SI-SDR dB|PESQ|STOI|Whisper WER|CTC WER|','|---:|---|---|---:|---:|---:|---:|---:|---:|']
    ordered=sorted(result['by_rt'],key=lambda r:(r['rt'],r['group'],r['method']))
    for r in ordered:lines.append(f"|{r['rt']}|{r['group']}|{r['method']}|{r['rows']}|{r['si_sdr_db']:.2f}|{r['pesq_wb']:.3f}|{r['stoi']:.3f}|{r['Whisper']['wer']:.2%}|{r['CTC']['wer']:.2%}|")
    lines+=['','WER为总S+D+I除以总参考词数；未截断超过100%的值。固定、反向和正确刚性训练组保留所有三个种子，各RT每种输出432条；原McNet、带噪及真值统计每RT各144条。记录重复，不是独立语句样本。','',
            '按系统（包括全部种子）、方向、说话人、空间场型和噪声的细分结果见 recognition/summary.json；参考识别只每条件计一次，见各后端reference_summary.json。','',
            '真值统计的改善描述替换统计信息后的响应，不能单独因果归于网络掩码，也不能外推到新房间、真实阵列或中文。有限语音/噪声交叉统计并不为零，滤波结果不保证理论最优。没有新的人工逐词或盲听标签。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    dump(REC/'completion_audit.json',{'status':'ok','completed_utc':now(),'cases':432,'systems':33,'rows_per_ASR':14256,
          'new_rows_per_ASR':864,'ASR':audit,'signal_sha256':sha(OUT/'signals.jsonl'),'summary_sha256':sha(REC/'summary.json')},True)
    archive('recognition_completion',report,[REC/'completion_audit.json',REC/'summary.json',REC/'frozen_plan.json',ROOT/'scripts/g21_asr_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','stage':'G21','utc':now(),'report':str(report.relative_to(ROOT)),
          'report_sha256':sha(report),'archive_sha256':sha(OUT/'stages/recognition_completion/archive_manifest.json')},True)
    dump(OUT/'status.json',{'stage':'complete_pending_readback','utc':now()})
