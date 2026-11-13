"""Audit and summarize matched room predictions and fixed-recognizer scores."""
import json
from collections import defaultdict
import numpy as np
from g22_common import ROOT,OUT,BASE,sha,read,dump,now,archive,metadata,guard,status
from g19_scores import load_cache,waveform
from mcse.asr_scoring import score_words,corpus_score

REC=OUT/'recognition';PRIOR=ROOT/'artifacts/g21_truth_covariance_v1/recognition'


def finish():
    guard();assert (OUT/'stages/ctc/archive_manifest.json').exists()
    plan=metadata(REC/'frozen_plan.json');path=OUT/'evaluation/combined_signals.jsonl'
    assert sha(path)==plan['signal_sha256']
    rows=read(path);signals={(r['system'],r['sample_id']):r for r in rows};assert len(rows)==len(signals)==16488
    cases={r['sample_id']:r for r in read(BASE/'manifest.jsonl')}
    registry=metadata(OUT/'signal_systems.json')
    direct={r['system']:r for r in registry['direct']};frontends=[n for n in direct if n!='noisy']
    names=['noisy']+frontends+[n+'_mask_'+m for n in frontends for m in ('mvdr','mwf')]
    expected={(n,s) for n in names for s in cases}|{('truth_covariance_'+m,s) for s,c in cases.items() if c['condition']!='clean' for m in ('mvdr','mwf')}
    assert set(signals)==expected
    sample_keys={};reference_keys={}
    for k,r in signals.items():
        _,sample_keys[k]=waveform(r['output_path'],r['output_sha256'])
        if r['sample_id'] not in reference_keys:_,reference_keys[r['sample_id']]=waveform(r['reference_path'],r['reference_sha256'])
    scores={};references={};audits={}
    for engine in ('Whisper','CTC'):
        folder=REC/engine;complete=metadata(folder/'COMPLETE.json');provenance=metadata(folder/'provenance.json')
        assert sha(folder/'cache_new.jsonl')==complete['cache_new_sha256']
        assert sha(folder/'asr_scores.jsonl')==complete['asr_scores_sha256']
        prior=metadata(PRIOR/engine/'provenance.json');cache,conflicts,sources=load_cache(engine,prior,PRIOR/engine)
        assert not conflicts and sources==provenance['source_caches']
        assert all(provenance[k]==prior[k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
        new=read(folder/'cache_new.jsonl')
        assert len(new)==sum(complete['new_decodes'].values()) and len({r['sample_sha256'] for r in new})==len(new)
        for r in new:
            assert r['sample_sha256'] not in cache and r['model_id']==prior['model_id']
            cache[r['sample_sha256']]=r['hypothesis']
        assert len(cache)==complete['cache_entries']
        joined={};refs={}
        from g22_asr import FIELDS
        for r in read(folder/'asr_scores.jsonl'):
            k=r['system'],r['sample_id'];assert k in signals and k not in joined
            assert all(r[field]==signals[k][field] for field in FIELDS)
            key,rkey=sample_keys[k],reference_keys[r['sample_id']]
            assert r['output_samples_sha256']==key and r['reference_samples_sha256']==rkey
            assert r['hypothesis']==cache[key] and r['reference_hypothesis']==cache[rkey]
            assert r['score']==score_words(r['transcript'],r['hypothesis'])
            assert r['reference_score']==score_words(r['transcript'],r['reference_hypothesis'])
            if r['sample_id'] in refs:assert refs[r['sample_id']]==r['reference_score']
            else:refs[r['sample_id']]=r['reference_score']
            joined[k]=r['score']
        assert set(joined)==expected and set(refs)==set(cases)
        scores[engine]=joined;references[engine]=refs
        audits[engine]={'rows':len(joined),'new_decodes':complete['new_decodes'],'new_cache_entries':len(new),
                        'references_checked':504,'all_alignments_recomputed':True,
                        'score_sha256':complete['asr_scores_sha256'],'cache_sha256':complete['cache_new_sha256']}
    group_of={n:r.get('geometry',r['kind']) for n,r in direct.items()}
    for name in frontends:
        for method in ('mvdr','mwf'):group_of[name+'_mask_'+method]=group_of[name]
    group_of.update(truth_covariance_mvdr='truth_covariance',truth_covariance_mwf='truth_covariance')
    def method(name):
        if name.endswith('_mvdr'):return 'mvdr'
        if name.endswith('_mwf'):return 'mwf'
        return 'direct'
    def aggregate(keys):
        values={'rows':len(keys),**{m:float(np.mean([signals[k]['metrics'][m] for k in keys])) for m in ('si_sdr_db','pesq_wb','stoi')},
                **{e:corpus_score(scores[e][k] for k in keys) for e in scores}}
        if all('clean_levels' in signals[k] for k in keys):
            values['clean_levels']={m:float(np.mean([signals[k]['clean_levels'][m] for k in keys])) for m in ('rms_gain_ratio','reference_normalized_squared_error')}
        return values
    definitions={'by_rt':('room','rt','group','method','clean'),
                 'by_system':('room','rt','system','clean'),
                 'by_direction':('room','rt','group','method','angle','clean'),
                 'by_speaker':('room','rt','group','method','speaker','clean'),
                 'by_field':('room','rt','group','method','field','relative_angle'),
                 'by_noise':('room','rt','group','method','noise')}
    result={}
    for label,fields in definitions.items():
        buckets=defaultdict(list)
        for k,r in signals.items():
            attrs={'room':r['room_id'],'rt':r['requested_rt60'],'group':group_of[r['system']],'method':method(r['system']),
                   'system':r['system'],'angle':r['target_azimuth'],'speaker':r['speaker_id'],
                   'clean':r['condition']=='clean','field':r['noise_field'],'relative_angle':r['relative_angle'],'noise':r['noise_id']}
            buckets[tuple(attrs[f] for f in fields)].append(k)
        result[label]=[{**dict(zip(fields,key)),**aggregate(keys)} for key,keys in buckets.items()]
    refs=[]
    for room in ('reference','new'):
        for rt in (0,.15,.35):
            for clean in (False,True):
                selected=[sid for sid,c in cases.items() if c['room_id']==room and c['requested_rt60']==rt and (c['condition']=='clean')==clean]
                refs.append({'room':room,'rt':rt,'clean':clean,'cases':len(selected),
                             **{e:corpus_score(references[e][s] for s in selected) for e in references}})
    result['references']=refs
    grouped={(r['room'],r['rt'],r['group'],r['method'],r['clean']):r for r in result['by_rt']}
    differences=[]
    for (room,rt,g,m,clean),a in grouped.items():
        if room!='reference':continue
        b=grouped['new',rt,g,m,clean];assert a['rows']==b['rows'] and all(a[e]['N']==b[e]['N'] for e in scores)
        differences.append({'rt':rt,'group':g,'method':m,'clean':clean,'rows_per_room':a['rows'],
                            **{k:b[k]-a[k] for k in ('si_sdr_db','pesq_wb','stoi')},
                            **{e+'_WER_pp':100*(b[e]['wer']-a[e]['wer']) for e in scores}})
    result['new_minus_reference']=differences
    pairs=defaultdict(dict)
    for k,r in signals.items():pairs[r['system'],r['matched_room_pair']][r['room_id']]=k
    assert len(pairs)==8244 and all(set(v)=={'reference','new'} for v in pairs.values())
    rt0=0;exact_pcm_rt0=0
    with (REC/'paired_room_deltas.jsonl').open('x',encoding='utf-8') as log:
        for (name,pair),entries in pairs.items():
            a,b=entries['reference'],entries['new'];ra,rb=signals[a],signals[b]
            if ra['requested_rt60']==0:
                rt0+=1
                if sample_keys[a]==sample_keys[b]:
                    assert all(scores[e][a]==scores[e][b] for e in scores);exact_pcm_rt0+=1
            row={'system':name,'matched_room_pair':pair,'reference_sample_id':a[1],'new_sample_id':b[1],
                 'rt':ra['requested_rt60'],'source_id':ra['source_id'],'speaker_id':ra['speaker_id'],
                 **{k:rb['metrics'][k]-ra['metrics'][k] for k in ('si_sdr_db','pesq_wb','stoi')},
                 **{e+'_errors_delta':scores[e][b]['errors']-scores[e][a]['errors'] for e in scores}}
            log.write(json.dumps(row,allow_nan=False)+'\n')
    assert rt0==2748
    dump(REC/'summary.json',result,True)
    report=ROOT/'docs/G22_双房间配对信号与识别综合报告.md'
    lines=['# G22 双房间配对信号与识别综合报告','','全部16488条信号及每后端评分完成，8244对房间输出按同一源语音和噪声条件配对。参数、模型和全部训练种子固定。每房间包含252条件，4位已用说话人各一句；不是新说话人或真实阵列测试。','',
           '以下仅带噪条件，斜线前后为原房间/新房间。真值系统不可部署，也不是理论上界。','',
           '|请求RT|前端组|输出方式|记录数/房间|SI-SDR dB|Whisper WER|CTC WER|','|---:|---|---|---:|---:|---:|---:|']
    for (room,rt,g,m,clean),a in sorted(grouped.items()):
        if room!='reference' or clean:continue
        b=grouped['new',rt,g,m,clean]
        lines.append(f"|{rt}|{g}|{m}|{a['rows']}|{a['si_sdr_db']:.2f} / {b['si_sdr_db']:.2f}|{a['Whisper']['wer']:.2%} / {b['Whisper']['wer']:.2%}|{a['CTC']['wer']:.2%} / {b['CTC']['wer']:.2%}|")
    lines+=['','微平均WER=(总S+D+I)/总参考词数，保留超过100%的值。干净控制、PESQ、STOI、RMS/NMSE、分系统（包含全部种子）、方向、说话人和场型见 recognition/summary.json。逐案例房间差见 paired_room_deltas.jsonl。参考按每条件计一次，未缓存的新房间参考实际解码。','',
            f'2748对RT0输出中有{exact_pcm_rt0}对PCM哈希完全一致，对应两个后端计分均相同。',
            '', '新旧房间共用防削波尺度，原房间重渲染，因此不直接套用G18/G19旧成绩。请求RT相同不意味着实测T20相同；结论包含房间边界与实际反射统计变化，不能独立归因于某个尺寸。四人四句和重复条件不当独立大样本，不据此声称显著性、稳定吞字或普遍房间泛化。没有新增训练或人工逐词标签。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    audit={'status':'ok','completed_utc':now(),'signals':16488,'ASR_rows_each':16488,'paired_outputs':8244,
           'RT0_pairs':rt0,'RT0_exact_PCM_pairs':exact_pcm_rt0,'ASR':audits,
           'summary_sha256':sha(REC/'summary.json'),'paired_deltas_sha256':sha(REC/'paired_room_deltas.jsonl')}
    dump(REC/'completion_audit.json',audit,True)
    archive('recognition_completion',report,[REC/'completion_audit.json',REC/'summary.json',REC/'frozen_plan.json',ROOT/'scripts/g22_asr_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','utc':now(),'stage':'G22','report':str(report.relative_to(ROOT)),
          'report_sha256':sha(report),'archive_sha256':sha(OUT/'stages/recognition_completion/archive_manifest.json')},True)
    status(stage='complete_pending_readback')
