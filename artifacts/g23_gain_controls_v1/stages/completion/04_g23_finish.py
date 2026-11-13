"""Paired scalar-gain readback, with no best-gain optimization."""
from collections import defaultdict
from g23_prepare import ROOT,OUT,G22,sha,dump,read,now,archive
from g23_asr import meta,FIELDS,status
from g19_scores import load_cache,waveform
from mcse.asr_scoring import score_words,corpus_score,normalize_english


def family(name):
    if name in ('noisy','clean_reference'):return name
    if name.startswith('truth_covariance_'):return name
    return 'rigid_'+('mvdr' if name.endswith('_mvdr') else 'mwf')


def finish():
    rec=OUT/'recognition';assert (OUT/'stages/ctc/archive_manifest.json').exists()
    plan=meta(rec/'frozen_plan.json');assert sha(OUT/'manifest.jsonl')==plan['manifest_sha256']
    rows=read(OUT/'manifest.jsonl');signals={(r['base_system'],r['sample_id'],r['gain_db']):r for r in rows}
    assert len(rows)==len(signals)==1440
    sample_keys={};ref_keys={}
    for k,r in signals.items():
        _,sample_keys[k]=waveform(r['output_path'],r['output_sha256'])
        if r['sample_id'] not in ref_keys:_,ref_keys[r['sample_id']]=waveform(r['reference_path'],r['reference_sha256'])
    scores={};audits={};details=[]
    for engine in ('Whisper','CTC'):
        folder=rec/engine;complete=meta(folder/'COMPLETE.json');prior=meta(G22/'recognition'/engine/'provenance.json')
        assert sha(folder/'asr_scores.jsonl')==complete['scores_sha256'] and sha(folder/'cache_new.jsonl')==complete['cache_sha256']
        cache,conflicts,sources=load_cache(engine,prior,G22/'recognition'/engine);assert not conflicts
        assert sources==meta(folder/'provenance.json')['source_caches']
        new=read(folder/'cache_new.jsonl');assert len(new)==complete['new_decodes']
        for r in new:
            assert r['sample_sha256'] not in cache and r['model_id']==prior['model_id'] and r['gain_db_first_seen'] in (-6,-12)
            cache[r['sample_sha256']]=r['hypothesis']
        joined={}
        for r in read(folder/'asr_scores.jsonl'):
            k=r['base_system'],r['sample_id'],r['gain_db'];assert k in signals and k not in joined
            assert all(r[f]==signals[k][f] for f in FIELDS)
            assert r['output_samples_sha256']==sample_keys[k] and r['reference_samples_sha256']==ref_keys[r['sample_id']]
            assert r['hypothesis']==cache[sample_keys[k]] and r['reference_hypothesis']==cache[ref_keys[r['sample_id']]]
            assert r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis'])
            joined[k]=r
        assert set(joined)==set(signals) and complete['zero_db_cache_hits']==480
        scores[engine]=joined
        for (system,sid,db),r in joined.items():
            if db==0:continue
            base=joined[system,sid,0];a,b=base['score'],r['score']
            details.append({'engine':engine,'base_system':system,'sample_id':sid,'gain_db':db,
                'room_id':r['room_id'],'rt':r['requested_rt60'],'speaker_id':r['speaker_id'],'angle':r['target_azimuth'],
                'normalized_hypothesis_changed':normalize_english(r['hypothesis'])!=normalize_english(base['hypothesis']),
                'raw_hypothesis_changed':r['hypothesis']!=base['hypothesis'],
                **{k+'_delta':b[k]-a[k] for k in ('S','D','I','errors')},'N':a['N'],
                'baseline_hypothesis':base['hypothesis'],'gain_hypothesis':r['hypothesis']})
        audits[engine]={'rows':1440,'zero_db_cache_hits':480,'new_decodes':len(new),'all_alignments_recomputed':True,
                        'score_sha256':complete['scores_sha256'],'cache_sha256':complete['cache_sha256']}
    def aggregate(keys):
        return {'rows':len(keys),**{e:corpus_score(scores[e][k]['score'] for k in keys) for e in scores}}
    dims={'by_family_gain':('family','gain_db'),'by_system_gain':('base_system','gain_db'),
          'by_room_rt':('room_id','rt','family','gain_db'),'by_speaker':('speaker_id','family','gain_db'),
          'by_angle':('angle','family','gain_db')}
    summary={}
    for label,fields in dims.items():
        groups=defaultdict(list)
        for k,r in signals.items():
            a={**r,'rt':r['requested_rt60'],'angle':r['target_azimuth'],'family':family(r['base_system'])}
            groups[tuple(a[f] for f in fields)].append(k)
        summary[label]=[{**dict(zip(fields,key)),**aggregate(keys)} for key,keys in groups.items()]
    changes=[]
    for engine in scores:
        for fam in sorted({family(r['base_system']) for r in rows}):
            for db in (-6,-12):
                ds=[d for d in details if d['engine']==engine and family(d['base_system'])==fam and d['gain_db']==db]
                assert ds
                changes.append({'engine':engine,'family':fam,'gain_db':db,'pairs':len(ds),
                    'normalized_hypothesis_changes':sum(d['normalized_hypothesis_changed'] for d in ds),
                    'raw_hypothesis_changes':sum(d['raw_hypothesis_changed'] for d in ds),
                    **{k:sum(d[k] for d in ds) for k in ('S_delta','D_delta','I_delta','errors_delta','N')}})
    summary['gain_changes']=changes
    dump(rec/'summary.json',summary,True);dump(rec/'paired_changes.json',details,True)
    report=ROOT/'docs/G23_固定增益与双识别器敏感性报告.md'
    lines=['# G23 固定增益与双识别器敏感性报告','','本实验只把已有波形乘固定正标量；所有0dB假设复用G22原记录。没有重新增强、修复或训练，未挑选最优增益。波形结构控制与非退化SI-SDR保持检查先于解码完成。','',
           '|输出类别|记录数/增益|增益dB|Whisper WER|CTC WER|','|---|---:|---:|---:|---:|']
    for r in sorted(summary['by_family_gain'],key=lambda r:(r['family'],-r['gain_db'])):
        lines.append(f"|{r['family']}|{r['rows']}|{r['gain_db']}|{r['Whisper']['wer']:.2%}|{r['CTC']['wer']:.2%}|")
    lines+=['','以下与同一系统、同一条件的0dB配对；文本变化先做既定英语规范化，不把纯标点变化算作词内容变化。','',
            '|识别器|类别|增益|配对数|规范化假设变化数|ΔS|ΔD|ΔI|Δ错误数|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in changes:lines.append(f"|{r['engine']}|{r['family']}|{r['gain_db']}|{r['pairs']}|{r['normalized_hypothesis_changes']}|{r['S_delta']}|{r['D_delta']}|{r['I_delta']}|{r['errors_delta']}|")
    lines+=['','不同输出类别没有相同的重复模型数：正确刚性组包含全部3种子，其他控制一个系统。WER均为总S+D+I/总N，不截断超过100%的值。逐系统、房间/RT、说话人和方向数据与全部配对文本保留。','',
            '若固定增益改变了ASR文本，说明本评估链对幅度敏感，不能把这种变化直接当成人声内容被删除或恢复。若未变化，也只说明本次条件与增益范围下的稳定性。没有听感不变、最佳部署增益、真实场景或统计显著性结论。仅用OOFFICE点源相对−80的48条件，是探索性评估器控制。',
            '', '原波形全部保留。降低播放电平可能影响人的可听度；数学上相对结构保持不代表固定音量下人耳可懂度保持。真值统计与干净参考始终仅作控制，TSOS未在此实现。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    dump(rec/'completion_audit.json',{'status':'ok','utc':now(),'rows_each_ASR':1440,'paired_nonzero_each_ASR':960,
          'ASR':audits,'summary_sha256':sha(rec/'summary.json'),'paired_changes_sha256':sha(rec/'paired_changes.json')},True)
    archive('completion',report,[rec/'completion_audit.json',rec/'summary.json',rec/'frozen_plan.json',ROOT/'scripts/g23_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','utc':now(),'report':str(report.relative_to(ROOT)),
          'report_sha256':sha(report),'archive_sha256':sha(OUT/'stages/completion/archive_manifest.json')},True)
    status(stage='complete_pending_readback')
