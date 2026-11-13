"""Full paired raw/level-controlled ASR audit without rewriting either regime."""
from collections import defaultdict
from g26_prepare import ROOT,OUT,SOURCE,sha,read,dump,now,archive
from g26_asr import meta,FIELDS,seed_cache,status
from mcse.asr_scoring import score_words,corpus_score,normalize_english


def finish():
    rec=OUT/'recognition';plan=meta(rec/'frozen_plan.json')
    assert (OUT/'stages/ctc/archive_manifest.json').exists() and sha(OUT/'manifest.jsonl')==plan['manifest_sha256']
    inputs={(r['system'],r['sample_id']):r for r in read(OUT/'manifest.jsonl')};assert len(inputs)==16488
    quality={(r['system'],r['sample_id']):r for r in read(SOURCE/'evaluation/combined_signals.jsonl')};assert set(quality)==set(inputs)
    all_scores={};ref_scores={};audits={};details=[]
    for e in ('Whisper','CTC'):
        folder=rec/e;marker=meta(folder/'COMPLETE.json');current=meta(folder/'provenance.json');prior,cache=seed_cache(e,plan)
        assert sha(folder/'asr_scores.jsonl')==marker['score_sha256'] and sha(folder/'cache_new.jsonl')==marker['cache_sha256']
        assert all(current[k]==prior[k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
        new=read(folder/'cache_new.jsonl');assert len(new)==sum(marker['new_decodes'].values())
        for r in new:
            assert r['sample_sha256'] not in cache and r['model_id']==prior['model_id']
            assert r['normalization_id']==current['input_preprocessing']['preprocessing_id']
            cache[r['sample_sha256']]=r['hypothesis']
        assert len(cache)==marker['final_cache_entries']
        raw_file=SOURCE/'recognition'/e/'asr_scores.jsonl';assert sha(raw_file)==plan['raw_comparators'][e]
        raw={(r['system'],r['sample_id']):r for r in read(raw_file)};assert set(raw)==set(inputs)
        normalized={};refs={}
        for r in read(folder/'asr_scores.jsonl'):
            k=r['system'],r['sample_id'];assert k in inputs and k not in normalized
            assert all(r[f]==inputs[k][f] for f in FIELDS)
            assert r['output_samples_sha256']==inputs[k]['normalized_output_samples_sha256']
            assert r['reference_samples_sha256']==inputs[k]['normalized_reference_samples_sha256']
            assert r['raw_output_samples_sha256']==raw[k]['output_samples_sha256']==inputs[k]['raw_output_samples_sha256']
            assert r['hypothesis']==cache[r['output_samples_sha256']] and r['reference_hypothesis']==cache[r['reference_samples_sha256']]
            assert r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis'])
            assert r['score']['N']==raw[k]['score']['N']
            item=(r['reference_score'],raw[k]['reference_score'])
            if r['sample_id'] in refs:assert refs[r['sample_id']]==item
            else:refs[r['sample_id']]=item
            normalized[k]=r
            details.append({'engine':e,'system':k[0],'sample_id':k[1],'room_id':r['room_id'],'rt':r['requested_rt60'],
                'normalized_hypothesis_changed':normalize_english(raw[k]['hypothesis'])!=normalize_english(r['hypothesis']),
                **{m+'_delta':r['score'][m]-raw[k]['score'][m] for m in ('S','D','I','errors')},'N':r['score']['N']})
        assert set(normalized)==set(inputs) and len(refs)==504
        all_scores[e]={'raw':raw,'level_control':normalized};ref_scores[e]=refs
        audits[e]={'rows':16488,'new_decodes':marker['new_decodes'],'normalized_cache_entries':len(cache),
                   'score_sha256':marker['score_sha256'],'cache_sha256':marker['cache_sha256'],'alignments_recomputed':16488}
    registry=meta(SOURCE/'signal_systems.json');groups={r['system']:r.get('geometry',r['kind']) for r in registry['direct']}
    for name in list(groups):
        if name!='noisy':
            for m in ('mvdr','mwf'):groups[name+'_mask_'+m]=groups[name]
    groups.update(truth_covariance_mvdr='truth_covariance',truth_covariance_mwf='truth_covariance')
    def method(n):return 'mvdr' if n.endswith('_mvdr') else 'mwf' if n.endswith('_mwf') else 'direct'
    def aggregate(keys,regime):
        values={'rows':len(keys),**{m:sum(quality[k]['metrics'][m] for k in keys)/len(keys) for m in ('si_sdr_db','pesq_wb','stoi')},
                **{e:corpus_score(all_scores[e][regime][k]['score'] for k in keys) for e in all_scores}}
        if all('clean_levels' in quality[k] for k in keys):
            values['raw_clean_levels']={m:sum(quality[k]['clean_levels'][m] for k in keys)/len(keys) for m in ('rms_gain_ratio','reference_normalized_squared_error')}
        return values
    dimensions={'by_rt':('regime','room','rt','group','method','clean'),
                'by_system':('regime','room','rt','system','clean'),
                'by_direction':('regime','room','rt','group','method','angle','clean'),
                'by_noise':('regime','room','rt','group','method','noise'),
                'by_field':('regime','room','rt','group','method','field','relative_angle'),
                'by_speaker':('regime','room','rt','group','method','speaker','clean')}
    result={}
    for section,fields in dimensions.items():
        buckets=defaultdict(list)
        for key,r in inputs.items():
            for regime in ('raw','level_control'):
                a={'regime':regime,'room':r['room_id'],'rt':r['requested_rt60'],'group':groups[r['system']],
                   'method':method(r['system']),'clean':r['condition']=='clean','system':r['system'],'angle':r['target_azimuth'],
                   'noise':r['noise_id'],'field':r['noise_field'],'relative_angle':r['relative_angle'],'speaker':r['speaker_id']}
                buckets[tuple(a[f] for f in fields)].append(key)
        result[section]=[{**dict(zip(fields,k)),**aggregate(keys,k[fields.index('regime')])} for k,keys in buckets.items()]
    by={(r['regime'],r['room'],r['rt'],r['group'],r['method'],r['clean']):r for r in result['by_rt']}
    comparisons=[]
    for room in ('reference','new'):
        for rt in (0,.15,.35):
            for group in ('mcnet','fixed','rotated','rigid','truth_covariance'):
                entry={'room':room,'rt':rt,'group':group}
                for regime in ('raw','level_control'):
                    a,b=by[regime,room,rt,group,'mvdr',False],by[regime,room,rt,group,'mwf',False]
                    assert a['rows']==b['rows']
                    entry[regime]={e+'_MWF_minus_MVDR_pp':100*(b[e]['wer']-a[e]['wer']) for e in all_scores}
                comparisons.append(entry)
    result['method_comparisons']=comparisons
    references=[]
    for room in ('reference','new'):
        for rt in (0,.15,.35):
            for clean in (False,True):
                ids={r['sample_id'] for r in inputs.values() if r['room_id']==room and r['requested_rt60']==rt and (r['condition']=='clean')==clean}
                references.append({'room':room,'rt':rt,'clean':clean,'cases':len(ids),
                    **{e:{regime:corpus_score(ref_scores[e][sid][0 if regime=='level_control' else 1] for sid in ids) for regime in ('raw','level_control')} for e in all_scores}})
    result['references']=references
    pairs=defaultdict(dict)
    for key,r in inputs.items():
        if r['requested_rt60']==0:pairs[r['system'],r['matched_room_pair']][r['room_id']]=key
    assert len(pairs)==2748
    for entries in pairs.values():
        a,b=entries['reference'],entries['new']
        assert inputs[a]['normalized_output_samples_sha256']==inputs[b]['normalized_output_samples_sha256']
        assert all(all_scores[e]['level_control'][a]['score']==all_scores[e]['level_control'][b]['score'] for e in all_scores)
    dump(rec/'summary.json',result,True);dump(rec/'raw_to_level_changes.json',details,True)
    report=ROOT/'docs/G26_完整双房间原始与电平控制识别对照.md'
    lines=['# G26 完整双房间原始与电平控制识别对照','','全部16488条输出和每后端评分完成。波形/模型/原质量指标不变，只增加固定规则的ASR输入副本条件。下表仅带噪，斜线前后为raw/电平控制，均为相同系统与输入条件配对；真值统计不作部署或理论上界。','',
           '|房间|请求RT|前端组|输出|记录数|Whisper WER|CTC WER|','|---|---:|---|---|---:|---:|---:|']
    for (regime,room,rt,g,m,clean),a in sorted(by.items()):
        if regime!='raw' or clean:continue
        b=by['level_control',room,rt,g,m,clean]
        lines.append(f"|{room}|{rt}|{g}|{m}|{a['rows']}|{a['Whisper']['wer']:.2%} / {b['Whisper']['wer']:.2%}|{a['CTC']['wer']:.2%} / {b['CTC']['wer']:.2%}|")
    flips={e:sum(r['raw'][e+'_MWF_minus_MVDR_pp']*r['level_control'][e+'_MWF_minus_MVDR_pp']<0 for r in comparisons) for e in all_scores}
    lines+=['',f"30个房间×RT×前端类别的MWF−MVDR比较中，严格符号翻转数为：Whisper {flips['Whisper']}，CTC {flips['CTC']}。等分不计翻转；各比较共享四人四句，不作独立统计检验。",'',
            '原始SI-SDR/PESQ/STOI、干净RMS/NMSE没有因ASR输入预处理而改变，仍见原始信号与本阶段汇总。干净控制、两类参考识别、全部系统/种子、方向、噪声、场型和说话人细分已保存；没有按有利结果剔除条件或选种子。2748对RT0的规范化输入与对应ASR计分均一致。','',
            '预处理规则固定于看G26分数之前，1015个输出记录受峰值约束，RMS可能低于目标。.05不是经本数据优化得到的最优电平。新分数属于另一评估条件，不能替换raw表或证明增强模型改善。此后验辅评分仍使用既有四句，只能帮助判断旧结论对电平条件的稳健性，不能声称新语音、真实阵列或广泛泛化。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    audit={'status':'ok','utc':now(),'rows_each_ASR':16488,'references':504,'RT0_pairs':2748,'ASR':audits,
           'method_sign_flips':flips,'summary_sha256':sha(rec/'summary.json'),'changes_sha256':sha(rec/'raw_to_level_changes.json')}
    dump(rec/'completion_audit.json',audit,True)
    archive('completion',report,[rec/'completion_audit.json',rec/'summary.json',rec/'frozen_plan.json',ROOT/'scripts/g26_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','utc':now(),'report':str(report.relative_to(ROOT)),
          'report_sha256':sha(report),'archive_sha256':sha(OUT/'stages/completion/archive_manifest.json')},True)
    status(stage='complete_pending_readback')
