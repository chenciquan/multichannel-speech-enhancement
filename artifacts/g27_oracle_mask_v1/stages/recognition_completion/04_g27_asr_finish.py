"""Complete oracle-mask recognition accounting and provenance acceptance."""
from collections import defaultdict
import numpy as np
from g27_signal import ROOT,OUT,SOURCE,sha,read,dump,now,meta,archive
from g27_asr import REC,LEVEL,FIELDS,seed
from mcse.asr_scoring import score_words,corpus_score


def finish():
    plan=meta(REC/'frozen_plan.json');assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    inputs={(r['system'],r['sample_id']):r for r in read(REC/'input_manifest.jsonl')};assert len(inputs)==864
    signals={(r['system'],r['sample_id']):r for r in read(OUT/'signals.jsonl')};assert set(signals)==set(inputs)
    scored={};old={};checks={}
    for engine in plan['engines']:
        scored[engine]={};old[engine]={};checks[engine]={}
        for regime in plan['regimes']:
            folder=REC/engine/regime;marker=meta(folder/'COMPLETE.json');prior,cache=seed(engine,regime,plan)
            p=meta(folder/'provenance.json');assert all(p[k]==prior[k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
            assert p['ASR_regime']==regime and p['input_preprocessing']==(plan['normalization'] if regime=='level_control' else None)
            assert sha(folder/'asr_scores.jsonl')==marker['score_sha256'] and sha(folder/'cache_new.jsonl')==marker['cache_sha256']
            new=read(folder/'cache_new.jsonl');assert len(new)==marker['fresh_decodes'] and marker['reference_new']==0
            for r in new:
                assert r['sample_sha256'] not in cache and r['model_id']==prior['model_id'] and r['ASR_regime']==regime
                assert r['normalization_id']==(plan['normalization']['preprocessing_id'] if regime=='level_control' else None)
                cache[r['sample_sha256']]=r['hypothesis']
            scores={};refs={}
            for r in read(folder/'asr_scores.jsonl'):
                key=r['system'],r['sample_id'];assert key in inputs and key not in scores
                assert all(r[k]==inputs[key][k] for k in FIELDS) and r['ASR_regime']==regime
                assert r['output_samples_sha256']==inputs[key]['inputs'][regime]['output_key']
                assert r['reference_samples_sha256']==inputs[key]['inputs'][regime]['reference_key']
                assert r['hypothesis']==cache[r['output_samples_sha256']] and r['reference_hypothesis']==cache[r['reference_samples_sha256']]
                assert r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis'])
                value=r['reference_samples_sha256'],r['reference_hypothesis'],r['reference_score']
                if r['sample_id'] in refs:assert refs[r['sample_id']]==value
                else:refs[r['sample_id']]=value
                scores[key]=r
            assert set(scores)==set(inputs) and len(refs)==432
            scored[engine][regime]=scores
            base=SOURCE if regime=='raw' else LEVEL
            old[engine][regime]={(r['system'],r['sample_id']):r for r in read(base/'recognition'/engine/'asr_scores.jsonl')}
            checks[engine][regime]={'rows':864,'references':432,'fresh_decodes':len(new),'reference_new':0,
                'score_sha256':marker['score_sha256'],'cache_sha256':marker['cache_sha256']}
    zero=defaultdict(dict)
    for key,r in inputs.items():
        if r['requested_rt60']==0:zero[r['system'],r['matched_room_pair']][r['room_id']]=key
    assert len(zero)==144
    for pair in zero.values():
        for engine in plan['engines']:
            for regime in plan['regimes']:
                a,b=(scored[engine][regime][pair[room]] for room in ('reference','new'))
                assert a['output_samples_sha256']==b['output_samples_sha256'] and a['score']==b['score']
    old_signal={(r['system'],r['sample_id']):r for r in read(SOURCE/'evaluation/combined_signals.jsonl')}
    registry=meta(SOURCE/'signal_systems.json');group={s['system']:s.get('geometry',s['kind']) for s in registry['direct']}
    selections=defaultdict(list)
    for key,r in signals.items():selections[r['room_id'],r['requested_rt60'],r['system'].split('_')[-1],'truth_ref_power_mask'].append(key)
    ids={r['sample_id'] for r in inputs.values()}
    for key,r in old_signal.items():
        if r['sample_id'] not in ids:continue
        method='mvdr' if r['system'].endswith('_mvdr') else 'mwf' if r['system'].endswith('_mwf') else None
        if method is None:continue
        name='truth_covariance' if r['system'].startswith('truth_covariance') else group.get(r['system'].replace('_mask_'+method,''))
        assert name in ('truth_covariance','mcnet','fixed','rotated','rigid')
        selections[r['room_id'],r['requested_rt60'],method,name].append(key)
    summary=[]
    for (room,rt,method,name),keys in sorted(selections.items()):
        new=name=='truth_ref_power_mask';quality=signals if new else old_signal
        row={'room':room,'rt':rt,'method':method,'group':name,'rows':len(keys),
            **{m:float(np.mean([quality[k]['metrics'][m] for k in keys])) for m in ('si_sdr_db','pesq_wb','stoi')}}
        for engine in plan['engines']:
            row[engine]={regime:corpus_score((scored if new else old)[engine][regime][k]['score'] for k in keys) for regime in plan['regimes']}
        summary.append(row)
    # Reference transcripts are independent of system replication; count each case once.
    references=[];first=next(iter(scored))
    for room in ('reference','new'):
        for rt in (0,.15,.35):
            case_ids={r['sample_id'] for r in inputs.values() if r['room_id']==room and r['requested_rt60']==rt}
            item={'room':room,'rt':rt,'cases':len(case_ids)}
            for engine in plan['engines']:
                item[engine]={}
                for regime in plan['regimes']:
                    unique={r['sample_id']:r['reference_score'] for r in scored[engine][regime].values() if r['sample_id'] in case_ids}
                    assert len(unique)==72;item[engine][regime]=corpus_score(unique.values())
            references.append(item)
    dump(REC/'summary.json',{'systems':summary,'references':references},True)
    report=ROOT/'docs/G27_真值功率比掩码信号与识别综合报告.md'
    lines=['# G27 真值功率比掩码信号与识别综合报告','','完整864输出；两个识别器各raw864行、电平控制864行。原有McNet、全部三种子适配组和真值协方差评分仅复用对应已验收条件。两个真值组均不可部署，不构成新方法成果或理论上界。','',
           '|房间|请求RT|输出|前端/统计|记录|SI-SDR|Whisper raw / 电平|CTC raw / 电平|','|---|---:|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['room']}|{r['rt']}|{r['method']}|{r['group']}|{r['rows']}|{r['si_sdr_db']:.3f}|{r['Whisper']['raw']['wer']:.2%} / {r['Whisper']['level_control']['wer']:.2%}|{r['CTC']['raw']['wer']:.2%} / {r['CTC']['level_control']['wer']:.2%}|")
    lines+=['', '完整S/D/I、参考转写评分、PESQ/STOI及全部系统/种子原始行均保留。三种子组记录数216，其他组72；分母重复同一72条件，不能当作更多独立语音。raw和电平控制不互换、不选更低者覆盖。峰值约束可能使RMS低于目标，规范化清单保存逐例诊断。', '',
        '真值掩码能否改善统计与识别，与模型能否学到、是否跨新语句泛化是三个问题。本阶段只检验第一个；不以此证明吞字、模型创新或训练收益。原掩码/协方差已有正式先例，应在独立开发数据上确定目标，并保留同预算成熟对照。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    done={'status':'ok','utc':now(),'new_outputs':864,'records_each_ASR':1728,'RT0_pairs_each_regime':144,
          'checks':checks,'summary_sha256':sha(REC/'summary.json'),'report_sha256':sha(report)}
    dump(OUT/'ASR_COMPLETE.json',done,True)
    archive('recognition_completion',report,[OUT/'ASR_COMPLETE.json',REC/'summary.json',REC/'frozen_plan.json',ROOT/'scripts/g27_asr_finish.py'])


def accept():
    plan=meta(REC/'frozen_plan.json');done=meta(OUT/'ASR_COMPLETE.json')
    assert done['status']=='ok' and meta(OUT/'power_asr_helper.json')['status']=='released'
    assert sha(REC/'summary.json')==done['summary_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    for engine in plan['engines']:
        for regime in plan['regimes']:
            folder=REC/engine/regime;check=done['checks'][engine][regime]
            assert sha(folder/'asr_scores.jsonl')==check['score_sha256'] and sha(folder/'cache_new.jsonl')==check['cache_sha256']
            assert len(read(folder/'asr_scores.jsonl'))==864;seed(engine,regime,plan)
    audit={'status':'ok','utc':now(),'outputs':864,'ASR_rows_each':1728,'stages':stages,'completion_sha256':sha(OUT/'ASR_COMPLETE.json'),
        'RT0_exact_pcm_pairs':144,'signal_power_released':True,'ASR_power_released':True,'training_performed':False}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G27_完整真值掩码诊断验收.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G27 完整真值掩码诊断验收\n\n864个信号输出、每识别器两条件共1728行评分及144对无反射一致性均通过。代码、输入、缓存、计分和各阶段快照哈希核对通过。首次文件哈希错误假设的失败与修正记录全部保留；未重算或改动波形。信号与识别两段临时保持唤醒均释放，无新训练、依赖或下载。\n\n[综合报告](G27_真值功率比掩码信号与识别综合报告.md)并列全部已有适配组三种子及两个真值诊断；真值系统不可部署，不认定创新或独立泛化。\n')
    archive('final_acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',OUT/'power_asr_helper.json',ROOT/'scripts/g27_asr_finish.py'])
    dump(OUT/'status.json',{'stage':'accepted','utc':now()})
