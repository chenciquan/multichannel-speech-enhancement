"""Audit complete hypotheses and alignments; supplementary results cannot overturn G31."""
from collections import defaultdict
import numpy as np
from g32_common import ROOT,OUT,sha,read,dump,now,meta,archive,status,data_guard
from g32_asr import REC,FIELDS,seed
from mcse.asr_scoring import score_words,corpus_score,normalize_english


def finish():
    plan=meta(REC/'frozen_plan.json');data_guard()
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    inputs={(r['system'],r['sample_id']):r for r in read(REC/'input_manifest.jsonl')};assert len(inputs)==468
    quality={(r['system'],r['sample_id']):r for r in read(OUT/'signals.jsonl')};assert set(inputs)==set(quality)
    dry={r['source_id']:r for r in meta(REC/'dry_anchors.json')};scores={};refs={};anchors={};checks={}
    for engine in plan['engines']:
        scores[engine]={};refs[engine]={};anchors[engine]={};checks[engine]={}
        for regime in plan['regimes']:
            folder=REC/engine/regime;done=meta(folder/'COMPLETE.json');prior,cache=seed(engine,regime,plan)
            current=meta(folder/'provenance.json');assert all(current[k]==prior[k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
            assert current['ASR_regime']==regime and current['input_preprocessing']==(plan['normalization'] if regime=='level_control' else None)
            assert sha(folder/'asr_scores.jsonl')==done['score_sha256'] and sha(folder/'cache_new.jsonl')==done['cache_sha256']
            assert sha(folder/'dry_anchor_scores.jsonl')==done['anchors_sha256']
            new=read(folder/'cache_new.jsonl');assert len(new)==sum(done['fresh'].values())
            for r in new:
                assert r['sample_sha256'] not in cache and r['model_id']==prior['model_id'] and r['ASR_regime']==regime
                assert r['normalization_id']==(plan['normalization']['preprocessing_id'] if regime=='level_control' else None)
                cache[r['sample_sha256']]=r['hypothesis']
            anchored={}
            for r in read(folder/'dry_anchor_scores.jsonl'):
                sid=r['source_id'];assert sid in dry and sid not in anchored and r['transcript']==dry[sid]['transcript']
                assert r['sample_sha256']==dry[sid]['inputs'][regime]['key'] and r['hypothesis']==cache[r['sample_sha256']]
                assert r['score']==score_words(r['transcript'],r['hypothesis']);anchored[sid]=r
            assert len(anchored)==4 and sum(r['score']['N'] for r in anchored.values())==16
            scored={};reference={}
            for r in read(folder/'asr_scores.jsonl'):
                key=r['system'],r['sample_id'];assert key in inputs and key not in scored
                assert all(r[k]==inputs[key][k] for k in FIELDS) and r['ASR_regime']==regime
                assert r['output_samples_sha256']==inputs[key]['inputs'][regime]['output_key']
                assert r['reference_samples_sha256']==inputs[key]['inputs'][regime]['reference_key']
                assert r['hypothesis']==cache[r['output_samples_sha256']] and r['reference_hypothesis']==cache[r['reference_samples_sha256']]
                assert r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis'])
                v=r['reference_samples_sha256'],r['reference_hypothesis'],r['reference_score']
                if r['sample_id'] in reference:assert reference[r['sample_id']]==v
                else:reference[r['sample_id']]=v
                scored[key]=r
            assert set(scored)==set(inputs) and len(reference)==36
            scores[engine][regime]=scored;refs[engine][regime]=reference;anchors[engine][regime]=anchored
            checks[engine][regime]={'rows':468,'references':36,'dry_anchors':4,'fresh_decodes':done['fresh'],
                'score_sha256':done['score_sha256'],'cache_sha256':done['cache_sha256'],'anchors_sha256':done['anchors_sha256']}
    def aggregate(keys):
        rs=[quality[k] for k in keys];result={'rows':len(keys),'unique_sources':len({r['source_id'] for r in rs}),
            **{m:float(np.mean([r['metrics'][m] for r in rs])) for m in ('si_sdr_db','pesq_wb','stoi')}}
        for engine in plan['engines']:result[engine]={regime:corpus_score(scores[engine][regime][k]['score'] for k in keys) for regime in plan['regimes']}
        if all(r['condition']=='clean' for r in rs):result['clean_levels']={m:float(np.mean([r['clean_levels'][m] for r in rs])) for m in ('rms_gain_ratio','reference_normalized_squared_error')}
        return result
    dimensions={'overall':('source_system','method','clean'),'by_rt':('source_system','method','requested_rt60','clean'),
        'by_speaker':('source_system','method','speaker_id','clean'),'by_source':('source_system','method','source_id','clean'),
        'by_field':('source_system','method','noise_field','requested_rt60')};summary={}
    for section,fields in dimensions.items():
        buckets=defaultdict(list)
        for key,r in quality.items():
            a={**r,'clean':r['condition']=='clean'};buckets[tuple(a[k] for k in fields)].append(key)
        summary[section]=[{**dict(zip(fields,k)),**aggregate(keys)} for k,keys in buckets.items()]
    references=[]
    for rt in (0,.16,.28):
        for clean in (False,True):
            ids={r['sample_id'] for r in inputs.values() if r['requested_rt60']==rt and (r['condition']=='clean')==clean}
            references.append({'rt':rt,'clean':clean,'cases':len(ids),**{e:{reg:corpus_score(refs[e][reg][sid][2] for sid in ids) for reg in plan['regimes']} for e in plan['engines']}})
    summary['reverberant_references']=references
    summary['dry_anchors']={e:{reg:{'corpus':corpus_score(r['score'] for r in anchors[e][reg].values()),'records':list(anchors[e][reg].values())} for reg in plan['regimes']} for e in plan['engines']}
    changes=[]
    for engine in plan['engines']:
        for key in inputs:
            raw=scores[engine]['raw'][key];level=scores[engine]['level_control'][key]
            changes.append({'engine':engine,'system':key[0],'sample_id':key[1],
                'normalized_hypothesis_changed':normalize_english(raw['hypothesis'])!=normalize_english(level['hypothesis']),
                **{m+'_delta':level['score'][m]-raw['score'][m] for m in ('S','D','I','errors')},'N':raw['score']['N']})
    dump(REC/'summary.json',summary,True);dump(REC/'raw_level_changes.json',changes,True)
    by={(r['source_system'],r['method'],r['clean']):r for r in summary['overall']}
    report=ROOT/'docs/G32_完整开发语句信号与双识别器综合报告.md'
    lines=['# G32 完整开发语句信号与双识别器综合报告','','四条旧开发整句/两人，36条件、468个系统输出；每识别器raw468与电平控制468行，另列原干声4条双条件参考。16个原始词token在条件间重复，带噪每系统N=96、干净N=48，不当作96/48个独立语句或词汇样本。G31主门槛未过的事实不变。', '',
        '## 带噪条件：每系统24行', '', '|系统|输出|SI-SDR|PESQ|STOI|Whisper WER raw/电平|CTC WER raw/电平|','|---|---|---:|---:|---:|---:|---:|']
    for (system,method,clean),r in sorted(by.items()):
        if clean:continue
        lines.append(f"|{system}|{method}|{r['si_sdr_db']:.3f}|{r['pesq_wb']:.3f}|{r['stoi']:.4f}|{r['Whisper']['raw']['wer']:.2%} / {r['Whisper']['level_control']['wer']:.2%}|{r['CTC']['raw']['wer']:.2%} / {r['CTC']['level_control']['wer']:.2%}|")
    lines+=['','## 干净条件：每系统12行','','|系统|输出|RMS增益比|NMSE|Whisper WER raw/电平|CTC WER raw/电平|','|---|---|---:|---:|---:|---:|']
    for (system,method,clean),r in sorted(by.items()):
        if not clean:continue
        c=r['clean_levels'];lines.append(f"|{system}|{method}|{c['rms_gain_ratio']:.4f}|{c['reference_normalized_squared_error']:.6f}|{r['Whisper']['raw']['wer']:.2%} / {r['Whisper']['level_control']['wer']:.2%}|{r['CTC']['raw']['wer']:.2%} / {r['CTC']['level_control']['wer']:.2%}|")
    lines+=['','## 原干声参考：每条件N=16','','|识别器|raw WER|电平控制WER|','|---|---:|---:|']
    for e in plan['engines']:lines.append(f"|{e}|{summary['dry_anchors'][e]['raw']['corpus']['wer']:.2%}|{summary['dry_anchors'][e]['level_control']['corpus']['wer']:.2%}|")
    lines+=['','逐条原转写、原干声假设、混响参考假设、完整S/D/I、全部RT/源/说话人/场型细分以及电平改变的文本均保存。固有拼写/人名/短句识别错误不能直接归因于增强。参考转写从未进入解码或模型提示；没有为降低WER改写原转写、英美拼写或分词。', '',
        '本轮是补充开发诊断，既不能推翻G31主门槛，也不把某一个后端/电平/滤波器的更低WER挑出来当胜利。短句、16词、两人、一个训练种子及模拟房间使结果很不稳定；条件共享说话内容，不据重复条件给出独立样本显著性。', '',
        '原始质量和干净电平没有随ASR预处理变化。峰值约束可能使RMS低于目标，实际逐例诊断另存。SI-SDR、PESQ/STOI、ASR S/D/I和人耳/声学词语保留分开解释，尚无新增逐词或盲听标签。跨本轮整句与G31裁剪的差异，不单独归因长度。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    complete={'status':'ok','utc':now(),'rows_each_ASR':936,'anchors_each_ASR':8,'references_each_regime':36,'checks':checks,
        'summary_sha256':sha(REC/'summary.json'),'changes_sha256':sha(REC/'raw_level_changes.json'),
        'report_sha256':sha(report),'G31_extension_gate_passed':False}
    dump(OUT/'ASR_COMPLETE.json',complete,True)
    archive('recognition_completion',report,[OUT/'ASR_COMPLETE.json',REC/'summary.json',REC/'raw_level_changes.json',REC/'input_audit.json',ROOT/'scripts/g32_asr_finish.py'])


def accept():
    plan=meta(REC/'frozen_plan.json');done=meta(OUT/'ASR_COMPLETE.json');assert done['status']=='ok'
    assert meta(OUT/'power_asr_helper.json')['status']=='released'
    assert sha(REC/'summary.json')==done['summary_sha256'] and sha(REC/'raw_level_changes.json')==done['changes_sha256']
    data_guard();assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    for engine in plan['engines']:
        for regime in plan['regimes']:
            seed(engine,regime,plan);folder=REC/engine/regime;check=done['checks'][engine][regime]
            assert sha(folder/'asr_scores.jsonl')==check['score_sha256'] and sha(folder/'cache_new.jsonl')==check['cache_sha256']
            assert sha(folder/'dry_anchor_scores.jsonl')==check['anchors_sha256']
            assert len(read(folder/'asr_scores.jsonl'))==468 and len(read(folder/'dry_anchor_scores.jsonl'))==4
    stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    audit={'status':'ok','utc':now(),'signal_rows':468,'ASR_rows_each':936,'dry_anchors_each':8,'stages':stages,
        'completion_sha256':sha(OUT/'ASR_COMPLETE.json'),'power_released':True,'training_performed':False,
        'G31_gate_unchanged':True,'unique_original_word_tokens':16}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G32_完整开发语句补充诊断验收.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G32 完整开发语句补充诊断验收\n\n36输入/目标全重放、468信号及各识别器936行+8干声参考完整覆盖。缓存与两种输入条件区分、全部词对齐及参考假设、代码/参数/数据/各阶段快照哈希核对通过。信号和识别临时唤醒均释放；冻结前接口失败及其修正保留。无新训练、安装或下载。\n\n[综合报告](G32_完整开发语句信号与双识别器综合报告.md)保留两个后端、两个电平和全部强对照。四条旧开发语句仅16词，不当最终泛化或人声误删证据；G31预登记扩展门槛仍为未通过。\n')
    archive('final_acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',OUT/'power_asr_helper.json',ROOT/'scripts/g32_asr_finish.py'])
    status(stage='accepted',G31_extension_gate_passed=False)
