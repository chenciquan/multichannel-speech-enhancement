"""Read back every fresh ASR observation and separate repeatability from gains."""
from collections import defaultdict
from g24_prepare import ROOT,OUT,sha,read,dump,now,archive
from g24_asr import meta,FIELDS,status
from g23_finish import family
from g19_scores import waveform
from mcse.asr_scoring import normalize_english,score_words,corpus_score


def finish():
    rec=OUT/'recognition';plan=meta(rec/'frozen_plan.json')
    assert (OUT/'stages/ctc/archive_manifest.json').exists() and sha(OUT/'manifest.jsonl')==plan['manifest_sha256']
    inputs={(r['base_system'],r['sample_id'],r['observation']):r for r in read(OUT/'manifest.jsonl')}
    scores={};audits={};details=[]
    for engine in ('Whisper','CTC'):
        folder=rec/engine;marker=meta(folder/'COMPLETE.json');obs=read(folder/'observations.jsonl')
        assert sha(folder/'observations.jsonl')==marker['observations_sha256'] and marker['actual_forwards']==1920 and marker['cache_skips']==0
        index={};parents={}
        for i,r in enumerate(obs):
            k=r['base_system'],r['sample_id'],r['observation'];assert k in inputs and k not in index
            assert r['decode_index']==i and r['actual_forward'] is True
            assert all(r[field]==inputs[k][field] for field in FIELDS)
            _,key=waveform(inputs[k]['output_path'],inputs[k]['output_sha256']);assert key==r['output_samples_sha256']
            assert r['score']==score_words(r['transcript'],r['hypothesis'])
            assert r['historical_baseline_score']==score_words(r['transcript'],r['historical_baseline_hypothesis'])
            base=k[:2]
            historical=(r['historical_baseline_hypothesis'],r['historical_baseline_score'])
            if base in parents:assert historical==parents[base]
            else:parents[base]=historical
            index[k]=r
        assert set(index)==set(inputs) and len(parents)==480
        for (system,sid),(old_hyp,old_score) in parents.items():
            first=index[system,sid,'original_first'];second=index[system,sid,'original_second']
            assert first['output_samples_sha256']==second['output_samples_sha256']
            cross=normalize_english(first['hypothesis'])!=normalize_english(old_hyp)
            repeat=normalize_english(first['hypothesis'])!=normalize_english(second['hypothesis'])
            for label in ('original_first','half','quarter','original_second'):
                r=index[system,sid,label];baseline=old_score if label=='original_first' else first['score']
                baseline_hyp=old_hyp if label=='original_first' else first['hypothesis']
                details.append({'engine':engine,'base_system':system,'sample_id':sid,'observation':label,
                    'gain_exact':r['gain_exact'],'room_id':r['room_id'],'rt':r['requested_rt60'],
                    'speaker_id':r['speaker_id'],'angle':r['target_azimuth'],
                    'cross_run_original_changed':cross,'within_run_original_changed':repeat,
                    'normalized_change':normalize_english(r['hypothesis'])!=normalize_english(baseline_hyp),
                    'raw_change':r['hypothesis']!=baseline_hyp,'N':baseline['N'],
                    **{k+'_delta':r['score'][k]-baseline[k] for k in ('S','D','I','errors')},
                    'baseline_hypothesis':baseline_hyp,'observed_hypothesis':r['hypothesis']})
        scores[engine]=index
        audits[engine]={'observations':1920,'actual_forwards':1920,'cache_skips':0,
                       'observation_sha256':marker['observations_sha256']}
    repeats=[]
    for e in scores:
        first=[d for d in details if d['engine']==e and d['observation']=='original_first']
        last=[d for d in details if d['engine']==e and d['observation']=='original_second']
        repeats.append({'engine':e,'bases':480,'cross_run_normalized_changes':sum(d['normalized_change'] for d in first),
                        'cross_run_raw_changes':sum(d['raw_change'] for d in first),
                        'within_run_normalized_changes':sum(d['normalized_change'] for d in last),
                        'within_run_raw_changes':sum(d['raw_change'] for d in last)})
    dims={'by_family':('family','observation'),'by_system':('base_system','observation'),
          'by_room_rt':('room_id','rt','family','observation'),'by_speaker':('speaker_id','family','observation'),
          'by_angle':('angle','family','observation')}
    summary={'repeatability':repeats}
    for label,fields in dims.items():
        groups=defaultdict(list)
        for key,r in inputs.items():
            attrs={**r,'family':family(r['base_system']),'rt':r['requested_rt60'],'angle':r['target_azimuth']}
            groups[tuple(attrs[f] for f in fields)].append(key)
        summary[label]=[{**dict(zip(fields,k)),'rows':len(keys),
                         **{e:corpus_score(scores[e][key]['score'] for key in keys) for e in scores}} for k,keys in groups.items()]
    changes=[]
    for e in scores:
        for fam in sorted({family(r['base_system']) for r in inputs.values()}):
            for label in ('half','quarter'):
                ds=[d for d in details if d['engine']==e and family(d['base_system'])==fam and d['observation']==label]
                stable=[d for d in ds if not d['cross_run_original_changed'] and not d['within_run_original_changed']]
                changes.append({'engine':e,'family':fam,'observation':label,'pairs':len(ds),
                    'normalized_changes':sum(d['normalized_change'] for d in ds),
                    'stable_baseline_pairs':len(stable),'changes_in_stable_baseline_pairs':sum(d['normalized_change'] for d in stable),
                    **{k:sum(d[k] for d in ds) for k in ('S_delta','D_delta','I_delta','errors_delta','N')}})
    summary['gain_changes']=changes;dump(rec/'summary.json',summary,True);dump(rec/'paired_observations.json',details,True)
    report=ROOT/'docs/G24_独立重复与精确增益识别验证报告.md'
    lines=['# G24 独立重复与精确增益识别验证报告','','全部观察都真实解码；没有用缓存跳过相同波形，也没有覆盖G22/G23的历史文本。半幅和四分之一幅波形逐样本精确反向还原，排除了输入缩放步骤的舍入损失。','',
           '|识别器|基准对数|跨次原幅度规范化变化|同次两遍原幅度规范化变化|','|---|---:|---:|---:|']
    for r in repeats:lines.append(f"|{r['engine']}|{r['bases']}|{r['cross_run_normalized_changes']}|{r['within_run_normalized_changes']}|")
    lines+=['','跨次比较当前第一次原幅度与G23的历史0dB；同次比较本次第二遍原幅度与第一遍。以下增益比较相对本次第一次原幅度，所有不稳定基准保留。','',
            '|识别器|类别|精确缩放|配对数|规范化变化|稳定基准对数|稳定基准中的变化|ΔS|ΔD|ΔI|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in changes:lines.append(f"|{r['engine']}|{r['family']}|{r['observation']}|{r['pairs']}|{r['normalized_changes']}|{r['stable_baseline_pairs']}|{r['changes_in_stable_baseline_pairs']}|{r['S_delta']}|{r['D_delta']}|{r['I_delta']}|")
    stable=all(r['cross_run_normalized_changes']==r['within_run_normalized_changes']==0 for r in repeats)
    if stable:
        lines+=['','本组原幅度跨次与同次的规范化文本均复现。若精确增益下仍有变化，支持本评估链对幅度敏感，而非输入缩放文件的舍入损失；不排除识别器内部的有限精度数值行为。']
    else:
        lines+=['','存在原幅度重复差异，不能将所有增益差异单独归因于幅度；已单列基准稳定与不稳定的观察，没有删除或替换历史结果。']
    lines+=['','全部四种观察的微WER、S/D/I及按系统/种子、房间/RT、说话人和方向的结果保存在 recognition/summary.json；全部配对文本保留。类别重复模型数不同，不混作独立语句大样本。','',
            '本实验不寻找最佳增益，不作为新的增强方法或吞字证明；没有新增增强推理、训练、数据集、人工听感或TSOS结果。精确数字波形结构保持不保证固定播放音量下人的可听度不变。只覆盖同一48个既有场景条件，不外推真实阵列或其他语言。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    dump(rec/'completion_audit.json',{'status':'ok','utc':now(),'ASR':audits,'repeatability':repeats,
          'summary_sha256':sha(rec/'summary.json'),'paired_observations_sha256':sha(rec/'paired_observations.json')},True)
    archive('completion',report,[rec/'completion_audit.json',rec/'summary.json',rec/'frozen_plan.json',ROOT/'scripts/g24_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','utc':now(),'report':str(report.relative_to(ROOT)),
          'report_sha256':sha(report),'archive_sha256':sha(OUT/'stages/completion/archive_manifest.json')},True)
    status(stage='complete_pending_readback')
