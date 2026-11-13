"""Keep raw/standardized ASR regimes separate and audit repeat observations."""
from collections import defaultdict
from g25_prepare import ROOT,OUT,PARENT,sha,read,dump,now,archive
from g25_asr import meta,FIELDS,status
from g23_finish import family
from g19_scores import waveform
from mcse.asr_scoring import normalize_english,score_words,corpus_score


def finish():
    rec=OUT/'recognition';plan=meta(rec/'frozen_plan.json')
    assert (OUT/'stages/ctc/archive_manifest.json').exists() and sha(OUT/'manifest.jsonl')==plan['manifest_sha256']
    inputs={(r['base_system'],r['sample_id'],r['observation']):r for r in read(OUT/'manifest.jsonl')};assert len(inputs)==960
    scores={};raw_scores={};details=[];audits={};repeatability=[]
    for engine in ('Whisper','CTC'):
        folder=rec/engine;marker=meta(folder/'COMPLETE.json');obs=read(folder/'observations.jsonl')
        assert sha(folder/'observations.jsonl')==marker['observations_sha256'] and marker['fresh_forwards']==960 and marker['cache_skips']==0
        parent_file=PARENT/'recognition'/engine/'observations.jsonl';assert sha(parent_file)==plan['parent'][engine]['observations_sha256']
        parent={(r['base_system'],r['sample_id']):r for r in read(parent_file) if r['observation']=='original_first'}
        joined={}
        for i,r in enumerate(obs):
            key=r['base_system'],r['sample_id'],r['observation'];assert key in inputs and key not in joined
            assert r['actual_forward'] and r['decode_index']==i and all(r[k]==inputs[key][k] for k in FIELDS)
            _,pcm=waveform(inputs[key]['output_path'],inputs[key]['output_sha256']);assert pcm==r['output_samples_sha256']
            old=parent[key[:2]];assert r['raw_hypothesis']==old['hypothesis'] and r['raw_score']==old['score']
            assert r['raw_source_samples_sha256']==old['output_samples_sha256']
            assert r['score']==score_words(r['transcript'],r['hypothesis']) and r['raw_score']==score_words(r['transcript'],r['raw_hypothesis'])
            joined[key]=r
        assert set(joined)==set(inputs) and len(parent)==480
        repeat=0;raw_repeat=0
        for base,old in parent.items():
            first,second=joined[(*base,'normalized_first')],joined[(*base,'normalized_second')]
            assert first['output_samples_sha256']==second['output_samples_sha256']
            changed=normalize_english(first['hypothesis'])!=normalize_english(second['hypothesis'])
            repeat+=changed;raw_repeat+=first['hypothesis']!=second['hypothesis']
            details.append({'engine':engine,'base_system':base[0],'sample_id':base[1],
                'room_id':first['room_id'],'rt':first['requested_rt60'],'speaker_id':first['speaker_id'],'angle':first['target_azimuth'],
                'normalized_repeat_changed':changed,
                'raw_to_normalized_changed':normalize_english(old['hypothesis'])!=normalize_english(first['hypothesis']),
                **{k+'_delta':first['score'][k]-old['score'][k] for k in ('S','D','I','errors')},'N':old['score']['N'],
                'raw_hypothesis':old['hypothesis'],'normalized_hypothesis':first['hypothesis'],'repeat_hypothesis':second['hypothesis']})
        repeatability.append({'engine':engine,'pairs':480,'normalized_repeat_changes':repeat,'raw_repeat_changes':raw_repeat})
        scores[engine]=joined;raw_scores[engine]=parent
        audits[engine]={'fresh_observations':960,'cache_skips':0,'raw_parent_pairs':480,'observation_sha256':marker['observations_sha256']}
    bases={key[:2]:r for key,r in inputs.items() if key[2]=='normalized_first'}
    def aggregate(keys,regime):
        return {'pairs':len(keys),**{e:corpus_score(raw_scores[e][k]['score'] if regime=='raw' else scores[e][(*k,regime)]['score'] for k in keys) for e in scores}}
    dims={'by_family':('family','regime'),'by_system':('base_system','regime'),
          'by_room_rt':('room_id','rt','family','regime'),'by_speaker':('speaker_id','family','regime'),
          'by_angle':('angle','family','regime')}
    summary={'repeatability':repeatability}
    for section,fields in dims.items():
        groups=defaultdict(list)
        for base,r in bases.items():
            for regime in ('raw','normalized_first','normalized_second'):
                attrs={**r,'rt':r['requested_rt60'],'angle':r['target_azimuth'],'family':family(r['base_system']),'regime':regime}
                groups[tuple(attrs[k] for k in fields)].append(base)
        summary[section]=[{**dict(zip(fields,key)),**aggregate(keys,key[fields.index('regime')])} for key,keys in groups.items()]
    changes=[]
    for e in scores:
        for fam in sorted({family(k[0]) for k in bases}):
            selected=[d for d in details if d['engine']==e and family(d['base_system'])==fam]
            changes.append({'engine':e,'family':fam,'pairs':len(selected),
                'raw_to_normalized_changes':sum(d['raw_to_normalized_changed'] for d in selected),
                'normalized_repeat_changes':sum(d['normalized_repeat_changed'] for d in selected),
                **{k:sum(d[k] for d in selected) for k in ('S_delta','D_delta','I_delta','errors_delta','N')}})
    summary['raw_to_normalized']=changes
    dump(rec/'summary.json',summary,True);dump(rec/'paired_observations.json',details,True)
    report=ROOT/'docs/G25_识别输入电平控制与原始评分对照报告.md'
    lines=['# G25 识别输入电平控制与原始评分对照报告','','本阶段仅改变ASR的输入副本，未改变增强音频或训练。固定RMS目标=.05、峰值上限=.95在看本阶段成绩前声明。G24全部二进制增益输入经此规则后字节完全相同；下面的两遍规范化观察都真实解码。','',
           '|识别器|重复对数|两遍规范化文本变化|','|---|---:|---:|']
    for r in repeatability:lines.append(f"|{r['engine']}|{r['pairs']}|{r['normalized_repeat_changes']}|")
    lines+=['','原幅度对照来自已验收G24的第一次原幅度（G24已核对跨次和同次重复性）。下面只计每个规范化输入的第一次，第二次用作重复控制，不翻倍为独立样本。','',
            '|类别|配对数/条件|Whisper WER：raw/电平控制|CTC WER：raw/电平控制|','|---|---:|---:|---:|']
    by={(r['family'],r['regime']):r for r in summary['by_family']}
    for fam in sorted({r['family'] for r in summary['by_family']}):
        a,b=by[fam,'raw'],by[fam,'normalized_first']
        lines.append(f"|{fam}|{a['pairs']}|{a['Whisper']['wer']:.2%} / {b['Whisper']['wer']:.2%}|{a['CTC']['wer']:.2%} / {b['CTC']['wer']:.2%}|")
    lines+=['','|识别器|类别|配对数|raw与电平控制文本变化|ΔS|ΔD|ΔI|Δ总错误|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in changes:lines.append(f"|{r['engine']}|{r['family']}|{r['pairs']}|{r['raw_to_normalized_changes']}|{r['S_delta']}|{r['D_delta']}|{r['I_delta']}|{r['errors_delta']}|")
    lines+=['','微WER使用总S+D+I/总参考词数。原始和电平控制评分属于不同评估条件，不能把后者的改变称为增强模型改善，也不挑更低的一个替换旧成绩。预处理ID、全部三种子、房间/RT、方向、说话人及配对文本均保留。',
            '', '输入的增益不变性不等于普遍ASR鲁棒性；该控制不修复语音内容，也不能解释所有噪声/混响问题。480对复用四人四句场景，没有显著性或新听感标签。真值系统及干净参考仅为诊断。raw音频的电平失真仍需用RMS/NMSE等原始指标观察。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    dump(rec/'completion_audit.json',{'status':'ok','utc':now(),'ASR':audits,'repeatability':repeatability,
          'summary_sha256':sha(rec/'summary.json'),'paired_sha256':sha(rec/'paired_observations.json')},True)
    archive('completion',report,[rec/'completion_audit.json',rec/'summary.json',rec/'frozen_plan.json',ROOT/'scripts/g25_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','utc':now(),'report':str(report.relative_to(ROOT)),
          'report_sha256':sha(report),'archive_sha256':sha(OUT/'stages/completion/archive_manifest.json')},True)
    status(stage='complete_pending_readback')
