#!/usr/bin/env python3
"""Paired cross-recognizer analysis, coverage audit and Chinese report."""
import json,runpy
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from g8_common import ROOT,OUT,ASSET,FILENAME,now,sha,read,dump,status,archive
from mcse.asr_scoring import corpus_score,score_words


def paired(rows, a, b):
    left={(r['sample_id'],r['system'].rsplit('_s',1)[1]):r for r in rows if r['system'] in a}
    right={(r['sample_id'],r['system'].rsplit('_s',1)[1]):r for r in rows if r['system'] in b}
    assert left and set(left)==set(right)
    bysource=defaultdict(lambda:np.zeros(4))
    for key,r in left.items():
        s=right[key];assert r['transcript']==s['transcript'] and r['score']['N']==s['score']['N']
        bysource[r['source_id']]+=np.array([r['score']['errors'],r['score']['N'],s['score']['errors'],s['score']['N']])
    matrix=np.stack([bysource[k] for k in sorted(bysource)]);t=matrix.sum(0);delta=t[0]/t[1]-t[2]/t[3]
    rng=np.random.default_rng(20261112);samples=matrix[rng.integers(len(matrix),size=(2000,len(matrix)))].sum(1)
    boots=samples[:,0]/samples[:,1]-samples[:,2]/samples[:,3]
    return {'delta_wer':float(delta),'conditional_source_bootstrap_2000_ci95':np.quantile(boots,[.025,.975]).tolist(),
      'sources':len(matrix),'paired_seed_cases':len(left),'a_errors':int(t[0]),'b_errors':int(t[2]),'N_per_method':int(t[1])}


def finish():
    plan=json.loads((OUT/'frozen_plan.json').read_text());ctc=read(OUT/'evaluation/asr_scores.jsonl');signals=[];whisper=[]
    for dataset,base in plan['datasets'].items():
        for r in read(ROOT/base['folder']/'signals.jsonl'):signals.append({**r,'dataset':dataset})
        for r in read(ROOT/base['folder']/'asr_scores.jsonl'):whisper.append({**r,'dataset':dataset})
    index=lambda rows:{(r['dataset'],r['system'],r['sample_id']):r for r in rows}
    ci=index(ctc);wi=index(whisper);si=index(signals);assert len(ci)==len(wi)==len(si)==12600 and set(ci)==set(wi)==set(si)
    cells={g+'_'+v:[g+'_'+v+'_s'+str(s) for s in (20261112,20261113,20261114)] for g in ('fixed','rotated') for v in ('without_clean_identity','complete')}
    engines={'Whisper':whisper,'CTC':ctc};summary={'coverage':dict(Counter(r['dataset'] for r in ctc)),'cells':{},'comparisons':[],'per_snr_comparisons':[],'clean_reference_floor':{},'g5_clean_input':{},'warnings':{}}
    for dataset in plan['datasets']:
        sig=[r for r in signals if r['dataset']==dataset];noises=sorted({r['noise_id'] for r in sig if r['condition']!='clean'})
        summary['cells'][dataset]={}
        for noise in noises:
            summary['cells'][dataset][noise]={}
            for group,names in {'noisy':['noisy'],'mcnet':['mcnet'],**cells}.items():
                summary['cells'][dataset][noise][group]={}
                for angle in (-40,20,70):
                    subset=[r for r in sig if r['condition']!='clean' and r['noise_id']==noise and r['target_azimuth']==angle and r['system'] in names]
                    keys=[(dataset,r['system'],r['sample_id']) for r in subset]
                    summary['cells'][dataset][noise][group][str(angle)]={engine:corpus_score(indexed[k]['score'] for k in keys) for engine,indexed in [('Whisper',wi),('CTC',ci)]}
            for variant in ('without_clean_identity','complete'):
                for angle in (-40,20,70):
                    def entry(snr=None):
                        value={'dataset':dataset,'noise_id':noise,'variant':variant,'target':angle,'snr_db':snr,'engines':{}}
                        for engine,rows in engines.items():
                            selected=[r for r in rows if r['dataset']==dataset and r['condition']!='clean' and r['noise_id']==noise and r['target_azimuth']==angle and (snr is None or r['snr_db']==snr)]
                            value['engines'][engine]=paired(selected,cells['rotated_'+variant],cells['fixed_'+variant])
                        value['sign_agreement']=bool(np.sign(value['engines']['Whisper']['delta_wer'])==np.sign(value['engines']['CTC']['delta_wer']))
                        return value
                    summary['comparisons'].append(entry())
                    for snr in sorted({r['snr_db'] for r in sig if r['noise_id']==noise and r['condition']!='clean'}):summary['per_snr_comparisons'].append(entry(snr))
        # Each exact reference representation once, not once per enhanced system.
        refs={}
        for r in ctc:
            if r['dataset']!=dataset:continue
            k=(r['source_id'],r['reference_samples_sha256']);refs[k]=(r,wi[(dataset,r['system'],r['sample_id'])])
        summary['clean_reference_floor'][dataset]={'unique_source_waveforms':len(refs),'unique_sources':len({k[0] for k in refs}),
           'Whisper':corpus_score(v[1]['reference_score'] for v in refs.values()),'CTC':corpus_score(v[0]['reference_score'] for v in refs.values()),
           'notice':'Exact scaled reference waveforms counted once; multiple representations of the same12 sources are not independent sentences.'}
    for group,names in {'noisy':['noisy'],'mcnet':['mcnet'],**cells}.items():
        subset=[r for r in signals if r['dataset']=='G5' and r['condition']=='clean' and r['system'] in names];keys=[('G5',r['system'],r['sample_id']) for r in subset]
        summary['g5_clean_input'][group]={engine:corpus_score(idx[k]['score'] for k in keys) for engine,idx in [('Whisper',wi),('CTC',ci)]}
    for engine,rows in engines.items():
        summary['warnings'][engine]={'WER_over100_rows':sum(r['score']['wer']>1 for r in rows),'insertions_above_N_rows':sum(r['score']['I']>r['score']['N'] for r in rows)}
    summary['comparison_sign_agreement']={'agree':sum(r['sign_agreement'] for r in summary['comparisons']),'total':len(summary['comparisons']),
       'notice':'Exploratory direction-of-effect agreement over dependent cells; not an accuracy or significance probability.'}
    dump(OUT/'evaluation/summary.json',summary,True)
    paths=json.loads((OUT/'evaluation/path_hash_index.json').read_text());cache=read(OUT/'evaluation/ctc_cache.jsonl');cacheidx={r['cache_key']:r for r in cache}
    checks={'all12600_pairs_unique_and_paired':len(ci)==12600 and set(ci)==set(wi)==set(si),
      'all_ctc_scores_recomputed_exactly':all(r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis']) for r in ctc),
      'all_saved_audio_hashes_still_match':all(sha(p)==v['file_sha256'] for p,v in paths.items()),
      'cache_unique_and_paths_resolved':len(cacheidx)==len(cache) and all(v['cache_key'] in cacheidx and cacheidx[v['cache_key']]['sample_sha256']==v['sample_sha256'] for v in paths.values()),
      'asset_hash_unchanged':sha(ASSET/FILENAME)==json.loads((OUT/'asset_provenance.json').read_text())['local_sha256'],
      'asset_report_archived_before_scoring':json.loads((OUT/'stages/asset/archive_manifest.json').read_text())['saved_utc']<json.loads((OUT/'asr_provenance.json').read_text())['started_utc'],
      'prior_signal_and_whisper_evidence_unchanged':all(sha(ROOT/v['folder']/'signals.jsonl')==v['signals_sha256'] and sha(ROOT/v['folder']/'asr_scores.jsonl')==v['whisper_sha256'] for v in plan['datasets'].values())}
    selected=json.loads((ROOT/'artifacts/g5_rotation_v1/selected_checkpoints.json').read_text())['systems'];checks['all12_SE_adapters_unchanged']=all(sha(ROOT/r['path'])==r['checkpoint_sha256'] for r in selected)
    bridge=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g8_integrity');repo=ROOT/'third_party/McNet';bridge['verify_upstream'](repo,repo/bridge['OFFLINE_CHECKPOINT']);checks['official_SE_source_and_weights_unchanged']=True
    audit={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'completed_utc':now(),'scored_rows':12600,'unique_CTC_decodes':len(cache),'new_SE_training':False}
    dump(OUT/'completion_audit.json',audit,True)
    if audit['status']!='ok':raise RuntimeError('G8 audit failed; preserve stage')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    for dataset,color in [('G5','#426db0'),('G7','#b2573a')]:
        v=[r for r in summary['comparisons'] if r['dataset']==dataset];ax.scatter([r['engines']['Whisper']['delta_wer']*100 for r in v],[r['engines']['CTC']['delta_wer']*100 for r in v],label=dataset,color=color)
    ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Rotated - fixed WER, Whisper (pp)');ax.set_ylabel('Rotated - fixed WER, CTC (pp)');ax.set_title('Same saved SE waves, different recognizers; lower is better');ax.grid(alpha=.2);ax.legend();fig.savefig(OUT/'evaluation/cross_ASR_effects.png',dpi=170);plt.close(fig)
    lines=['# G8 双识别器交叉核查报告','','## 问题与边界','','本阶段仅对已有G5/G7增强音频增加官方wav2vec2 base960h贪心CTC识别。未训练增强器、未重新选择权重、未增加源句、未采用ASR损失。G5与G7复用同12句/3人，12,600行是系统×条件记录，不是12,600个独立句子。',
      '', '两个识别器均使用相同英语词归一化和编辑距离；WER=S+D+I除以参考词数，保留>100%的值。应比较同一个识别器中的方法差值，不能把两识别器绝对WER差当成增强器进步。CTC不使用外部LM或提示，但仍有学习到的语言规律。',
      '', '## 干净参考的识别下限','','参考按源句及精确float32样本哈希去重，缩放不同的同句仍可能多条，不能视为额外独立语料。','','|数据|源句/参考波形|Whisper WER|CTC WER|','|---|---:|---:|---:|']
    for ds,r in summary['clean_reference_floor'].items():lines.append(f'|{ds}|{r["unique_sources"]}/{r["unique_source_waveforms"]}|{r["Whisper"]["wer"]:.1%}|{r["CTC"]["wer"]:.1%}|')
    lines+=['','## 同一识别器中的旋转减固定效应','','全部种子保留；每句合并配对条件和种子，再按12句做2,000次bootstrap。区间仅条件于这3位说话人和已采样噪声，未作多重比较控制。负值更好。各新噪声合并−5/0/+5dB；G5仅0dB。逐SNR结果另保存在机器汇总。','','|数据/噪声|干净约束|方向|Whisper ΔWER百分点 [区间]|CTC ΔWER百分点 [区间]|符号一致|','|---|---|---:|---|---|---|']
    for r in summary['comparisons']:
        def fmt(engine):
            x=r['engines'][engine];lo,hi=x['conditional_source_bootstrap_2000_ci95'];return f'{x["delta_wer"]*100:+.1f} [{lo*100:+.1f},{hi*100:+.1f}]'
        lines.append(f'|{r["dataset"]}/{r["noise_id"]}|{r["variant"]}|{r["target"]}|{fmt("Whisper")}|{fmt("CTC")}|{"是" if r["sign_agreement"] else "否"}|')
    agree=summary['comparison_sign_agreement'];lines+=['',f'上述依赖条件中效应符号一致{agree["agree"]}/{agree["total"]}；此计数不是统计显著率或方法正确率。区间跨零、识别器不一致和不利条件均需保留。',
      '', '![识别器间效应](../artifacts/g8_cross_asr_v1/evaluation/cross_ASR_effects.png)','','## 全部系统组的方向WER','','每格为Whisper / CTC，按总编辑次数与参考词数微平均，适配器合并三种子。','','|数据/噪声|系统组|−40°|20°|70°|','|---|---|---:|---:|---:|']
    for ds,noises in summary['cells'].items():
        for noise,groups in noises.items():
            for group,angles in groups.items():lines.append('|'+ds+'/'+noise+'|'+group+'|'+'|'.join(f'{angles[str(t)]["Whisper"]["wer"]:.1%} / {angles[str(t)]["CTC"]["wer"]:.1%}' for t in (-40,20,70))+'|')
    lines+=['','## 干净输入对照及解释','','G5干净输入仍全部计分；G7没有新增干净增强对照。下表也不是额外独立句子。','','|系统组|Whisper WER|CTC WER|','|---|---:|---:|']
    for group,v in summary['g5_clean_input'].items():lines.append(f'|{group}|{v["Whisper"]["wer"]:.1%}|{v["CTC"]["wer"]:.1%}|')
    lines+=['','双识别器一致只能加强“对这两种识别后端的任务表现”证据，不能证明真实音素保留；不一致则限制以单一ASR为依据的结论。已有人工辅助试听未确认稳定吞字，本阶段没有新增人工标签。仍不应自动设计词语恢复模块。',
      '', '研究结论必须同时结合G5/G7信号指标：若提高某方向而降低20°或70°，应如实称作方向权衡。现有旋转增强、LoRA、干净约束均是已有技术，本次核查不建立新颖性。更有意义的后续应先明确稳健的多指标目标，并设成熟方法对照，不能因测试结果挑一个最佳种子。',
      '', '## 审计与来源','', f'覆盖G5=3,528、G7=9,072，共12,600对；唯一CTC解码{len(cache):,}。全部音频/原识别结果/增强权重哈希、计分重算、缓存映射及阶段顺序审计{audit["status"]}。无新依赖安装、驱动/CUDA修改。',
      '', '[G8A来源与资源](G8A_CTC识别器来源与资源验证.md)、[机器汇总](../artifacts/g8_cross_asr_v1/evaluation/summary.json)、[完成审计](../artifacts/g8_cross_asr_v1/completion_audit.json)。',
      '', '模型依据：[torchaudio官方文档及MIT许可说明](https://docs.pytorch.org/audio/2.5.0/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html)。数据沿用此前合法公开LibriSpeech/DEMAND小规模材料，见G5/G7源清单。','', 'UTC：'+now()]
    report=ROOT/'docs/G8_双识别器交叉核查报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('completion',report,[OUT/'completion_audit.json',OUT/'evaluation/summary.json',OUT/'asr_provenance.json',ROOT/'scripts/g8_evaluate.py',Path(__file__)])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'stage':'G8','completed_utc':now(),'status':'ok','report':str(report.relative_to(ROOT))},True);status(stage='complete',report=str(report.relative_to(ROOT)))


if __name__=='__main__':finish()
