#!/usr/bin/env python3
"""Full frozen-system comparator table, audit and Chinese stage report."""
import json,runpy
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
from g9_common import ROOT,OUT,now,sha,read,dump,status,archive
from mcse.asr_scoring import corpus_score,score_words


def finish():
    plan=json.loads((OUT/'frozen_plan.json').read_text());new=read(OUT/'evaluation/signals.jsonl');signals=list(new)
    ctcs=read(ROOT/'artifacts/g8_cross_asr_v1/evaluation/asr_scores.jsonl');whispers=[]
    for ds,spec in plan['datasets'].items():
        folder=ROOT/spec['folder'];assert sha(folder/'signals.jsonl')==spec['signals_sha256'] and sha(folder/'asr_scores.jsonl')==spec['whisper_sha256']
        signals.extend({**r,'dataset':ds} for r in read(folder/'signals.jsonl'));whispers.extend({**r,'dataset':ds} for r in read(folder/'asr_scores.jsonl'))
    whisper_new=read(OUT/'evaluation/Whisper/asr_scores.jsonl');ctc_new=read(OUT/'evaluation/CTC/asr_scores.jsonl');whispers+=whisper_new;ctcs+=ctc_new
    key=lambda r:(r['dataset'],r['system'],r['sample_id']);wi={key(r):r for r in whispers};ci={key(r):r for r in ctcs};si={key(r):r for r in signals}
    assert len(signals)==len(wi)==len(ci)==15300 and set(si)==set(wi)==set(ci)
    groups={'noisy':['noisy'],'mcnet':['mcnet']}
    groups.update({g+'_'+v:[g+'_'+v+'_s'+str(s) for s in (20261112,20261113,20261114)] for g in ('fixed','rotated') for v in ('without_clean_identity','complete')})
    groups.update({s:[s] for s in plan['systems']})
    def agg(items):
        return {'records':len(items),'sources':len({r['source_id'] for r in items}),'speakers':len({r['speaker_id'] for r in items}),
          'metrics':{k:float(np.mean([r['metrics'][k] for r in items])) for k in ('si_sdr_db','pesq_wb','stoi')},
          'Whisper':corpus_score(wi[key(r)]['score'] for r in items),'CTC':corpus_score(ci[key(r)]['score'] for r in items)}
    summary={'coverage':{'new_signals':len(new),'new_asr_per_engine':len(ctc_new),'combined_per_engine':15300,'sources':12,'speakers':3},'cells':{},'per_snr':{},'per_field':{},'per_speaker':{},'clean':{},'timing':{}}
    for ds in plan['datasets']:
        subset=[r for r in signals if r['dataset']==ds and r['condition']!='clean'];summary['cells'][ds]={};summary['per_snr'][ds]={}
        for noise in sorted({r['noise_id'] for r in subset}):
            summary['cells'][ds][noise]={};summary['per_snr'][ds][noise]={}
            for group,names in groups.items():
                choose=lambda **kw:[r for r in subset if r['noise_id']==noise and r['system'] in names and all(r[k]==v for k,v in kw.items())]
                summary['cells'][ds][noise][group]={str(t):agg(choose(target_azimuth=t)) for t in (-40,20,70)}
                summary['per_snr'][ds][noise][group]={str(snr):{str(t):agg(choose(snr_db=snr,target_azimuth=t)) for t in (-40,20,70)} for snr in sorted({r['snr_db'] for r in choose()})}
        summary['per_field'][ds]={group:{field:agg([r for r in subset if r['system'] in names and r['condition']==field]) for field in ('point','multi8')} for group,names in groups.items()}
        summary['per_speaker'][ds]={group:{sp:agg([r for r in subset if r['system'] in names and r['speaker_id']==sp]) for sp in sorted({r['speaker_id'] for r in subset})} for group,names in groups.items()}
    for group,names in groups.items():summary['clean'][group]=agg([r for r in signals if r['dataset']=='G5' and r['condition']=='clean' and r['system'] in names])
    for system in plan['systems']:
        rows=[r for r in new if r['system']==system];t=np.array([r['inference_seconds'] for r in rows]);m=np.array([r['original_mcnet_inference_seconds'] for r in rows])
        summary['timing'][system]={'mean_filter_seconds':float(t.mean()),'median_filter_seconds':float(np.median(t)),'p90_filter_seconds':float(np.quantile(t,.9)),
          'mean_filter_RTF':float(np.mean([r['inference_seconds']/r['audio_seconds'] for r in rows])),
          'requires_McNet':system=='mcnet_mask_mvdr','mean_previous_McNet_seconds':float(m.mean()) if system=='mcnet_mask_mvdr' else None,
          'mean_filter_plus_previous_McNet_seconds':float((m+t).mean()) if system=='mcnet_mask_mvdr' else None,'notice':'Different process/device contexts; approximate sum, not measured end-to-end pipeline latency.'}
    checks={'all2700_new_signal_pairs_unique':len(new)==2700 and len({key(r) for r in new})==2700,
      'all17_systems_on_all900_cases':all(n==900 for n in Counter(r['system'] for r in signals).values()) and len({r['system'] for r in signals})==17,
      'all_new_metric_rows_ok':all(r['status']=='ok' for r in new),
      'all_new_output_hashes':all(sha(r['output_path'])==r['output_sha256'] for r in new),
      'all_new_input_and_reference_hashes':all(sha(r['noisy_path'])==r['input_sha256'] and sha(r['reference_path'])==r['reference_sha256'] for r in new),
      'both_new_ASR_exact_pair_coverage':{key(r) for r in new}=={key(r) for r in whisper_new}=={key(r) for r in ctc_new},
      'both_new_ASR_scores_recompute':all(r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis']) for r in whisper_new+ctc_new)}
    original={key(r):r for r in signals if r['system']=='mcnet'}
    checks['main_mask_input_is_frozen_McNet_not_clean_or_DOA']=all(r['mask_estimate_path']==original[(r['dataset'],'mcnet',r['sample_id'])]['output_path'] and not r['beamforming_diagnostics']['oracle_clean_reference'] and not r['beamforming_diagnostics']['oracle_DOA'] for r in new if r['system']=='mcnet_mask_mvdr')
    a=json.loads((OUT/'evaluation/signal_provenance.json').read_text());p=json.loads((OUT/'stages/preflight/archive_manifest.json').read_text());s=json.loads((OUT/'stages/signal/archive_manifest.json').read_text());w=json.loads((OUT/'evaluation/Whisper/provenance.json').read_text());c=json.loads((OUT/'evaluation/CTC/provenance.json').read_text())
    checks.update(preflight_archived_before_benchmark=p['saved_utc']<a['started_utc'],signal_report_before_Whisper=s['saved_utc']<w['started_utc'],
      Whisper_report_before_CTC=json.loads((OUT/'stages/whisper/archive_manifest.json').read_text())['saved_utc']<c['started_utc'])
    selected=json.loads((ROOT/'artifacts/g5_rotation_v1/selected_checkpoints.json').read_text())['systems'];checks['all12_adapters_unchanged']=all(sha(ROOT/r['path'])==r['checkpoint_sha256'] for r in selected)
    bridge=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g9_integrity');repo=ROOT/'third_party/McNet';bridge['verify_upstream'](repo,repo/bridge['OFFLINE_CHECKPOINT']);checks['official_base_unchanged']=True
    audit={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'coverage':summary['coverage'],'completed_utc':now()};dump(OUT/'completion_audit.json',audit,True)
    if audit['status']!='ok':raise RuntimeError('G9 audit failed, preserve evidence')
    dump(OUT/'evaluation/summary.json',summary,True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(17,7),layout='constrained');curves=[('mcnet','McNet','#777777','--'),('rotated_without_clean_identity','rotated LoRA,no identity','#4769a8','-'),('mcnet_mask_mvdr','estimated-mask MVDR','#248669','-'),('oracle_ref_mask_mvdr','oracle-mask MVDR','#b2573a','--'),('oracle_doa_delay_sum','oracle-DOA DS','#8a55a5',':')]
    for col,(ds,noise) in enumerate((ds,n) for ds,items in summary['cells'].items() for n in items):
        for group,label,color,style in curves:
            values=summary['cells'][ds][noise][group];x=[-40,20,70]
            axes[0,col].plot(x,[values[str(t)]['metrics']['si_sdr_db'] for t in x],style+'o',label=label,color=color)
            axes[1,col].plot(x,[values[str(t)]['CTC']['wer']*100 for t in x],style+'o',color=color)
        axes[0,col].set_title(ds+'/'+noise)
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xlabel('Target azimuth(deg)')
    axes[0,0].set_ylabel('SI-SDR(dB)');axes[1,0].set_ylabel('CTC WER(%)');axes[0,0].legend(fontsize=6);fig.suptitle('Existing12-source simulation benchmark; oracle methods use extra truth')
    fig.savefig(OUT/'evaluation/beamforming_profiles.png',dpi=170);plt.close(fig)
    lines=['# G9 成熟波束形成对照与机制诊断报告','','## 范围','','G8验收归档后冻结协议，G9A五项前置检查及G9B信号报告均在后续步骤前保存。新增3系统×900条件=2,700条信号、每识别器2,700条评分。与旧14系统合并17系统，每后端15,300行；源句仍仅12句/3人。G5/G7已查看条件上的探索性比较，不是新盲测或真实录音验证。',
      '', '主对照mcnet_mask_mvdr仅使用六通道带噪输入及原McNet估计。oracle_ref_mask_mvdr使用干净参考估计掩码，oracle_doa_delay_sum使用真实方位；两项为特权信息诊断，不是公平盲方法，也不保证数值上优于所有估计方法。无新训练、依赖、语料或权重。',
      '', '## 方向结果','','每格为SI-SDR dB / Whisper WER / CTC WER。旧适配器完整合并三种子，无最优种子挑选。G7合并−5/0/+5dB，G5仅0dB；逐SNR、场型、说话人与PESQ/STOI保存在机器汇总。以下仅描述点估计，不作额外显著性检验。','','|数据/噪声|系统组|−40°|20°|70°|','|---|---|---:|---:|---:|']
    for ds,noises in summary['cells'].items():
        for noise,items in noises.items():
            for group,values in items.items():lines.append('|'+ds+'/'+noise+'|'+group+'|'+'|'.join(f'{values[str(t)]["metrics"]["si_sdr_db"]:.2f} / {values[str(t)]["Whisper"]["wer"]:.1%} / {values[str(t)]["CTC"]["wer"]:.1%}' for t in (-40,20,70))+'|')
    lines+=['','![方向对照](../artifacts/g9_beamforming_v1/evaluation/beamforming_profiles.png)','','## 干净输入保真','','复用G5的36条件；同12句的不同方向不能当额外独立句。WER变化是识别器任务表现，PESQ/STOI或能量变化不直接等于内容丢失。','','|系统组|SI-SDR|PESQ|STOI|Whisper WER|CTC WER|','|---|---:|---:|---:|---:|---:|']
    for group,r in summary['clean'].items():lines.append(f'|{group}|{r["metrics"]["si_sdr_db"]:.2f}|{r["metrics"]["pesq_wb"]:.3f}|{r["metrics"]["stoi"]:.3f}|{r["Whisper"]["wer"]:.1%}|{r["CTC"]["wer"]:.1%}|')
    lines+=['','## 计算成本','','CPU双精度离线整句滤波，不是流式实时系统。滤波计时包含本函数STFT/ISTFT，不含读盘/加载/指标；估计掩码主对照还要运行McNet，以下两段成本之和来自不同上下文，只是近似，不能代替端到端时延。','','|系统|滤波平均秒|滤波P90秒|滤波平均RTF|还需原McNet平均秒|','|---|---:|---:|---:|---:|']
    for group,r in summary['timing'].items():lines.append(f'|{group}|{r["mean_filter_seconds"]:.4f}|{r["p90_filter_seconds"]:.4f}|{r["mean_filter_RTF"]:.4f}|{r["mean_previous_McNet_seconds"] if r["requires_McNet"] else "不需要，但有真值信息"}|')
    lines+=['','## 解释、风险与审计','','若估计掩码MVDR已优于轻量适配，则不能只凭优于原McNet认定改进价值；应如实保留更强成熟基线。若只有oracle改善，可用于提示掩码或方位信息限制，但不能直接推导可部署效果。所有条件上的模拟阵列/无混响假设仍限制实际应用价值。',
      '', '本阶段没有新的人工吞字标签，也没有新颖性证据。全部原权重和已有识别证据冻结；主对照输入来源、2,700条新结果、双ASR覆盖/重计分、音频哈希和阶段归档顺序审计为'+audit['status']+'。数值异常不回填；有效行均真实计分。',
      '', '[G9A协议与QA](G9A_成熟波束形成对照协议与验证.md)、[G9B信号与耗时](G9B_波束形成信号与耗时报告.md)、[Whisper初评](G9C_Whisper波束形成初评.md)、[机器汇总](../artifacts/g9_beamforming_v1/evaluation/summary.json)、[审计](../artifacts/g9_beamforming_v1/completion_audit.json)。',
      '', '方法依据：[torchaudio2.5 Souden API](https://docs.pytorch.org/audio/2.5.0/generated/torchaudio.functional.mvdr_weights_souden.html)、[官方教程](https://docs.pytorch.org/audio/2.5.0/tutorials/mvdr_tutorial.html)、[Souden等IEEE TASLP2010](https://doi.org/10.1109/TASL.2009.2025790)。','', 'UTC：'+now()]
    report=ROOT/'docs/G9_成熟波束形成对照与机制诊断报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('completion',report,[OUT/'completion_audit.json',OUT/'evaluation/summary.json',OUT/'evaluation/Whisper/provenance.json',OUT/'evaluation/CTC/provenance.json',Path(__file__)])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'stage':'G9','completed_utc':now(),'status':'ok','report':str(report.relative_to(ROOT))},True);status(stage='complete')


if __name__=='__main__':finish()
