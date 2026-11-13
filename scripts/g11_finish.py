#!/usr/bin/env python3
"""Complete seed coverage and conditional component-response report."""
import json,runpy
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from g11_common import ROOT,OUT,G10,now,sha,read,dump,status,archive,selected_rows
from mcse.asr_scoring import corpus_score

METRICS=('speech_error_nmse','target_gain_db','noise_reduction_db','component_output_snr_db')


def aggregate(rows):
    assert rows
    optional=[r['metrics']['target_component_si_sdr_db'] for r in rows if r['metrics']['target_component_si_sdr_db'] is not None]
    return {'records':len(rows),'sources':len({r['source_id'] for r in rows}),'speakers':len({r['speaker_id'] for r in rows}),
      'mean':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in METRICS},'median':{k:float(np.median([r['metrics'][k] for r in rows])) for k in METRICS},
      'target_component_SI_SDR_mean_valid':float(np.mean(optional)) if optional else None,'target_component_SI_valid_count':len(optional),
      'speech_output_power_floored':sum(r['metrics']['output_speech_power_floored'] for r in rows),'noise_output_power_floored':sum(r['metrics']['output_noise_power_floored'] for r in rows),
      'mean_input_component_SNR_db':float(np.mean([r['metrics']['component_input_snr_db'] for r in rows])),
      'Whisper':corpus_score(r['Whisper'] for r in rows),'CTC':corpus_score(r['CTC'] for r in rows)}


def compare(rows,a,b):
    speakers=sorted({r['speaker_id'] for r in rows});assert len(speakers)==4
    def matrix(group):
        return np.array([[np.mean([r['metrics'][k] for r in rows if r['group']==group and r['speaker_id']==sp]) for k in METRICS] for sp in speakers])
    left,right=matrix(a),matrix(b);assert np.isfinite(left).all() and np.isfinite(right).all()
    difference=left-right;point=difference.mean(0);rng=np.random.default_rng(20261112);boots=difference[rng.integers(4,size=(2000,4))].mean(1)
    lo,hi=np.quantile(boots,[.025,.975],axis=0)
    return {'a':a,'b':b,'speakers':4,'delta':{k:float(point[i]) for i,k in enumerate(METRICS)},
      'speaker_cluster_bootstrap2000_ci95':{k:[float(lo[i]),float(hi[i])] for i,k in enumerate(METRICS)},
      'per_speaker':{sp:{k:float(difference[j,i]) for i,k in enumerate(METRICS)} for j,sp in enumerate(speakers)}}


def finish(rows=None):
    if rows is None:rows=read(OUT/'analysis/components.jsonl')
    plan=json.loads((OUT/'frozen_plan.json').read_text());noises=('OOFFICE','PCAFETER','NPARK','PSTATION');groups=('original','fixed','rotated')
    summary={'coverage':dict(Counter(r['group'] for r in rows)),'sources':16,'speakers':4,'seeds':[20261112,20261113,20261114],
      'definition':'Arithmetic mean per-case NMSE; arithmetic mean of per-case dB ratios,not dB of pooled power. Four-speaker intervals conditional on existing cases/seeds.',
      'cells':{},'per_seed':{},'per_field':{},'per_speaker':{},'comparisons':{}}
    for noise in noises:
        subset=[r for r in rows if r['noise_id']==noise];summary['cells'][noise]={g:{str(t):aggregate([r for r in subset if r['group']==g and r['target_azimuth']==t]) for t in (-40,20,70)} for g in groups}
        for t in (-40,20,70):
            selection=[r for r in subset if r['target_azimuth']==t]
            for a,b in [('rotated','fixed'),('fixed','original'),('rotated','original')]:summary['comparisons'][f'{noise}|{t}|{a}-{b}']=compare(selection,a,b)
    summary['per_seed']={s:{noise:{str(t):aggregate([r for r in rows if r['system']==s and r['noise_id']==noise and r['target_azimuth']==t]) for t in (-40,20,70)} for noise in noises} for s in sorted({r['system'] for r in rows})}
    summary['per_field']={g:{f:aggregate([r for r in rows if r['group']==g and r['condition']==f]) for f in ('point','multi8')} for g in groups}
    summary['per_speaker']={g:{sp:aggregate([r for r in rows if r['group']==g and r['speaker_id']==sp]) for sp in sorted({r['speaker_id'] for r in rows})} for g in groups}
    verified=json.loads((OUT/'analysis/verified_source_files.json').read_text());expected={(r['system'],r['sample_id']) for r in selected_rows()}
    wi={(r['system'],r['sample_id']):r for r in read(G10/'evaluation/Whisper/asr_scores.jsonl')};ci={(r['system'],r['sample_id']):r for r in read(G10/'evaluation/CTC/asr_scores.jsonl')}
    checks={'all4032_exact_pairs':len(rows)==4032 and {(r['system'],r['sample_id']) for r in rows}==expected,
      'all_reference_exact':all(r['reference_reconstruction_max_abs']==0 for r in rows),
      'all_beams_reproduced':all(r['reproduced_beam_max_abs']<=plan['saved_beam_match_max_abs'] for r in rows),
      'all_linearity_and_energy_accounting':all(r['metrics']['linearity_max_abs']<=r['linearity_tolerance'] and r['metrics']['snr_accounting_error_db']<=1e-8 for r in rows),
      'all_primary_component_metrics_finite':all(np.isfinite(r['metrics'][k]) for r in rows for k in METRICS),
      'all_prior_input_estimate_output_hashes':all(sha(p)==h for p,h in verified.items()),
      'all_frozen_metadata_and_source_code_unchanged':all(sha(ROOT/p)==h for p,h in plan['source_evidence_sha256'].items()),
      'ASR_joins_equal_old_counts':all(all(r['Whisper'][k]==wi[(r['system'],r['sample_id'])]['score'][k] and r['CTC'][k]==ci[(r['system'],r['sample_id'])]['score'][k] for k in ('N','S','D','I','correct','errors','wer')) for r in rows),
      'three_complete_adapter_seeds':{r['seed'] for r in rows if r['group']!='original'}=={20261112,20261113,20261114} and all(n==576 for n in Counter(r['system'] for r in rows).values()),
      'CPU_only_CUDA_never_initialized':not torch.cuda.is_initialized()}
    provenance=json.loads((OUT/'analysis/provenance.json').read_text());preflight=json.loads((OUT/'stages/preflight/archive_manifest.json').read_text());checks['preflight_archived_before_full_analysis']=preflight['saved_utc']<provenance['started_utc']
    registry=json.loads((G10/'systems.json').read_text());checks['six_adapters_still_unchanged']=all(r['kind']!='adapter' or sha(ROOT/r['path'])==r['checkpoint_sha256'] for r in registry['direct'])
    bridge=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g11_integrity');repo=ROOT/'third_party/McNet';bridge['verify_upstream'](repo,repo/bridge['OFFLINE_CHECKPOINT']);checks['official_base_still_unchanged']=True
    audit={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'rows':len(rows),'completed_utc':now(),
      'max_reference_error':max(r['reference_reconstruction_max_abs'] for r in rows),'max_old_output_reproduction_error':max(r['reproduced_beam_max_abs'] for r in rows),
      'max_linearity_error':max(r['metrics']['linearity_max_abs'] for r in rows),'no_new_training_or_ASR':True}
    dump(OUT/'completion_audit.json',audit,True)
    if audit['status']!='ok':raise RuntimeError('G11 completion audit failed,preserve stage')
    dump(OUT/'analysis/summary.json',summary,True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,4,figsize=(17,10),layout='constrained')
    for col,noise in enumerate(noises):
        for group,color in [('original','#777777'),('fixed','#b2573a'),('rotated','#4769a8')]:
            v=summary['cells'][noise][group];x=[-40,20,70]
            for row,metric in enumerate(('target_gain_db','noise_reduction_db','speech_error_nmse')):
                y=[v[str(t)]['mean'][metric] for t in x]
                if metric=='speech_error_nmse':y=[max(a,1e-12) for a in y]
                axes[row,col].plot(x,y,'-o',label=group,color=color)
        axes[0,col].set_title(noise);axes[2,col].set_yscale('log')
        for ax in axes[:,col]:ax.set_xlabel('Target azimuth(deg)');ax.grid(alpha=.2)
    for ax,label in zip(axes[:,0],('Target component gain(dB)','Noise reduction(dB)','Target distortion NMSE(log)')):ax.set_ylabel(label)
    axes[0,0].legend(fontsize=7);fig.suptitle('Frozen conditional MVDR response;16 reused sources,4 speakers,3 adapter seeds')
    fig.savefig(OUT/'analysis/component_profiles.png',dpi=170);plt.close(fig)
    lines=['# G11 固定MVDR的目标语音与残余噪声诊断','','## 能回答什么','','在G10已评估16句/4人上，重建模拟干净六通道，取N=X−S。只用原带噪输入与旧神经估计重算每个已冻结MVDR滤波器，确认旧输出后固定W，分别计算W作用于S与N的响应。真值不参与掩码估计，不产生新的可部署oracle增强器；不训练、不下载、不重新识别。',
      '', '7前端×576带噪条件=4,032条：原始前端576，固定/旋转前端各三种子1,728。保留全部种子。该分解只针对已确定的线性滤波器，不能直接分解非线性神经网络的直接输出，也不能据此证明某个词/辅音被删除。它有助于解释目标失真与噪声压低的区别，不是新增强算法或独立新测试。',
      '', '## 指标定义','','目标失真NMSE=mean((目标分量输出−干净参考)^2)/mean(干净参考^2)，小者好，包含尺度误差；目标增益=10log10(输出目标功率/输入目标功率)，0dB是总体能量保持，但不保证细节；噪声衰减=10log10(输入噪声功率/输出噪声功率)，大者好；分量SNR比较的是输出目标和残余噪声功率，必须与失真一起读。全句功率不等于原活动语音标定0dB。',
      '', '表内NMSE为逐例比值的算术均值；dB为逐例dB均值，并非合并功率后的比值。纯增益衰减可能几乎不影响SI-SDR，所以另列目标增益。1e-30功率下限及未定义目标SI-SDR显式记数；权重范数只作白噪声功率放大诊断，不当作未经验证单位目标响应下的WNG。','','|噪声|掩码前端组|方向|目标NMSE|目标增益dB|噪声衰减dB|分量输出SNR dB|旧Whisper/CTC WER|','|---|---|---:|---:|---:|---:|---:|---:|']
    for noise in noises:
        for group in groups:
            for t in (-40,20,70):
                v=summary['cells'][noise][group][str(t)];m=v['mean'];lines.append(f'|{noise}|{group}|{t}|{m["speech_error_nmse"]:.3f}|{m["target_gain_db"]:.2f}|{m["noise_reduction_db"]:.2f}|{m["component_output_snr_db"]:.2f}|{v["Whisper"]["wer"]:.1%} / {v["CTC"]["wer"]:.1%}|')
    lines+=['','![分量方向曲线](../artifacts/g11_components_v1/analysis/component_profiles.png)','','## 旋转减固定：条件分量差异','','按4位说话人有放回抽样2,000次，每人保留全部句子/条件/种子，先在每人组内算指标均值。每人4句、相同条件数，因此与本表总体均值配对。区间仅为小簇探索性描述，未作多重比较推断。NMSE负差好；增益更接近0不能单独当作更优；噪声衰减/分量SNR正差好。','','|噪声/方向|Δ目标NMSE [区间]|Δ目标增益dB [区间]|Δ噪声衰减dB [区间]|','|---|---|---|---|']
    for noise in noises:
        for t in (-40,20,70):
            value=summary['comparisons'][f'{noise}|{t}|rotated-fixed']
            def fmt(metric):
                lo,hi=value['speaker_cluster_bootstrap2000_ci95'][metric];return f'{value["delta"][metric]:+.3f} [{lo:+.3f},{hi:+.3f}]'
            lines.append(f'|{noise}/{t}|{fmt("speech_error_nmse")}|{fmt("target_gain_db")}|{fmt("noise_reduction_db")}|')
    lines+=['','## 完整性、边界与后续','','全部4,032条参考重建误差为0；最大旧输出复现误差'+str(audit['max_old_output_reproduction_error'])+'，最大线性误差'+str(audit['max_linearity_error'])+'。输入、估计、旧输出、源FLAC、原始主干与六适配器哈希通过，原ASR计分仅关联未重跑，CUDA未初始化。审计：'+audit['status']+'。',
      '', '量化S后定义N=X−S，N包括混合/干净量化差额；这与假设拥有无误差真实录音分量不同。能量保留或失真不等于音素/语义保留，旧WER仅并列展示，不据相关性作因果断言。直接神经输出与MVDR之间的全部差异仍不能由本分解完全解释。仅首批合成阵列/0dB/四录音/四说话人，缺真实混响、实际硬件与新的人工盲听证据。',
      '', '如果某组通过更少压低噪声换来更少失真，应如实称为权衡；若同时改善，需要在其他独立环境继续确认。不能从机制诊断直接宣称LoRA＋MVDR的新颖性。接下来的算法选择应先依据失败条件与已有研究对照，不能把本批重新当未查看最终测试。',
      '', '[预检与定义](G11A_分量诊断协议与还原验证.md)、[完整分量核对](G11B_完整分量诊断核对记录.md)、[机器汇总](../artifacts/g11_components_v1/analysis/summary.json)、[审计](../artifacts/g11_components_v1/completion_audit.json)。',
      '', '经典依据：[Souden等，On Optimal Frequency-Domain Multichannel Linear Filtering for Noise Reduction，IEEE TASLP2010](https://doi.org/10.1109/TASL.2009.2025790)。本阶段使用明确声明的全句经验指标，并非复现其全部理论或证明。','', 'UTC：'+now()]
    report=ROOT/'docs/G11_固定MVDR目标语音与残余噪声诊断报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('completion',report,[OUT/'completion_audit.json',OUT/'analysis/summary.json',OUT/'analysis/provenance.json',ROOT/'scripts/g11_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'stage':'G11','status':'ok','completed_utc':now(),'report':str(report.relative_to(ROOT))},True);status(stage='complete')


if __name__=='__main__':finish()
