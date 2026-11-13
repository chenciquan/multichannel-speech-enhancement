#!/usr/bin/env python3
"""Fresh-speaker paired output-mode analysis, cluster intervals, final audit."""
import json,runpy
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf
from g10_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive
from mcse.asr_scoring import score_words,corpus_score


def cluster_effect(signals,wi,ci,a,b):
    speakers=sorted({r['speaker_id'] for r in signals});assert len(speakers)==4
    def matrix(names):
        values=[]
        for sp in speakers:
            rows=[r for r in signals if r['speaker_id']==sp and r['system'] in names];assert rows
            w=corpus_score(wi[(r['system'],r['sample_id'])]['score'] for r in rows);c=corpus_score(ci[(r['system'],r['sample_id'])]['score'] for r in rows)
            values.append([sum(r['metrics']['si_sdr_db'] for r in rows),len(rows),w['errors'],w['N'],c['errors'],c['N']])
        return np.array(values,dtype=float)
    left,right=matrix(a),matrix(b)
    def ratio(v):return np.stack([v[...,0]/v[...,1],v[...,2]/v[...,3],v[...,4]/v[...,5]],axis=-1)
    point=ratio(left.sum(0))-ratio(right.sum(0));rng=np.random.default_rng(20261112);ids=rng.integers(4,size=(2000,4))
    boots=ratio(left[ids].sum(1))-ratio(right[ids].sum(1));lo,hi=np.quantile(boots,[.025,.975],axis=0)
    return {'speakers':4,'a':a,'b':b,'delta':{k:float(point[i]) for i,k in enumerate(('si_sdr_db','Whisper_WER','CTC_WER'))},
      'speaker_cluster_bootstrap2000_ci95':{k:[float(lo[i]),float(hi[i])] for i,k in enumerate(('si_sdr_db','Whisper_WER','CTC_WER'))},
      'per_speaker':{sp:{k:float(x) for k,x in zip(('si_sdr_db','Whisper_WER','CTC_WER'),ratio(left[j])-ratio(right[j]))} for j,sp in enumerate(speakers)}}


def finish():
    plan=json.loads((OUT/'frozen_plan.json').read_text());registry=json.loads((OUT/'systems.json').read_text());signals=read(OUT/'evaluation/combined_signals.jsonl');whisper=read(OUT/'evaluation/Whisper/asr_scores.jsonl');ctc=read(OUT/'evaluation/CTC/asr_scores.jsonl')
    key=lambda r:(r['system'],r['sample_id']);wi={key(r):r for r in whisper};ci={key(r):r for r in ctc};si={key(r):r for r in signals}
    assert len(signals)==len(wi)==len(ci)==9360 and set(si)==set(wi)==set(ci)
    groups={'noisy':['noisy'],'mcnet':['mcnet'],'fixed':[s['system'] for s in registry['direct'] if s.get('geometry')=='fixed'],
       'rotated':[s['system'] for s in registry['direct'] if s.get('geometry')=='rotated']}
    groups.update({name+'_mvdr':[s+'_mask_mvdr' for s in groups[name]] for name in ('mcnet','fixed','rotated')})
    def aggregate(rows):
        return {'records':len(rows),'sources':len({r['source_id'] for r in rows}),'speakers':len({r['speaker_id'] for r in rows}),
          'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('si_sdr_db','pesq_wb','stoi')},
          'Whisper':corpus_score(wi[key(r)]['score'] for r in rows),'CTC':corpus_score(ci[key(r)]['score'] for r in rows)}
    summary={'coverage':{'systems':15,'conditions_per_system':624,'signal_and_each_ASR_rows':9360,'sources':16,'speakers':4,'SNR_db':[0]},
      'cells':{},'clean':{},'per_speaker':{},'per_field':{},'output_mode_effects':{},'geometry_effects':{},'reference_ASR':{},'timing':{}}
    noisy=[r for r in signals if r['condition']!='clean']
    for noise in plan['noise_ids']:
        subset=[r for r in noisy if r['noise_id']==noise];summary['cells'][noise]={}
        for group,names in groups.items():summary['cells'][noise][group]={str(t):aggregate([r for r in subset if r['system'] in names and r['target_azimuth']==t]) for t in (-40,20,70)}
        for target in (-40,20,70):
            rows=[r for r in subset if r['target_azimuth']==target]
            for model in ('mcnet','fixed','rotated'):
                summary['output_mode_effects'][f'{noise}|{target}|{model}']=cluster_effect(rows,wi,ci,groups[model+'_mvdr'],groups[model])
            for mode in ('direct','mvdr'):
                suffix='' if mode=='direct' else '_mvdr';summary['geometry_effects'][f'{noise}|{target}|{mode}']=cluster_effect(rows,wi,ci,groups['rotated'+suffix],groups['fixed'+suffix])
    for group,names in groups.items():
        summary['clean'][group]=aggregate([r for r in signals if r['system'] in names and r['condition']=='clean'])
        summary['per_speaker'][group]={sp:aggregate([r for r in noisy if r['system'] in names and r['speaker_id']==sp]) for sp in sorted({r['speaker_id'] for r in noisy})}
        summary['per_field'][group]={field:aggregate([r for r in noisy if r['system'] in names and r['condition']==field]) for field in ('point','multi8')}
    refs={}
    for r in ctc:
        if r['condition']=='clean':refs.setdefault(r['source_id'],(r,wi[key(r)]))
    assert len(refs)==16
    summary['reference_ASR']={'one_clean_reference_per_source':16,'Whisper':corpus_score(v[1]['reference_score'] for v in refs.values()),'CTC':corpus_score(v[0]['reference_score'] for v in refs.values()),
      'notice':'Unscaled-equivalent clean-condition reference once per source; ASR baseline,not guaranteed WER lower bound.'}
    for group,names in groups.items():
        rows=[r for r in signals if r['system'] in names]
        summary['timing'][group]={'mean_stage_seconds':float(np.mean([r['inference_seconds'] for r in rows])),
          'mean_previous_direct_seconds':float(np.mean([r['direct_inference_seconds'] for r in rows])) if group.endswith('_mvdr') else None,
          'notice':'Pure stage timer in differing device contexts,not measured end-to-end latency or streaming claim.'}
    source=read(BASE/'sources/speech_manifest.jsonl');exclusions=json.loads((BASE/'sources/status.json').read_text())['excluded'];cases=read(BASE/'manifest.jsonl');paths={}
    for r in signals:
        for p,h in [(r['output_path'],r['output_sha256']),(r['noisy_path'],r['input_sha256']),(r['reference_path'],r['reference_sha256'])]:
            if p in paths:assert paths[p]==h
            paths[p]=h
    inputok=all(sha(p)==h for p,h in paths.items());dump(OUT/'evaluation/readback_file_hashes.json',paths,True)
    canonical=defaultdict(set)
    for r in cases:
        x,sr=sf.read(r['noisy_path'],dtype='float32',always_2d=True);s,rr=sf.read(r['reference_path'],dtype='float32');assert sr==rr==16000
        import hashlib
        canonical[r['counterfactual_group']].add((hashlib.sha256(x[:,4].tobytes()).hexdigest(),hashlib.sha256(s.tobytes()).hexdigest()))
    expected={(s['system'],r['sample_id']) for s in registry['direct']+registry['beam'] for r in cases}
    checks={'all9360_unique_signal_ASR_pairs':set(si)==set(wi)==set(ci)==expected,'all_metrics_ok':all(r['status']=='ok' for r in signals),
      'all_saved_file_hashes_match':bool(inputok),'all_input_reference_angle_groups_identical':len(canonical)==208 and all(len(v)==1 for v in canonical.values()),
      'all16_fresh_sources4_new_speakers':len(source)==16 and len({r['speaker_id'] for r in source})==4 and all(n==4 for n in Counter(r['speaker_id'] for r in source).values()) and all(r[k] not in exclusions[k] for r in source for k in ('speaker_id','source_id','sha256')),
      'fresh_source_files_match':all(sha(BASE/'sources'/r['path'])==r['sha256'] for r in source),
      'all6_frozen_adapters_match':all(r['kind']!='adapter' or sha(ROOT/r['path'])==r['checkpoint_sha256'] for r in registry['direct']),
      'beam_code_unchanged':sha(ROOT/'src/mcse/beamforming.py')==plan['beamforming_code_sha256'],
      'both_ASR_edit_scores_recompute':all(r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis']) for r in whisper+ctc)}
    direct_index={(r['system'],r['sample_id']):r for r in signals if not r['system'].endswith('_mask_mvdr')}
    checks['every_MVDR_mask_from_corresponding_neural_estimate']=all(r['mask_estimate_path']==direct_index[(r['source_system'],r['sample_id'])]['output_path'] and r['mask_estimate_sha256']==direct_index[(r['source_system'],r['sample_id'])]['output_sha256'] and not r['beamforming_diagnostics']['oracle_clean_reference'] and not r['beamforming_diagnostics']['oracle_DOA'] for r in signals if r['system'].endswith('_mask_mvdr'))
    prep=json.loads((OUT/'stages/preparation/archive_manifest.json').read_text());sp=json.loads((OUT/'evaluation/provenance.json').read_text());beam=json.loads((OUT/'evaluation/beam_provenance.json').read_text());sa=json.loads((OUT/'stages/signal/archive_manifest.json').read_text())
    checks.update(data_archived_before_neural=prep['saved_utc']<sp['started_utc'],direct_report_before_MVDR=sa['saved_at_utc']<beam['started_utc'],
      MVDR_report_before_Whisper=json.loads((OUT/'stages/beamforming/archive_manifest.json').read_text())['saved_utc']<json.loads((OUT/'evaluation/Whisper/provenance.json').read_text())['started_utc'],
      Whisper_report_before_CTC=json.loads((OUT/'stages/whisper/archive_manifest.json').read_text())['saved_utc']<json.loads((OUT/'evaluation/CTC/provenance.json').read_text())['started_utc'])
    bridge=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g10_integrity');repo=ROOT/'third_party/McNet';bridge['verify_upstream'](repo,repo/bridge['OFFLINE_CHECKPOINT']);checks['original_base_unchanged']=True
    audit={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'coverage':summary['coverage'],'completed_utc':now(),'new_training':False};dump(OUT/'completion_audit.json',audit,True)
    if audit['status']!='ok':raise RuntimeError('G10 audit failed, keep evidence')
    dump(OUT/'evaluation/summary.json',summary,True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(17,7),layout='constrained')
    for col,noise in enumerate(plan['noise_ids']):
        for model,color in [('mcnet','#777777'),('fixed','#b2573a'),('rotated','#4769a8')]:
            for suffix,style in [('', '--'),('_mvdr','-')]:
                v=summary['cells'][noise][model+suffix];x=[-40,20,70]
                axes[0,col].plot(x,[v[str(t)]['metrics']['si_sdr_db'] for t in x],style+'o',color=color,label=model+suffix)
                axes[1,col].plot(x,[v[str(t)]['CTC']['wer']*100 for t in x],style+'o',color=color)
        axes[0,col].set_title(noise)
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xlabel('Target azimuth(deg)')
    axes[0,0].set_ylabel('SI-SDR(dB)');axes[1,0].set_ylabel('CTC WER(%)');axes[0,0].legend(fontsize=6);fig.suptitle('16 fresh utterances,4 speakers; same frozen adapters,no new training')
    fig.savefig(OUT/'evaluation/fresh_voice_profiles.png',dpi=170);plt.close(fig)
    lines=['# G10 新说话人直接输出与MVDR机制验证报告','','## 范围与独立性','','G9验收归档后冻结方法、系统、源选择及条件。按档案顺序取得4位此前未用于本项目的说话人，每人4句完整短语音，共16句。排除所有G1/G2/G3/G5说话人/语句ID/源SHA。当前只验证增强实验的说话人隔离，不保证公开内容从未出现在Whisper预训练中。',
      '', '既有四种噪声、0dB、三目标方向、两点噪声相对方向/八源场，576带噪+48干净条件/系统。8直接输出+7估计掩码MVDR=15系统，共9,360信号及每个识别器9,360评分。新增4位说话人，条件记录与三种子不是额外独立说话人。仍为无混响名义六麦模拟、英语朗读；未训练或用新语音选择权重，未添加oracle主对照。',
      '', '收窄到无干净约束分支是基于已查看G5—G9证据的机制问题选择；不能由此证明其他分支普遍无效，也不能把成熟LoRA+MVDR组合称为已建立创新。',
      '', '## 主要方向结果','','每格为SI-SDR dB / Whisper WER / CTC WER，适配器合并全部三种子。全部PESQ/STOI、每说话人、场型及逐例S/D/I保存在机器文件。WER保留大于100%的值，不当作人声丢失比例。','','|噪声|系统组|−40°|20°|70°|','|---|---|---:|---:|---:|']
    for noise,models in summary['cells'].items():
        for model,v in models.items():lines.append('|'+noise+'|'+model+'|'+'|'.join(f'{v[str(t)]["metrics"]["si_sdr_db"]:.2f} / {v[str(t)]["Whisper"]["wer"]:.1%} / {v[str(t)]["CTC"]["wer"]:.1%}' for t in (-40,20,70))+'|')
    lines+=['','![新说话人方向曲线](../artifacts/g10_fresh_voice_v1/evaluation/fresh_voice_profiles.png)','','## MVDR减直接输出：同前端配对效应','','按4位说话人有放回抽样2,000次，固定随机种子；每次保留该人的全部句子、条件和训练种子，计数后重新计算微平均WER。这里只是少量簇的探索性描述区间，非多重比较控制显著性检验。SI正值好，WER负值好。逐人效应和几何分布效应另存机器汇总。','','|噪声/方向|前端|ΔSI-SDR [区间]|ΔWhisper WER百分点 [区间]|ΔCTC WER百分点 [区间]|','|---|---|---|---|---|']
    for name,r in summary['output_mode_effects'].items():
        def fmt(metric,scale=1):
            v=r['delta'][metric];lo,hi=r['speaker_cluster_bootstrap2000_ci95'][metric];return f'{v*scale:+.2f} [{lo*scale:+.2f},{hi*scale:+.2f}]'
        noise,t,model=name.split('|');lines.append(f'|{noise}/{t}|{model}|{fmt("si_sdr_db")}|{fmt("Whisper_WER",100)}|{fmt("CTC_WER",100)}|')
    lines+=['','## 干净输入与识别基准','','原始干净参考每源句只计一次，共16句；这是识别器参考基准，不是保证的WER下界。Whisper '+f'{summary["reference_ASR"]["Whisper"]["wer"]:.1%}'+'，CTC '+f'{summary["reference_ASR"]["CTC"]["wer"]:.1%}'+'。','','|系统组|SI-SDR|PESQ|STOI|Whisper WER|CTC WER|','|---|---:|---:|---:|---:|---:|']
    for model,r in summary['clean'].items():lines.append(f'|{model}|{r["metrics"]["si_sdr_db"]:.2f}|{r["metrics"]["pesq_wb"]:.3f}|{r["metrics"]["stoi"]:.3f}|{r["Whisper"]["wer"]:.1%}|{r["CTC"]["wer"]:.1%}|')
    lines+=['','## 解释与后续限制','','输出形式如果提高SI-SDR却让一项或两项WER变差，应保留该权衡；不能凭波形指标断定语音内容更完整。两个ASR相同方向仍不等于真实词语或辅音保留，当前没有新的人工逐词或盲听标签。负结果也可能说明掩码不足、整句PSD或输入域失配，需要独立机制证据，不能自动归因某一模块。',
      '', '只有四人、固定0dB和已知四录音，缺真实混响、硬件阵列和中文实用场景。后续修改任何权重或方法须另立开发/验证数据，已查看本批不得再次充当未经触碰的最终测试。不能保证论文发表。',
      '', '## 归档与审计','', '源报告→数据→直接信号→MVDR信号→Whisper→CTC联合报告依序保存；完整覆盖、音频/源/权重哈希、208组参考麦克风一致性、无oracle掩码来源和双ASR重计分审计为'+audit['status']+'。原McNet代码及权重未改，无新增依赖或训练。滤波成本还需加此前端推理，机器汇总的分段时耗不是实测端到端时延。',
      '', '[源选择](G10A_新说话人语料选择记录.md)、[数据协议](G10B_新说话人匹配模拟协议与数据.md)、[直接信号](G10C_新说话人直接输出信号报告.md)、[MVDR信号](G10D_新说话人MVDR输出信号报告.md)、[Whisper初评](G10E_Whisper新说话人初评.md)、[机器汇总](../artifacts/g10_fresh_voice_v1/evaluation/summary.json)、[审计](../artifacts/g10_fresh_voice_v1/completion_audit.json)。',
      '', '来源：[OpenSLR12 CC-BY4.0](https://www.openslr.org/12)、[DEMAND官方记录](https://zenodo.org/records/1227121)、[torchaudio Souden API](https://docs.pytorch.org/audio/2.5.0/generated/torchaudio.functional.mvdr_weights_souden.html)；模型/识别器依据沿用G5/G8/G9已保存资料。','', 'UTC：'+now()]
    report=ROOT/'docs/G10_新说话人直接输出与MVDR机制验证报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('completion',report,[OUT/'completion_audit.json',OUT/'evaluation/summary.json',OUT/'evaluation/Whisper/provenance.json',OUT/'evaluation/CTC/provenance.json',Path(__file__)])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'stage':'G10','status':'ok','completed_utc':now(),'report':str(report.relative_to(ROOT))},True);status(stage='complete')


if __name__=='__main__':finish()
