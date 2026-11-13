#!/usr/bin/env python3
import json,runpy
from collections import Counter
import numpy as np
from g16_common import ROOT,OUT,now,sha,dump,read,status,archive
from mcse.asr_scoring import corpus_score,score_words
from g10_finish import cluster_effect


def finish():
    plan=json.loads((OUT/'frozen_plan.json').read_text());reg=json.loads((OUT/'systems.json').read_text());ev=OUT/'evaluation';signals=read(ev/'combined_signals.jsonl');w=read(ev/'Whisper/asr_scores.jsonl');c=read(ev/'CTC/asr_scores.jsonl');cases=read(OUT/'case_manifest.jsonl')
    key=lambda r:(r['system'],r['sample_id']);si={key(r):r for r in signals};wi={key(r):r for r in w};ci={key(r):r for r in c};systems=reg['baseline']+reg['direct']+reg['beam'];expected={(s['system'],r['sample_id']) for s in systems for r in cases}
    assert len(signals)==len(w)==len(c)==len(expected)==3528 and set(si)==set(wi)==set(ci)==expected
    groups={'noisy':['noisy'],'mcnet':['mcnet']}
    for geom,name in [('fixed','fixed'),('rotated','counter'),('rigid','rigid')]:groups[name]=[r['system'] for r in systems if r['kind']=='adapter' and r.get('geometry')==geom];assert len(groups[name])==3
    groups.update({g+'_mvdr':[s+'_mask_mvdr' for s in groups[g]] for g in ('mcnet','fixed','counter','rigid')})
    def aggregate(rows):
        assert rows
        return {'records':len(rows),'sources':len({r['source_id'] for r in rows}),'speakers':len({r['speaker_id'] for r in rows}),
            'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('si_sdr_db','pesq_wb','stoi')},'Whisper':corpus_score(wi[key(r)]['score'] for r in rows),'CTC':corpus_score(ci[key(r)]['score'] for r in rows)}
    noisy=[r for r in signals if r['condition']!='clean'];summary={'coverage':{'systems':21,'conditions_per_system':168,'combined_rows_each_backend':3528,'new_outputs':1008,'reused_baseline_rows':2520,'reused_sources':8,'speakers':4},
        'cells':{},'clean':{},'per_seed':{},'per_field':{},'per_speaker':{},'training_effects':{},'output_effects':{},'reference_ASR':{},'scope':'DEVELOPMENT on already viewed voices,no fresh confirmation'}
    for nid in plan['noise_ids']:
        a=[r for r in noisy if r['noise_id']==nid];summary['cells'][nid]={g:{str(t):aggregate([r for r in a if r['system'] in names and r['target_azimuth']==t]) for t in (-40,20,70)} for g,names in groups.items()}
        for t in (-40,20,70):
            subset=[r for r in a if r['target_azimuth']==t]
            for comparison in ('fixed','counter'):
                for suffix in ('','_mvdr'):summary['training_effects'][f'{nid}|{t}|rigid_minus_{comparison}{suffix}']=cluster_effect(subset,wi,ci,groups['rigid'+suffix],groups[comparison+suffix])
            for g in ('mcnet','fixed','counter','rigid'):summary['output_effects'][f'{nid}|{t}|{g}']=cluster_effect(subset,wi,ci,groups[g+'_mvdr'],groups[g])
    for g,names in groups.items():
        summary['clean'][g]=aggregate([r for r in signals if r['condition']=='clean' and r['system'] in names])
        summary['per_speaker'][g]={sp:aggregate([r for r in noisy if r['system'] in names and r['speaker_id']==sp]) for sp in sorted({r['speaker_id'] for r in noisy})}
        summary['per_field'][g]={str((field,rel)):aggregate([r for r in noisy if r['system'] in names and r['condition']==field and r['relative_angle']==rel]) for field,rel in [('point',-80),('point',25),('multi8',None)]}
    summary['per_seed']={s['system']:{str(t):aggregate([r for r in noisy if r['system']==s['system'] and r['target_azimuth']==t]) for t in (-40,20,70)} for s in systems}
    refs=[r for r in signals if r['system']=='noisy' and r['condition']=='clean' and r['target_azimuth']==20];assert len(refs)==8
    summary['reference_ASR']={'one_actual_clean_reference_per_source':8,'Whisper':corpus_score(wi[key(r)]['reference_score'] for r in refs),'CTC':corpus_score(ci[key(r)]['reference_score'] for r in refs),'notice':'ASR baseline,not a guaranteed lower bound'}
    baseline=read(OUT/'baseline_signals.jsonl');idx={key(r):r for r in baseline};checks={'all3528_unique_pairs_each_backend':set(si)==set(wi)==set(ci)==expected,'all_metrics_ok':all(r['status']=='ok' for r in signals),
        'all2520_baseline_records_unchanged':all(si[k]==r for k,r in idx.items()),'all_new_models_registered_step3000':len(reg['direct'])==3 and all(r['step']==3000 and sha(ROOT/r['path'])==r['checkpoint_sha256'] for r in reg['direct']),
        'both_ASR_scores_recompute':all(r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis']) for r in w+c),
        'frozen_beamforming_unchanged':sha(ROOT/'src/mcse/beamforming.py')==plan['beamforming_sha256']}
    files={}
    for r in signals:
        for p,h in ((r['noisy_path'],r['input_sha256']),(r['reference_path'],r['reference_sha256']),(r['output_path'],r['output_sha256'])):
            if p in files:assert files[p]==h
            files[p]=h
    checks['all_audio_hashes_match']=all(sha(p)==h for p,h in files.items());dump(ev/'readback_file_hashes.json',files,True)
    checks['all_MVDR_mask_sources_match_neural_outputs']=all(r['mask_estimate_path']==si[(r['source_system'],r['sample_id'])]['output_path'] and r['mask_estimate_sha256']==si[(r['source_system'],r['sample_id'])]['output_sha256'] and not r['beamforming_diagnostics']['oracle_clean_reference'] and not r['beamforming_diagnostics']['oracle_DOA'] for r in signals if r['system'].endswith('_mask_mvdr'))
    for earlier,later,prov in [('preparation','signal',ev/'provenance.json'),('signal','beamforming',ev/'beam_provenance.json'),('beamforming','Whisper',ev/'Whisper/provenance.json'),('whisper','CTC',ev/'CTC/provenance.json')]:checks[f'{earlier}_before_{later}']=json.loads((OUT/f'stages/{earlier}/archive_manifest.json').read_text())['saved_utc']<json.loads(prov.read_text())['started_utc']
    bridge=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g16_base');repo=ROOT/'third_party/McNet';bridge['verify_upstream'](repo,repo/bridge['OFFLINE_CHECKPOINT']);checks['original_base_unchanged']=True
    audit={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'coverage':summary['coverage'],'completed_utc':now()};dump(OUT/'completion_audit.json',audit,True);assert audit['status']=='ok';dump(ev/'summary.json',summary,True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col,nid in enumerate(plan['noise_ids']):
        for g,color in [('fixed','#b2573a'),('counter','#4769a8'),('rigid','#24836c')]:
            for suffix,style in [('', '--'),('_mvdr','-')]:
                v=summary['cells'][nid][g+suffix];axes[0,col].plot([-40,20,70],[v[str(t)]['metrics']['si_sdr_db'] for t in (-40,20,70)],style+'o',color=color,label=g+suffix);axes[1,col].plot([-40,20,70],[v[str(t)]['CTC']['wer']*100 for t in (-40,20,70)],style+'o',color=color)
        axes[0,col].set_title(nid)
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xlabel('Target azimuth(deg)')
    axes[0,0].set_ylabel('SI-SDR(dB)');axes[1,0].set_ylabel('CTC WER(%)');axes[0,0].legend(fontsize=6);fig.suptitle('Development:fixed/counter/rigid TRAIN policies;correct TEST geometry;all3seeds')
    fig.savefig(ev/'training_policy_profiles.png',dpi=160);plt.close(fig)
    lines=['# G16 固定、反向与刚性旋转训练的正确几何开发诊断','','G15三种子预定3000步权重注册并验收后运行。本轮是已查看G14语句的开发诊断：8句/4人、两噪声、0dB、三方向/三空间场及干净控制，全部正确测试几何。21系统×168=3528信号及每后端3528评分；其中2520基线来自G14原记录，新增3直接+3MVDR共1008输出。没有再次训练或按结果选种子。',
      '', 'fixed/counter/rigid都是rank4同预算训练；counter实际为旧反向移动增强。G15全2048例保持旧参考麦/监督/采样/随机状态，仅改噪声移动方向。正确刚性训练并非先验优胜者。标准MVDR及两个固定识别器沿用，不是新结构或创新已证实。',
      '', '## 分噪声和方向','','每格SI-SDR dB / Whisper WER / CTC WER；适配器合并全部三种子，WER不截断。PESQ/STOI、每种子/说话人/场型及全部配对效应保存机器汇总。','','|噪声|系统|−40°|20°|70°|','|---|---|---:|---:|---:|']
    for nid,models in summary['cells'].items():
        for g,v in models.items():lines.append(f'|{nid}|{g}|'+ '|'.join(f'{v[str(t)]["metrics"]["si_sdr_db"]:.2f} / {v[str(t)]["Whisper"]["wer"]:.1%} / {v[str(t)]["CTC"]["wer"]:.1%}' for t in (-40,20,70))+'|')
    lines+=['','![开发方向曲线](../artifacts/g16_rigid_dev_v1/evaluation/training_policy_profiles.png)','','## 干净控制','','|系统|SI-SDR|PESQ|STOI|Whisper WER|CTC WER|','|---|---:|---:|---:|---:|---:|']
    for g,r in summary['clean'].items():lines.append(f'|{g}|{r["metrics"]["si_sdr_db"]:.2f}|{r["metrics"]["pesq_wb"]:.3f}|{r["metrics"]["stoi"]:.3f}|{r["Whisper"]["wer"]:.1%}|{r["CTC"]["wer"]:.1%}|')
    lines+=['','## 解释和验收','','开发条件已经多次查看，不能据此宣称最终泛化或挑选训练超参数后仍称未触碰测试。语音质量与两个ASR不必同向；WER超过100%或删除增加不自动证明吞字，当前无新人工逐词/盲听标签。四说话人bootstrap2000区间只是探索性描述，无多重比较显著性声称。',
      '', '保留全部三种子、不利方向和干净控制。若正确训练效果不佳，先报告负结果，不更换注册步数或漏掉某种子。后续应将全部权重/方法冻结，再登记并取得新的完整说话人语句做小规模确认，明确测试几何和旧噪声范围；混响另行压力测试。',
      '', '新模型加载QA、98项测试、旧波形复现及新验证指标复现先归档；输入/旧基线报告→新直接输出→MVDR→Whisper→CTC联合报告依次保存。全覆盖、全部音频/权重SHA、原记录保留和双ASR重计分审计通过。资源仅对新系统实测，旧基线时耗是历史记录，不能混作本轮同时测量时延。',
      '', '[注册QA](G16A_正确训练权重注册与推理兼容验证.md)、[输入与基线](G16B_正确几何开发诊断输入与基线复用.md)、[机器汇总](../artifacts/g16_rigid_dev_v1/evaluation/summary.json)、[审计](../artifacts/g16_rigid_dev_v1/completion_audit.json)。','', 'UTC：'+now()]
    report=ROOT/'docs/G16_三种方向训练策略开发诊断报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');archive('completion',report,[OUT/'completion_audit.json',ev/'summary.json',ev/'Whisper/provenance.json',ev/'CTC/provenance.json',ROOT/'scripts/g16_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','stage':'G16','completed_utc':now(),'report':str(report.relative_to(ROOT))},True);status(stage='complete_pending_acceptance')


if __name__=='__main__':finish()
