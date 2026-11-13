#!/usr/bin/env python3
import json,runpy
from collections import Counter
import numpy as np
from g17_common import ROOT,OUT,BASE,now,sha,dump,read,status,archive
from mcse.asr_scoring import corpus_score,score_words
from g10_finish import cluster_effect


def finish():
    plan=json.loads((OUT/'frozen_plan.json').read_text());reg=json.loads((OUT/'systems.json').read_text());ev=OUT/'evaluation';signals=read(ev/'combined_signals.jsonl');w=read(ev/'Whisper/asr_scores.jsonl');c=read(ev/'CTC/asr_scores.jsonl');cases=read(BASE/'manifest.jsonl')
    key=lambda r:(r['system'],r['sample_id']);si={key(r):r for r in signals};wi={key(r):r for r in w};ci={key(r):r for r in c};systems=reg['baseline']+reg['direct']+reg['beam'];expected={(s['system'],r['sample_id']) for s in systems for r in cases}
    assert len(signals)==len(w)==len(c)==len(expected)==3528 and set(si)==set(wi)==set(ci)==expected
    groups={'noisy':['noisy'],'mcnet':['mcnet']}
    for geom,name in [('fixed','fixed'),('rotated','counter'),('rigid','rigid')]:groups[name]=[r['system'] for r in systems if r['kind']=='adapter' and r.get('geometry')==geom];assert len(groups[name])==3
    groups.update({g+'_mvdr':[s+'_mask_mvdr' for s in groups[g]] for g in ('mcnet','fixed','counter','rigid')})
    def aggregate(rows):
        assert rows
        return {'records':len(rows),'sources':len({r['source_id'] for r in rows}),'speakers':len({r['speaker_id'] for r in rows}),
            'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('si_sdr_db','pesq_wb','stoi')},'Whisper':corpus_score(wi[key(r)]['score'] for r in rows),'CTC':corpus_score(ci[key(r)]['score'] for r in rows)}
    noisy=[r for r in signals if r['condition']!='clean'];summary={'coverage':{'systems':21,'conditions_per_system':168,'combined_rows_each_backend':3528,'new_outputs':3528,'reused_baseline_rows':0,'fresh_sources':8,'speakers':4},
        'cells':{},'clean':{},'per_seed':{},'per_field':{},'per_speaker':{},'training_effects':{},'output_effects':{},'reference_ASR':{},'scope':'Fresh4project speakers8completeutterances,all21methods frozen before fetch;not broadnoise/realroomconfirmation'}
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
    checks={'all3528_unique_pairs_each_backend':set(si)==set(wi)==set(ci)==expected,'all_metrics_ok':all(r['status']=='ok' for r in signals),
        'all9_adapters_step3000_and_hashes_match':len([r for r in reg['direct'] if r['kind']=='adapter'])==9 and all(r['kind']!='adapter' or (r['step']==3000 and sha(ROOT/r['path'])==r['checkpoint_sha256']) for r in reg['direct']),
        'both_ASR_scores_recompute':all(r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis']) for r in w+c),
        'frozen_beamforming_unchanged':sha(ROOT/'src/mcse/beamforming.py')==plan['beamforming_sha256']}
    source=read(BASE/'sources/speech_manifest.jsonl');excluded=json.loads((BASE/'sources/status.json').read_text())['excluded']
    checks['fresh8sources4speakers_excluded_from_project']=len(source)==8 and len({r['speaker_id'] for r in source})==4 and all(n==2 for n in Counter(r['speaker_id'] for r in source).values()) and all(r[k] not in excluded[k] for r in source for k in ('source_id','speaker_id','sha256'))
    checks['all_fresh_source_files_match']=all(sha(BASE/'sources'/r['path'])==r['sha256'] for r in source)
    checks['sources_archived_before_simulation']=json.loads((OUT/'stages/sources/archive_manifest.json').read_text())['saved_utc']<json.loads((OUT/'stages/preparation/archive_manifest.json').read_text())['saved_utc']
    checks['all_source_exclusion_manifests_unchanged']=all(sha(ROOT/p)==h for p,h in json.loads((BASE/'sources/status.json').read_text())['prior_manifests_sha256'].items())
    checks['methods_frozen_before_new_source_fetch']=plan['frozen_utc']<json.loads((BASE/'sources/status.json').read_text())['started_utc']
    files={}
    for r in signals:
        for p,h in ((r['noisy_path'],r['input_sha256']),(r['reference_path'],r['reference_sha256']),(r['output_path'],r['output_sha256'])):
            if p in files:assert files[p]==h
            files[p]=h
    checks['all_audio_hashes_match']=all(sha(p)==h for p,h in files.items());dump(ev/'readback_file_hashes.json',files,True)
    checks['all_MVDR_mask_sources_match_neural_outputs']=all(r['mask_estimate_path']==si[(r['source_system'],r['sample_id'])]['output_path'] and r['mask_estimate_sha256']==si[(r['source_system'],r['sample_id'])]['output_sha256'] and not r['beamforming_diagnostics']['oracle_clean_reference'] and not r['beamforming_diagnostics']['oracle_DOA'] for r in signals if r['system'].endswith('_mask_mvdr'))
    for earlier,later,prov in [('preparation','signal',ev/'provenance.json'),('signal','beamforming',ev/'beam_provenance.json'),('beamforming','Whisper',ev/'Whisper/provenance.json'),('whisper','CTC',ev/'CTC/provenance.json')]:checks[f'{earlier}_before_{later}']=json.loads((OUT/f'stages/{earlier}/archive_manifest.json').read_text())['saved_utc']<json.loads(prov.read_text())['started_utc']
    bridge=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g17_base');repo=ROOT/'third_party/McNet';bridge['verify_upstream'](repo,repo/bridge['OFFLINE_CHECKPOINT']);checks['original_base_unchanged']=True
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
    axes[0,0].set_ylabel('SI-SDR(dB)');axes[1,0].set_ylabel('CTC WER(%)');axes[0,0].legend(fontsize=6);fig.suptitle('Fresh8utterances4speakers;fixed/counter/rigid TRAIN;correctTESTgeometry;all3seeds')
    fig.savefig(ev/'training_policy_profiles.png',dpi=160);plt.close(fig)
    lines=['# G17 三种方向训练策略的新说话人确认','','G16开发负结果和全部方法冻结后，按官方归档顺序取得4位本项目从未使用的说话人、每人2句完整语音。排除全部G1/G2/G3/G5/G10话人/语句ID/源SHA，源记录与数据协议先于评分归档。两条旧噪声、0dB、三方向/三空间场及干净控制，正确测试几何；21系统×168=3528条全部本轮生成的信号及每后端3528评分。没有新训练或按新结果筛种子/角度。',
      '', 'fixed/counter/rigid都是rank4同预算训练；counter实际为旧反向移动增强。G15全2048例保持旧参考麦/监督/采样/随机状态，仅改噪声移动方向。正确刚性训练并非先验优胜者。标准MVDR及两个固定识别器沿用，不是新结构或创新已证实。',
      '', '## 分噪声和方向','','每格SI-SDR dB / Whisper WER / CTC WER；适配器合并全部三种子，WER不截断。PESQ/STOI、每种子/说话人/场型及全部配对效应保存机器汇总。','','|噪声|系统|−40°|20°|70°|','|---|---|---:|---:|---:|']
    for nid,models in summary['cells'].items():
        for g,v in models.items():lines.append(f'|{nid}|{g}|'+ '|'.join(f'{v[str(t)]["metrics"]["si_sdr_db"]:.2f} / {v[str(t)]["Whisper"]["wer"]:.1%} / {v[str(t)]["CTC"]["wer"]:.1%}' for t in (-40,20,70))+'|')
    lines+=['','![新语音方向曲线](../artifacts/g17_fresh_confirmation_v1/evaluation/training_policy_profiles.png)','','## 干净控制','','|系统|SI-SDR|PESQ|STOI|Whisper WER|CTC WER|','|---|---:|---:|---:|---:|---:|']
    for g,r in summary['clean'].items():lines.append(f'|{g}|{r["metrics"]["si_sdr_db"]:.2f}|{r["metrics"]["pesq_wb"]:.3f}|{r["metrics"]["stoi"]:.3f}|{r["Whisper"]["wer"]:.1%}|{r["CTC"]["wer"]:.1%}|')
    lines+=['','## 解释和验收','','源句是取数前完成排除的项目新说话人，方法/步数/输出形式先冻结；查看本轮后它也成为已查看数据，后续调参不能继续称未经触碰最终测试。旧噪声不是新录音，仅四人英语短朗读、自由场名义阵列，不能外推真实混响、中文或广泛人群。基础识别器预训练对公开语料的曝光未独立审计。语音质量与两个ASR不必同向；WER超过100%或删除增加不自动证明吞字，当前无新人工逐词/盲听标签。四说话人bootstrap2000区间只是探索性描述，无多重比较显著性声称。',
      '', '保留全部三种子、不利方向和干净控制。若正确训练效果不佳，先报告负结果，不更换注册步数或漏掉某种子。即使某项指标优势成立，也只在本次范围；下一步混响另行受控压力测试。不能因一轮成绩改注册步数再复用该确认集作为全新测试。',
      '', '沿用G16已验收注册QA及98项测试，共享数值代码不改；本轮源/排除报告→数据→直接输出→MVDR→Whisper→CTC联合报告依次保存。全覆盖、新源排除、全部音频/权重SHA、掩码来源和双ASR重计分审计通过。全部本轮系统按相同FP32/batch1顺序运行并逐模型释放缓存，计时仍不是流式或端到端部署声称。',
      '', '[注册QA](G16A_正确训练权重注册与推理兼容验证.md)、[新语音来源](G17A_新确认语音选择与排除记录.md)、[数据协议](G17B_新语音正确几何模拟协议与数据.md)、[机器汇总](../artifacts/g17_fresh_confirmation_v1/evaluation/summary.json)、[审计](../artifacts/g17_fresh_confirmation_v1/completion_audit.json)。','', 'UTC：'+now()]
    report=ROOT/'docs/G17_三种方向训练策略新说话人确认报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');archive('completion',report,[OUT/'completion_audit.json',ev/'summary.json',ev/'Whisper/provenance.json',ev/'CTC/provenance.json',ROOT/'scripts/g17_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','stage':'G17','completed_utc':now(),'report':str(report.relative_to(ROOT))},True);status(stage='complete_pending_acceptance')


if __name__=='__main__':finish()
