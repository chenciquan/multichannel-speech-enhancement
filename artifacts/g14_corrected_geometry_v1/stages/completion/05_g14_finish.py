#!/usr/bin/env python3
import json,runpy
from collections import Counter
import numpy as np
import soundfile as sf
from g14_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive
from mcse.asr_scoring import score_words,corpus_score
from g10_finish import cluster_effect


def protocol_effect(rows,wi,ci,names):
    speakers=sorted({r['speaker_id'] for r in rows});assert len(speakers)==4
    def matrix(protocol):
        result=[]
        for sp in speakers:
            a=[r for r in rows if r['speaker_id']==sp and r['system'] in names and r['geometry_protocol']==protocol];assert a
            w=corpus_score(wi[(r['system'],r['sample_id'])]['score'] for r in a);c=corpus_score(ci[(r['system'],r['sample_id'])]['score'] for r in a)
            result.append([sum(r['metrics']['si_sdr_db'] for r in a),len(a),w['errors'],w['N'],c['errors'],c['N']])
        return np.array(result,dtype=float)
    a,b=matrix('rigid_rotation'),matrix('legacy_counterrotation');assert np.array_equal(a[:,[1,3,5]],b[:,[1,3,5]])
    def ratios(v):return np.stack([v[...,0]/v[...,1],v[...,2]/v[...,3],v[...,4]/v[...,5]],axis=-1)
    delta=ratios(a.sum(0))-ratios(b.sum(0));rng=np.random.default_rng(20261112);ids=rng.integers(4,size=(2000,4));boots=ratios(a[ids].sum(1))-ratios(b[ids].sum(1));lo,hi=np.quantile(boots,[.025,.975],axis=0);keys=('si_sdr_db','Whisper_WER','CTC_WER')
    return {'delta':{k:float(delta[i]) for i,k in enumerate(keys)},'speaker_cluster_bootstrap2000_ci95':{k:[float(lo[i]),float(hi[i])] for i,k in enumerate(keys)},
        'per_speaker':{sp:{k:float(x) for k,x in zip(keys,ratios(a[j])-ratios(b[j]))} for j,sp in enumerate(speakers)},'notice':'rigid minus legacy TEST geometry,same frozen models;four clusters exploratory'}


def finish():
    plan=json.loads((OUT/'frozen_plan.json').read_text());reg=json.loads((OUT/'systems.json').read_text());ev=OUT/'evaluation';signals=read(ev/'combined_signals.jsonl');w=read(ev/'Whisper/asr_scores.jsonl');c=read(ev/'CTC/asr_scores.jsonl');cases=read(BASE/'manifest.jsonl')
    key=lambda r:(r['system'],r['sample_id']);si={key(r):r for r in signals};wi={key(r):r for r in w};ci={key(r):r for r in c};expected={(s['system'],r['sample_id']) for s in reg['direct']+reg['beam'] for r in cases}
    assert len(expected)==5040 and len(signals)==len(w)==len(c)==5040 and set(si)==set(wi)==set(ci)==expected
    groups={'noisy':['noisy'],'mcnet':['mcnet'],'fixed':[r['system'] for r in reg['direct'] if r.get('geometry')=='fixed'],'legacy_rotated':[r['system'] for r in reg['direct'] if r.get('geometry')=='rotated']}
    groups.update({g+'_mvdr':[s+'_mask_mvdr' for s in groups[g]] for g in ('mcnet','fixed','legacy_rotated')})
    def aggregate(rows):
        assert rows
        return {'records':len(rows),'sources':len({r['source_id'] for r in rows}),'speakers':len({r['speaker_id'] for r in rows}),
          'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('si_sdr_db','pesq_wb','stoi')},'Whisper':corpus_score(wi[key(r)]['score'] for r in rows),'CTC':corpus_score(ci[key(r)]['score'] for r in rows)}
    noisy=[r for r in signals if r['condition']!='clean'];summary={'coverage':{'systems':15,'cases_per_system':336,'signal_and_each_ASR_pairs':5040,'reused_utterances':8,'speakers':4},
        'cells':{},'clean':{},'per_seed':{},'per_speaker':{},'per_field':{},'test_geometry_effects':{},'output_mode_effects':{},'legacy_training_policy_effects':{},'reference_ASR':{},'negative_controls':{}}
    for protocol in plan['geometry_protocols']:
        subset=[r for r in noisy if r['geometry_protocol']==protocol];summary['cells'][protocol]={}
        for nid in plan['noise_ids']:
            sub=[r for r in subset if r['noise_id']==nid];summary['cells'][protocol][nid]={g:{str(t):aggregate([r for r in sub if r['system'] in names and r['target_azimuth']==t]) for t in (-40,20,70)} for g,names in groups.items()}
            for t in (-40,20,70):
                a=[r for r in sub if r['target_azimuth']==t]
                for g in ('mcnet','fixed','legacy_rotated'):summary['output_mode_effects'][f'{protocol}|{nid}|{t}|{g}']=cluster_effect(a,wi,ci,groups[g+'_mvdr'],groups[g])
                for suffix in ('','_mvdr'):summary['legacy_training_policy_effects'][f'{protocol}|{nid}|{t}|{suffix or "direct"}']=cluster_effect(a,wi,ci,groups['legacy_rotated'+suffix],groups['fixed'+suffix])
        clean=[r for r in signals if r['condition']=='clean' and r['geometry_protocol']==protocol];summary['clean'][protocol]={g:aggregate([r for r in clean if r['system'] in names]) for g,names in groups.items()}
        summary['per_seed'][protocol]={s['system']:{str(t):aggregate([r for r in subset if r['system']==s['system'] and r['target_azimuth']==t]) for t in (-40,20,70)} for s in reg['direct']+reg['beam']}
        summary['per_speaker'][protocol]={g:{sp:aggregate([r for r in subset if r['system'] in names and r['speaker_id']==sp]) for sp in sorted({r['speaker_id'] for r in subset})} for g,names in groups.items()}
        summary['per_field'][protocol]={g:{str((field,rel)):aggregate([r for r in subset if r['system'] in names and r['condition']==field and r['relative_angle']==rel]) for field,rel in [('point',-80),('point',25),('multi8',None)]} for g,names in groups.items()}
    for nid in plan['noise_ids']:
        for t in (-40,20,70):
            rows=[r for r in noisy if r['noise_id']==nid and r['target_azimuth']==t]
            for g,names in groups.items():summary['test_geometry_effects'][f'{nid}|{t}|{g}']=protocol_effect(rows,wi,ci,names)
    files={}
    for r in signals:
        for p,h in ((r['noisy_path'],r['input_sha256']),(r['reference_path'],r['reference_sha256']),(r['output_path'],r['output_sha256'])):
            if p in files:assert files[p]==h
            files[p]=h
    fileok=all(sha(p)==h for p,h in files.items());dump(ev/'readback_file_hashes.json',files,True)
    pairs={};control_errors=[];hyp_diff=0;exact=0
    for r in signals:pairs.setdefault((r['system'],r['geometry_pair']),{})[r['geometry_protocol']]=r
    assert len(pairs)==2520 and all(len(v)==2 for v in pairs.values())
    for (_,pair),v in pairs.items():
        a=v['legacy_counterrotation'];b=v['rigid_rotation']
        if a['target_azimuth']==20 or a['condition']=='clean':
            x,sr=sf.read(a['noisy_path'],dtype='float32',always_2d=True);xx,rr=sf.read(b['noisy_path'],dtype='float32',always_2d=True);assert np.array_equal(x,xx)
            y,_=sf.read(a['output_path'],dtype='float32');z,_=sf.read(b['output_path'],dtype='float32');error=float(np.max(np.abs(y-z)));control_errors.append(error);exact+=np.array_equal(y,z)
            hyp_diff+=wi[key(a)]['hypothesis']!=wi[key(b)]['hypothesis'] or ci[key(a)]['hypothesis']!=ci[key(b)]['hypothesis']
    assert len(control_errors)==1080 and max(control_errors)<=plan['negative_control_output_max_abs']
    summary['negative_controls']={'pairs':1080,'exact_waveforms':int(exact),'maximum_abs_difference':max(control_errors),'any_backend_hypothesis_difference_pairs':int(hyp_diff),'predefined_tolerance':plan['negative_control_output_max_abs']}
    cleanrefs=[r for r in signals if r['system']=='noisy' and r['condition']=='clean' and r['geometry_protocol']=='rigid_rotation' and r['target_azimuth']==20];assert len(cleanrefs)==8
    summary['reference_ASR']={'unique_utterances':8,'selection':'one actual clean-control PCM24 reference per source,rigid20deg','Whisper':corpus_score(wi[key(r)]['reference_score'] for r in cleanrefs),'CTC':corpus_score(ci[key(r)]['reference_score'] for r in cleanrefs),'notice':'Reference ASR baseline,not guaranteed lower bound'}
    direct={key(r):r for r in signals if r['system'] in {s['system'] for s in reg['direct']}}
    checks={'all5040_signal_and_each_ASR_unique_pairs':set(si)==set(wi)==set(ci)==expected,'all_metrics_ok':all(r['status']=='ok' for r in signals),'all_audio_file_hashes_match':fileok,
      'all1080_negative_controls_match_tolerance':max(control_errors)<=plan['negative_control_output_max_abs'],'all_six_adapters_match':all(r['kind']!='adapter' or sha(ROOT/r['path'])==r['checkpoint_sha256'] for r in reg['direct']),
      'geometry_code_frozen':sha(ROOT/'src/mcse/geometry.py')==plan['geometry_code_sha256'],'legacy_rotation_source_unchanged':sha(ROOT/'scripts/g5_prepare.py')==plan['legacy_rotation_source_sha256'],
      'both_ASR_edit_scores_recompute':all(r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis']) for r in w+c),
      'all_MVDR_masks_from_corresponding_estimate':all(r['mask_estimate_path']==direct[(r['source_system'],r['sample_id'])]['output_path'] and r['mask_estimate_sha256']==direct[(r['source_system'],r['sample_id'])]['output_sha256'] and not r['beamforming_diagnostics']['oracle_clean_reference'] and not r['beamforming_diagnostics']['oracle_DOA'] for r in signals if r['system'].endswith('_mask_mvdr'))}
    for earlier,later,prov in [('preparation','signal',ev/'provenance.json'),('signal','beamforming',ev/'beam_provenance.json'),('beamforming','Whisper',ev/'Whisper/provenance.json'),('whisper','CTC',ev/'CTC/provenance.json')]:checks[f'{earlier}_before_{later}']=json.loads((OUT/f'stages/{earlier}/archive_manifest.json').read_text())['saved_utc']<json.loads(prov.read_text())['started_utc']
    bridge=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g14_integrity');repo=ROOT/'third_party/McNet';bridge['verify_upstream'](repo,repo/bridge['OFFLINE_CHECKPOINT']);checks['original_base_unchanged']=True
    audit={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'coverage':summary['coverage'],'completed_utc':now()};dump(OUT/'completion_audit.json',audit,True);assert audit['status']=='ok';dump(ev/'summary.json',summary,True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,protocol in enumerate(plan['geometry_protocols']):
        for g,color in [('mcnet','#777777'),('fixed','#b2573a'),('legacy_rotated','#4769a8')]:
            for suffix,style in [('', '--'),('_mvdr','-')]:
                v=[aggregate([r for r in noisy if r['geometry_protocol']==protocol and r['target_azimuth']==t and r['system'] in groups[g+suffix]]) for t in (-40,20,70)]
                axes[0,col].plot([-40,20,70],[r['metrics']['si_sdr_db'] for r in v],style+'o',color=color,label=g+suffix);axes[1,col].plot([-40,20,70],[r['CTC']['wer']*100 for r in v],style+'o',color=color)
        axes[0,col].set_title(protocol)
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xlabel('Target azimuth(deg)')
    axes[0,0].set_ylabel('SI-SDR(dB)');axes[1,0].set_ylabel('CTC WER(%)');axes[0,0].legend(fontsize=6);fig.suptitle('TEST geometry corrected;all TRAINED weights unchanged;8 reused utterances')
    fig.savefig(ev/'geometry_profiles.png',dpi=160);plt.close(fig)
    lines=['# G14 正确刚性旋转与旧反向旋转的配对自由场诊断','','## 本次解决什么','','G13确认旧符号错误后，在不改变任何权重的情况下，仅替换测试噪声移动方向。8句完整旧G10语音/4人、两条旧噪声原片段、原八源方向、0dB、三目标角、两点源基准相对角及八源场；两协议共336条件/系统，冻结15系统5040信号及每ASR5040评分。不是新独立最终测试。',
      '', 'rigid_rotation中目标/噪声同向运动，点源实际相对角保持−80/+25；legacy_counterrotation中目标与噪声反向，实际角另列字段。旧rotated适配器仍是旧反向旋转训练，下面称legacy_rotated；本轮没有正确旋转重训，不能据其表现认定正确训练无效或有效。',
      '', '同组跨两协议/三目标共享片段、尺度和参考麦波形；20°与全部干净输入是协议不变负控制。所有新数据报告先归档再推理，保存全部三种子及负结果。无新增依赖、下载或模型。',
      '', '## 分协议、噪声与方向结果','','每格SI-SDR dB / Whisper WER / CTC WER；适配器全部三种子合并，WER不截断。PESQ/STOI、逐种子/说话人/场型及所有效应另存。','','|测试协议|噪声|系统|−40°|20°|70°|','|---|---|---|---:|---:|---:|']
    for protocol,noises in summary['cells'].items():
        for nid,models in noises.items():
            for g,v in models.items():lines.append(f'|{protocol}|{nid}|{g}|'+ '|'.join(f'{v[str(t)]["metrics"]["si_sdr_db"]:.2f} / {v[str(t)]["Whisper"]["wer"]:.1%} / {v[str(t)]["CTC"]["wer"]:.1%}' for t in (-40,20,70))+'|')
    lines+=['','![配对几何方向曲线](../artifacts/g14_corrected_geometry_v1/evaluation/geometry_profiles.png)','','## 同一冻结模型：正确测试几何减旧测试几何','','按4说话人聚类抽样2000次，保留本人所有句子/条件/训练种子，重新累计WER计数；区间只是少量簇探索性描述，不作多重比较显著性声称。SI正值/错误率负值通常较好，但这里比较的是测试难度改变，并非算法改进收益。','','|噪声/方向|系统|ΔSI-SDR|ΔWhisper百分点|ΔCTC百分点|','|---|---|---:|---:|---:|']
    for label,r in summary['test_geometry_effects'].items():
        nid,t,g=label.split('|');d=r['delta'];lines.append(f'|{nid}/{t}|{g}|{d["si_sdr_db"]:+.2f}|{d["Whisper_WER"]*100:+.2f}|{d["CTC_WER"]*100:+.2f}|')
    nc=summary['negative_controls'];ref=summary['reference_ASR']
    lines+=['','## 负控制、参考识别与解释边界','',f'1080对负控制输出最大波形差{nc["maximum_abs_difference"]:.3g}，预定阈值1e-5；逐样本完全相同{nc["exact_waveforms"]}对，任一ASR假设不同{nc["any_backend_hypothesis_difference_pairs"]}对。',
      '', f'每源只取一个实际干净PCM24参考，共8句：Whisper {ref["Whisper"]["wer"]:.1%}、CTC {ref["CTC"]["wer"]:.1%}。这是参考识别基准，不是保证的WER下界。干净组完整指标及协议对照另存，不能用总均值掩盖退步。',
      '', '本轮修复几何定义和结论可信度，不把“换了测试场景成绩变化”当模型创新。若旧legacy_rotated在正确场景仍好，只能说明已有权重在这些新渲染条件下的表现；若变差，也不能据此否定正确整场景旋转训练。未来重训需要正确训练/验证和未查看确认集。',
      '', '两个识别器并不替代逐词/盲听，未证实稳定吞字。仍无混响/实阵列/中文；所有语句、噪声录音已查看，不能反复充当全新最终测试。旧证据保留，并以G13更正限定其解释。',
      '', '全覆盖、音频/权重SHA、双ASR重计分、负控制和阶段顺序审计通过；报告与快照保存后再决定后续训练或混响。',
      '', '[数据协议](G14A_正确几何配对自由场协议与数据.md)、[G13更正](G13A_旋转坐标审计与历史结论更正.md)、[机器汇总](../artifacts/g14_corrected_geometry_v1/evaluation/summary.json)、[审计](../artifacts/g14_corrected_geometry_v1/completion_audit.json)。','', 'UTC：'+now()]
    report=ROOT/'docs/G14_正确几何配对自由场诊断报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');archive('completion',report,[OUT/'completion_audit.json',ev/'summary.json',ev/'Whisper/provenance.json',ev/'CTC/provenance.json',ROOT/'scripts/g14_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','stage':'G14','completed_utc':now(),'report':str(report.relative_to(ROOT))},True);status(stage='complete_pending_readback')


if __name__=='__main__':finish()
