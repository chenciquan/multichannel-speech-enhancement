#!/usr/bin/env python3
"""RT-stratified frozen-method analysis and complete file/score audit."""
import json,runpy
from collections import Counter
import numpy as np
from g18_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive
from mcse.asr_scoring import score_words,corpus_score
from g10_finish import cluster_effect


def finish():
    plan=json.loads((OUT/'frozen_plan.json').read_text());reg=json.loads((OUT/'systems.json').read_text());ev=OUT/'evaluation'
    signals=read(ev/'combined_signals.jsonl');w=read(ev/'Whisper/asr_scores.jsonl');c=read(ev/'CTC/asr_scores.jsonl');cases=read(BASE/'manifest.jsonl')
    key=lambda r:(r['system'],r['sample_id']);si={key(r):r for r in signals};wi={key(r):r for r in w};ci={key(r):r for r in c}
    expected={(s['system'],r['sample_id']) for s in reg['direct']+reg['beam'] for r in cases};assert len(expected)==10584 and len(si)==len(wi)==len(ci)==10584 and set(si)==set(wi)==set(ci)==expected
    groups={'noisy':['noisy'],'mcnet':['mcnet'],'fixed':[r['system'] for r in reg['direct'] if r.get('geometry')=='fixed'],'counter':[r['system'] for r in reg['direct'] if r.get('geometry')=='rotated']}
    groups['rigid']=[r['system'] for r in reg['direct'] if r.get('geometry')=='rigid']
    groups.update({name+'_mvdr':[s+'_mask_mvdr' for s in groups[name]] for name in ('mcnet','fixed','counter','rigid')})
    def aggregate(rows):
        assert rows
        return {'records':len(rows),'sources':len({r['source_id'] for r in rows}),'speakers':len({r['speaker_id'] for r in rows}),
            'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('si_sdr_db','pesq_wb','stoi')},
            'Whisper':corpus_score(wi[key(r)]['score'] for r in rows),'CTC':corpus_score(ci[key(r)]['score'] for r in rows)}
    summary={'coverage':{'systems':21,'cases_per_system':504,'signal_and_each_ASR_rows':10584,'reused_utterances':8,'speakers':4,'new_training':False},
        'cells':{},'clean':{},'per_speaker':{},'per_seed':{},'per_field':{},'effects':{},'reference_ASR':{},'interpretation':'One simulated room,reused voices,requested RT labels not measured T60,denoising to reverberant target.'}
    for rt in plan['requested_RT60_seconds']:
        rk=str(rt);noisy=[r for r in signals if r['condition']!='clean' and r['requested_rt60']==rt];clean=[r for r in signals if r['condition']=='clean' and r['requested_rt60']==rt]
        summary['cells'][rk]={};summary['clean'][rk]={g:aggregate([r for r in clean if r['system'] in names]) for g,names in groups.items()}
        for nid in plan['noise_ids']:
            sub=[r for r in noisy if r['noise_id']==nid]
            summary['cells'][rk][nid]={g:{str(t):aggregate([r for r in sub if r['system'] in names and r['target_azimuth']==t]) for t in (-40,20,70)} for g,names in groups.items()}
            for target in (-40,20,70):
                a=[r for r in sub if r['target_azimuth']==target]
                for g in ('mcnet','fixed','counter','rigid'):summary['effects'][f'{rt}|{nid}|{target}|{g}_mvdr_minus_direct']=cluster_effect(a,wi,ci,groups[g+'_mvdr'],groups[g])
                for suffix in ('','_mvdr'):summary['effects'][f'{rt}|{nid}|{target}|counter_minus_fixed{suffix}']=cluster_effect(a,wi,ci,groups['counter'+suffix],groups['fixed'+suffix])
            for target in (-40,20,70):
                a=[r for r in sub if r['target_azimuth']==target]
                for comparison in ('counter','fixed'):
                    for suffix in ('','_mvdr'):
                        summary['effects'][f'{rt}|{nid}|{target}|rigid_minus_{comparison}{suffix}']=cluster_effect(a,wi,ci,groups['rigid'+suffix],groups[comparison+suffix])
        summary['per_speaker'][rk]={g:{sp:aggregate([r for r in noisy if r['system'] in names and r['speaker_id']==sp]) for sp in sorted({r['speaker_id'] for r in noisy})} for g,names in groups.items()}
        summary['per_seed'][rk]={s['system']:{str(t):aggregate([r for r in noisy if r['system']==s['system'] and r['target_azimuth']==t]) for t in (-40,20,70)} for s in reg['direct']+reg['beam']}
        summary['per_field'][rk]={g:{field:aggregate([r for r in noisy if r['system'] in names and r['condition']==field[0] and r['relative_angle']==field[1]]) for field in [('point',-80),('point',25),('multi8',None)]} for g,names in groups.items()}
        # JSON object keys are strings, while keeping the two point-source separations distinct.
        summary['per_field'][rk]={g:{str(k):v for k,v in m.items()} for g,m in summary['per_field'][rk].items()}
        refs=[r for r in clean if r['system']=='noisy'];assert len(refs)==24
        summary['reference_ASR'][rk]={'representations':24,'unique_utterances':8,'Whisper':corpus_score(wi[key(r)]['reference_score'] for r in refs),'CTC':corpus_score(ci[key(r)]['reference_score'] for r in refs),
             'notice':'Actual clean-control PCM24 references,one per source and angle; not independent24sentences or guaranteed WER lower bound.'}
    files={}
    for r in signals:
        for p,h in ((r['noisy_path'],r['input_sha256']),(r['reference_path'],r['reference_sha256']),(r['output_path'],r['output_sha256'])):
            if p in files:assert files[p]==h
            files[p]=h
    fileok=all(sha(p)==h for p,h in files.items());dump(ev/'readback_file_hashes.json',files,True)
    direct={key(r):r for r in signals if r['system'] in {s['system'] for s in reg['direct']}}
    checks={'all10584_unique_pairs_each_backend':set(si)==set(wi)==set(ci)==expected,'all_metrics_ok':all(r['status']=='ok' for r in signals),
       'all_input_reference_output_hashes_match':fileok,'full_conditions':len(cases)==504 and sum(r['condition']=='clean' for r in cases)==72,
       'eight_reused_sources_four_speakers':len({r['source_id'] for r in cases})==8 and len({r['speaker_id'] for r in cases})==4,
       'whole_convolution_lengths':all(r['frames']==r['dry_frames']+7679 and r['frames']<64000 for r in cases),
       'source_files_match':all(sha(ROOT/'data/g17_v1/sources'/r['path'])==r['sha256'] for r in json.loads((OUT/'selected_sources.json').read_text())),
       'all99_RIR_files_match':all(sha(r['path'])==r['sha256'] for r in json.loads((OUT/'RIR_manifest.json').read_text())),
       'all_nine_adapters_unchanged':all(r['kind']!='adapter' or sha(ROOT/r['path'])==r['checkpoint_sha256'] for r in reg['direct']),
       'beam_code_unchanged':sha(ROOT/'src/mcse/beamforming.py')==plan['beamforming_sha256'],
       'both_ASR_scores_recompute':all(r['score']==score_words(r['transcript'],r['hypothesis']) and r['reference_score']==score_words(r['transcript'],r['reference_hypothesis']) for r in w+c),
       'all_MVDR_masks_use_corresponding_neural_estimate':all(r['mask_estimate_path']==direct[(r['source_system'],r['sample_id'])]['output_path'] and r['mask_estimate_sha256']==direct[(r['source_system'],r['sample_id'])]['output_sha256'] and not r['beamforming_diagnostics']['oracle_clean_reference'] and not r['beamforming_diagnostics']['oracle_DOA'] for r in signals if r['system'].endswith('_mask_mvdr'))}
    for earlier,later,prov in [('preparation','signal',ev/'provenance.json'),('signal','beamforming',ev/'beam_provenance.json'),('beamforming','Whisper',ev/'Whisper/provenance.json'),('whisper','CTC',ev/'CTC/provenance.json')]:
        checks[f'{earlier}_archived_before_{later}']=json.loads((OUT/f'stages/{earlier}/archive_manifest.json').read_text())['saved_utc']<json.loads(prov.read_text())['started_utc']
    bridge=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g18_integrity');repo=ROOT/'third_party/McNet';bridge['verify_upstream'](repo,repo/bridge['OFFLINE_CHECKPOINT']);checks['original_base_unchanged']=True
    audit={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'coverage':summary['coverage'],'completed_utc':now()};dump(OUT/'completion_audit.json',audit,True)
    if audit['status']!='ok':raise RuntimeError('G18 audit failed; evidence retained')
    dump(ev/'summary.json',summary,True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for col,rt in enumerate(plan['requested_RT60_seconds']):
        for g,color in [('fixed','#b2573a'),('counter','#4769a8'),('rigid','#24836c')]:
            for suffix,style in [('', '--'),('_mvdr','-')]:
                v=[aggregate([r for r in signals if r['condition']!='clean' and r['requested_rt60']==rt and r['target_azimuth']==t and r['system'] in groups[g+suffix]]) for t in (-40,20,70)]
                axes[0,col].plot([-40,20,70],[r['metrics']['si_sdr_db'] for r in v],style+'o',color=color,label=g+suffix)
                axes[1,col].plot([-40,20,70],[r['CTC']['wer']*100 for r in v],style+'o',color=color)
        axes[0,col].set_title(f'Requested RT60={rt}s')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xlabel('Target azimuth(deg)')
    axes[0,0].set_ylabel('SI-SDR(dB)');axes[1,0].set_ylabel('CTC WER(%)');axes[0,0].legend(fontsize=6)
    fig.suptitle('Single simulated room;8 reused utterances,4 speakers;reverberant targets');fig.savefig(ev/'room_profiles.png',dpi=160);plt.close(fig)
    lines=['# G18 固定房间反射下的训练策略与输出形式压力测试','','## 范围与目标','','复用G17的8句完整语音/4位说话人，旧OOFFICE/NPARK，0dB，三方向、三空间场和请求RT60=0/0.15/0.35秒。单间6×5×3m模拟，不是多房间/实测六通道。504条件×21系统=10584信号及每识别器10584评分，无新训练或按新成绩选种子。所有目标是参考麦混响语音，评估降噪，不宣称去混响。',
        '', '99个房间响应采用成熟MIT生成器。0.48秒有限响应窗、完整语句卷积尾部保留；连续噪声含1秒预卷积。请求Sabine参数与实际衰减须区分，逐响应T20外推/尾端能量另存。无反射条件也是有限距离球面传播，固定全局八源星座也区别于G17逐组随机场。源星座在固定房间内同向移动，墙不转，反射场不作刚性旋转；不能把与G17的全部差值全称混响影响。',
        '', '## 分混响、噪声和方向结果','','每格SI-SDR dB / Whisper WER / CTC WER；fixed/counter/rigid均为全部三种子。PESQ/STOI及全部种子/说话人/场型/效应区间另存机器汇总。WER可超过100%，不解释成人声丢失率。','','|请求RT|噪声|系统|−40°|20°|70°|','|---:|---|---|---:|---:|---:|']
    for rt,noises in summary['cells'].items():
        for nid,models in noises.items():
            for g,v in models.items():lines.append(f'|{rt}|{nid}|{g}|'+ '|'.join(f'{v[str(t)]["metrics"]["si_sdr_db"]:.2f} / {v[str(t)]["Whisper"]["wer"]:.1%} / {v[str(t)]["CTC"]["wer"]:.1%}' for t in (-40,20,70))+'|')
    lines+=['','![房间方向曲线](../artifacts/g18_room_stress_v1/evaluation/room_profiles.png)','','## 干净输入和参考识别基准','','|请求RT|系统|SI-SDR|PESQ|STOI|Whisper WER|CTC WER|','|---:|---|---:|---:|---:|---:|---:|']
    for rt,models in summary['clean'].items():
        for g,r in models.items():lines.append(f'|{rt}|{g}|{r["metrics"]["si_sdr_db"]:.2f}|{r["metrics"]["pesq_wb"]:.3f}|{r["metrics"]["stoi"]:.3f}|{r["Whisper"]["wer"]:.1%}|{r["CTC"]["wer"]:.1%}|')
    lines+=['','每RT参考基准取干净控制中8句×3方向的24种实际PCM24表示；跨方向并非新增语句。不同房间响应本身可能改变ASR难度，不把输出WER变化全部归因增强模型。基准如下：']
    for rt,r in summary['reference_ASR'].items():lines.append(f'- 请求RT={rt}：Whisper {r["Whisper"]["wer"]:.1%}，CTC {r["CTC"]["wer"]:.1%}。这是参考识别基准，非保证的WER下界。')
    lines+=['','## 解释边界与归档','','适配器方向分布、输出形式和混响可能共同改变目标失真/残余噪声。即使两后端一致，也不能直接证明某个辅音或词尾被保留；当前没有新增逐词/盲听标签。保留所有不利方向与干净控制，不能因均值好看宣称全条件优势。',
        '', '配对效应按4位说话人聚类抽样2000次，保留各人全部条件及训练种子，只是少量簇的探索性区间，无多重比较显著性声称。已查看的G17语句不能再次充当全新最终确认集。没有新噪声录音、多房间、真实阵列、中文或实时验证，不能保证论文发表。',
        '', '物理/生成器与容量→数据→直接信号→MVDR→Whisper→CTC依次存档。全覆盖、全部音频/源/RIR/权重SHA、双ASR重计分和阶段顺序审计通过；原模型和驱动/CUDA配置未改，无新依赖或下载。反向训练与正确刚性训练均作为固定策略保留，不预设谁优胜。后续必须先读回结果和资源，再决定新的受控问题。',
        '', '[数据协议](G18B_单房间反射压力数据与几何协议.md)、[机器汇总](../artifacts/g18_room_stress_v1/evaluation/summary.json)、[审计](../artifacts/g18_room_stress_v1/completion_audit.json)、[RIR记录](../artifacts/g18_room_stress_v1/RIR_manifest.json)。','', 'UTC：'+now()]
    report=ROOT/'docs/G18_固定房间反射压力测试报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('completion',report,[OUT/'completion_audit.json',ev/'summary.json',ev/'Whisper/provenance.json',ev/'CTC/provenance.json',ROOT/'scripts/g18_finish.py'])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','stage':'G18','completed_utc':now(),'report':str(report.relative_to(ROOT))},True);status(stage='complete_pending_readback')


if __name__=='__main__':finish()
