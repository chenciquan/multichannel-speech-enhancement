#!/usr/bin/env python3
"""Independent fresh-speaker comparison, audit and Chinese report for G5."""
import hashlib,html,json,runpy
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
import g4_run as rt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g5_rotation_v1';EVAL=OUT/'evaluation';BASE=ROOT/'data/g5_v1'
def compare(signals,asr,a,b,off):
    idx={(r['system'],r['sample_id']):r for r in signals};ai={(r['system'],r['sample_id']):r for r in asr};groups=defaultdict(list)
    for r in signals:
        if r['system']!=a[0] or r['condition']=='clean' or (off and r['target_azimuth']==20):continue
        sid=r['sample_id'];metric={k:[] for k in ('si_sdr_db','pesq_wb','stoi')};ed=[];n=[]
        for x,y in zip(a,b):
            for k in metric:metric[k].append(idx[(x,sid)]['metrics'][k]-idx[(y,sid)]['metrics'][k])
            ax,by=ai[(x,sid)]['score'],ai[(y,sid)]['score'];assert ax['N']==by['N'];ed.append(ax['errors']-by['errors']);n.append(ax['N'])
        groups[r['source_id']].append([*[float(np.mean(v)) for v in metric.values()],float(np.mean(ed)),float(np.mean(n))])
    data=np.array([[*np.mean(np.array(g)[:,:3],axis=0),*np.sum(np.array(g)[:,3:],axis=0)] for g in groups.values()])
    def calc(d):return np.r_[d[:,:3].mean(axis=0),d[:,3].sum()/d[:,4].sum()]
    point=calc(data);rng=np.random.default_rng(2026111508);boot=np.array([calc(data[rng.integers(0,len(data),len(data))]) for _ in range(2000)]);lo,hi=np.quantile(boot,[.025,.975],axis=0)
    return {'a':a,'b':b,'off_target_only':off,'sources':len(data),'conditional_source_bootstrap_2000':{k:{'delta':float(point[i]),'ci95':[float(lo[i]),float(hi[i])]} for i,k in enumerate(('si_sdr_db','pesq_wb','stoi','wer'))},
            'caution':'12 source clusters, only3 speakers; descriptive/exploratory intervals, no broad significance or novelty claim'}
def review(cases,signals,asr,selected):
    first={}
    for r in sorted(cases,key=lambda r:r['source_id']):first.setdefault(r['speaker_id'],r['source_id'])
    chosen=sorted([r for r in cases if r['source_id'] in first.values() and r['noise_id']=='OOFFICE' and r['condition']=='point' and r['relative_angle']==-80],key=lambda r:(r['source_id'],r['target_azimuth']))
    assert len(chosen)==9;seed=selected['completed_seeds'][0];names=['noisy','mcnet']+[s['system'] for s in selected['systems'] if s['seed']==seed]
    idx={(r['system'],r['sample_id']):r for r in signals};ai={(r['system'],r['sample_id']):r for r in asr};audio={}
    for r in chosen:
        sid=r['sample_id'];audio[(sid,'reference')]=sf.read(r['reference_path'],dtype='float32')[0]
        for name in names:audio[(sid,name)]=sf.read(idx[(name,sid)]['output_path'],dtype='float32')[0]
    peak=max(float(np.abs(a).max()) for a in audio.values());gain=np.float32(.95/peak if peak>1 else 1.);folder=OUT/'review';folder.mkdir();cards=[]
    for r in chosen:
        sid=r['sample_id'];players=[]
        for name in ['reference']+names:
            p=folder/'audio'/name/(sid+'.wav');p.parent.mkdir(parents=True,exist_ok=True);sf.write(p,audio[(sid,name)]*gain,16000,subtype='FLOAT')
            text='' if name=='reference' else f'；ASR错误 {ai[(name,sid)]["score"]["errors"]}；'+html.escape(ai[(name,sid)]['hypothesis'])
            players.append(f'<p>{html.escape(name)}{text}<br><audio controls preload="none" src="audio/{name}/{sid}.wav"></audio></p>')
        cards.append(f'<section><h2>{r["source_id"]}，目标{r["target_azimuth"]}°</h2><p>{html.escape(r["transcript"])}</p>'+''.join(players)+'</section>')
    page='<!doctype html><meta charset="utf-8"><title>G5 新说话人对照</title><style>body{max-width:1000px;margin:30px auto;font:16px system-ui;line-height:1.6}section{border:1px solid #bbb;padding:20px;margin:20px 0}audio{width:500px;max-width:100%}</style><h1>G5 新说话人的配对方向对照</h1><p>每位新说话人按ID选第一句，三种目标方向；相对噪声方位−80°，OOFFICE，0dB；固定使用首个种子，不选择最好种子。不是盲听，也没有人工判定。全页共用试听衰减 '+f'{float(gain):.6f}'+'，不用于计分。WER不是人声丢失比例。</p>'+''.join(cards)
    (folder/'index.html').write_text(page,encoding='utf-8');rt.dump(folder/'selection.json',{'sample_ids':[r['sample_id'] for r in chosen],'rule':'first source per fresh speaker, all directions, first seed','common_gain':float(gain),'systems':names})

def finish_report(cases,systems,selected):
    rt.OUT=OUT;rt.EVAL=EVAL;signals=rt.read(EVAL/'signals.jsonl');asr=rt.read(EVAL/'asr_scores.jsonl');ai={(r['system'],r['sample_id']):r for r in asr};seeds=selected['completed_seeds']
    cells={f'{g}_{v}':[s['system'] for s in selected['systems'] if s['geometry']==g and s['variant']==v] for g in ('fixed','rotated') for v in ('without_clean_identity','complete')}
    def rows(names,condition=None,target=None):return [r for r in signals if r['system'] in names and (condition is None or r['condition']==condition) and (target is None or r['target_azimuth']==target)]
    summary={'coverage':{'systems':len(systems),'signal_rows':len(signals),'asr_rows':len(asr),'fresh_sources':12,'fresh_speakers':3,'sources_per_speaker':4,'completed_seeds':seeds},'cells':{},'comparisons':{}}
    for cell,names in cells.items():
        summary['cells'][cell]={'noisy':rt.aggregate([r for r in rows(names) if r['condition']!='clean'],ai),'clean':rt.aggregate(rows(names,'clean'),ai),
          'by_target':{str(t):rt.aggregate([r for r in rows(names,target=t) if r['condition']!='clean'],ai) for t in (-40,20,70)},
          'by_field':{f:rt.aggregate(rows(names,f),ai) for f in ('point','multi8')},
          'clean_by_target':{str(t):rt.aggregate(rows(names,'clean',t),ai) for t in (-40,20,70)}}
    for v in ('without_clean_identity','complete'):
        for off in (False,True):summary['comparisons'][v+('_off_target' if off else '_all')]=compare(signals,asr,cells['rotated_'+v],cells['fixed_'+v],off)
    rt.dump(EVAL/'summary.json',summary)
    old=[]
    for p in ('data/sources/g1_v1/source_manifest.jsonl','data/sources/g2_confirmation_v1/source_manifest.jsonl','data/g3_v1/sources/speech_manifest.jsonl'):old+=rt.read(ROOT/p)
    fresh=rt.read(BASE/'sources/speech_manifest.jsonl');speakers=Counter(r['speaker_id'] for r in fresh);rawgroups=defaultdict(set);inputok=True
    for r in cases:
        inputok &= rt.sha(r['noisy_path'])==r['input_sha256'] and rt.sha(r['reference_path'])==r['reference_sha256']
        x,_=sf.read(r['noisy_path'],dtype='float32',always_2d=True);s,_=sf.read(r['reference_path'],dtype='float32');key=(r['source_id'],r['noise_id'],r['condition'],r['relative_angle'])
        rawgroups[key].add((hashlib.sha256(x[:,4].tobytes()).hexdigest(),hashlib.sha256(s.tobytes()).hexdigest()))
    expected={(s['system'],r['sample_id']) for s in systems for r in cases};sigkeys={(r['system'],r['sample_id']) for r in signals};prov=json.loads((EVAL/'provenance.json').read_text());ap=json.loads((EVAL/'asr_provenance.json').read_text())
    prep=json.loads((OUT/'stages/preparation/archive_manifest.json').read_text());tr=json.loads((OUT/'stages/training/archive_manifest.json').read_text());sigstage=json.loads((OUT/'stages/signal/archive_manifest.json').read_text())
    results=[]
    for sy in selected['systems']:results.append(json.loads((ROOT/sy['path']).parent.joinpath('result.json').read_text()))
    checks={'fresh_speakers_disjoint_from_entire_project':not set(speakers)&{r['speaker_id'] for r in old},'three_balanced_fresh_speakers':len(speakers)==3 and set(speakers.values())=={4},
      'fresh_source_hashes_disjoint':not {r['sha256'] for r in fresh}&{r['sha256'] for r in old},
      'all_cases_and_asr_present':sigkeys==expected=={(r['system'],r['sample_id']) for r in asr} and len(signals)==len(asr)==len(expected),
      'all_metric_pairs_ok':all(r['status']=='ok' for r in signals),'inputs_unchanged':bool(inputok),'counterfactual_reference_samples_equal':all(len(v)==1 for v in rawgroups.values()),
      'outputs_unchanged':all(rt.sha(r['output_path'])==r['output_sha256'] for r in signals),
      'all_complete_cells_3000_steps':all(r['completed_steps']==3000 for r in results),'all_original_bases_frozen':all(r['base_unchanged'] and r['base_digest_before']==r['base_digest_after'] for r in results),
      'paired_training_references_equal':json.loads((OUT/'PAIRED_TRAIN_COMPLETE.json').read_text())['paired_reference_mic_and_target_equal'],
      'saved_paired_arrays_verified':json.loads((OUT/'paired_data_audit.json').read_text())['status']=='ok',
      'preparation_archived_before_training':all(datetime.fromisoformat(prep['saved_at_utc'])<datetime.fromisoformat(json.loads((ROOT/sy['path']).parent.joinpath('config.json').read_text())['started_at']) for sy in selected['systems']),
      'training_archived_before_confirmation':datetime.fromisoformat(tr['saved_at_utc'])<datetime.fromisoformat(prov['started_utc']),
      'signal_archived_before_asr':datetime.fromisoformat(sigstage['saved_at_utc'])<datetime.fromisoformat(ap['started_utc']),
      'confirmation_selection_sha_unchanged':rt.sha(BASE/'confirmation/manifest.jsonl')==selected['confirmation_manifest_sha256']}
    runtime=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g5_baseline_audit');repo=ROOT/'third_party/McNet';runtime['verify_upstream'](repo,repo/runtime['OFFLINE_CHECKPOINT']);checks['official_base_unchanged']=True
    review(cases,signals,asr,selected)
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            if tag=='audio':self.paths.append(dict(attrs)['src'])
    parser=Links();parser.feed((OUT/'review/index.html').read_text());checks['review_63_mono_links']=len(parser.paths)==63 and all(sf.info(OUT/'review'/p).channels==1 for p in parser.paths)
    audit={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'coverage':summary['coverage']};rt.dump(OUT/'completion_audit.json',audit)
    if audit['status']!='ok':raise RuntimeError('G5 completion audit failed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure=EVAL/'figures';figure.mkdir();fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for v,color in [('without_clean_identity','#4769a8'),('complete','#b54c35')]:
        for g,ls in [('fixed','--'),('rotated','-')]:
            d=summary['cells'][g+'_'+v]['by_target'];x=[-40,20,70]
            axes[0].plot(x,[d[str(t)]['metrics']['si_sdr_db'] for t in x],ls+'o',color=color,label=g+' / '+v)
            axes[1].plot(x,[d[str(t)]['asr']['wer']*100 for t in x],ls+'o',color=color)
    axes[0].set_ylabel('SI-SDR (dB)');axes[1].set_ylabel('WER (%)')
    for ax in axes:ax.set_xlabel('Target azimuth (deg)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Fresh balanced speakers; seed-averaged cells; SNR0');fig.savefig(figure/'fresh_direction_transfer.png',dpi=170);plt.close(fig)
    lines=['# G5 整场景旋转与新说话人确认报告','','## 完成范围与研究问题','',f'在G4报告与审计归档后开展。完成{len(seeds)}个完整四单元种子组、{len(selected["systems"])}次训练、{len(selected["systems"])*3000}次有效更新。每组固定/旋转×干净约束有/无，均为rank4、3,584参数、FP32、3,000步。额外种子只按预登记时间余量决定，不依据结果。',
      '', '整场景旋转是成熟的数据增强强对照，不是本项目发明的新模块。目标与所有噪声方向共同旋转，人声/噪声相对关系保持；配对样本共享裁片、噪声片段、SNR和共同幅度尺度，参考第5麦输入与监督目标完全相同。模型没有接收目标方向标签，只接收六通道音频。',
      '', '验证使用原验证集两位说话人各3句，共54个多方向条件，按相同SI-SDR/STOI门限规则选检查点，没有ASR反传或WER调参。确认集是此前未出现的3位说话人，各4句，完全排除旧说话人/源文件哈希；目标−40/20/70°，OOFFICE/PCAFETER，0dB。点声源两种相对角度与八源叠加，共216个带噪条件和36个干净输入。点声源占测试条件2/3，八源占1/3，分组结果完整保存。',
      '',f'确认评估{len(systems)}个系统×252例={len(signals):,}个信号对及识别结果；它们只有12句独立来源。噪声仍为原录音后段，不声称独立的新噪声录音。','',
      '## 三方向新说话人结果','','单元格SI-SDR dB / WER；适配器按全部完整种子平均，输入与McNet为单次固定基线。','','|系统组|目标−40°|目标20°|目标70°|','|---|---:|---:|---:|']
    for name in ('noisy','mcnet'):
        values={str(t):rt.aggregate([r for r in rows([name],target=t) if r['condition']!='clean'],ai) for t in (-40,20,70)}
        lines.append('|'+name+'|'+'|'.join(f'{values[str(t)]["metrics"]["si_sdr_db"]:.2f} / {values[str(t)]["asr"]["wer"]:.1%}' for t in (-40,20,70))+'|')
    for cell,d in summary['cells'].items():lines.append('|'+cell+'|'+'|'.join(f'{d["by_target"][str(t)]["metrics"]["si_sdr_db"]:.2f} / {d["by_target"][str(t)]["asr"]["wer"]:.1%}' for t in (-40,20,70))+'|')
    lines+=['','![方向迁移](../artifacts/g5_rotation_v1/evaluation/figures/fresh_direction_transfer.png)','','## 旋转减固定：非训练方向的配对差异','','先在每句内合并条件与配对种子，再按12句做2,000次探索性bootstrap。只有3位说话人，不作广泛显著性或新颖性声明。WER是编辑错误比例，可因插入超过100%，不是人声丢失百分比。','','|干净约束|ΔSI-SDR dB [95%区间]|ΔSTOI|ΔWER百分点 [95%区间]|','|---|---|---:|---|']
    for v in ('without_clean_identity','complete'):
        c=summary['comparisons'][v+'_off_target']['conditional_source_bootstrap_2000'];s=c['si_sdr_db'];w=c['wer']
        lines.append(f'|{v}|{s["delta"]:+.3f} [{s["ci95"][0]:+.3f}, {s["ci95"][1]:+.3f}]|{c["stoi"]["delta"]:+.4f}|{w["delta"]*100:+.2f} [{w["ci95"][0]*100:+.2f}, {w["ci95"][1]*100:+.2f}]|')
    lines+=['','## 干净语音与质量权衡','','|系统组|干净SI-SDR dB|干净STOI|干净WER|噪声PESQ|','|---|---:|---:|---:|---:|']
    for cell,d in summary['cells'].items():lines.append(f'|{cell}|{d["clean"]["metrics"]["si_sdr_db"]:.3f}|{d["clean"]["metrics"]["stoi"]:.4f}|{d["clean"]["asr"]["wer"]:.2%}|{d["noisy"]["metrics"]["pesq_wb"]:.3f}|')
    lines+=['','必须同时检查20°名义条件、非训练方向和干净输入：不能把方向泛化收益当成全条件更优。完整逐方向、点声源/八源、每种子与全部错误计数保留于机器汇总/逐例文件。固定Whisper tiny.en仍有识别器偏好及未知预训练重叠，WER变化不能独立证明音素完整或真实吞字。没有新增人工听感结论。',
      '', '## 限制、审计与后续','','这些结果限于名义阵列的自由场平面波、短英文句、三个新说话人和0dB；未验证真实房间、混响、阵列更换、距离变化及硬件误差。旋转增强、采样均衡、LoRA和重建约束均有先前研究。若旋转已解释大部分收益，后续创新必须胜过这个更强基线，不能只胜过原始McNet。负结果同样保留，不保证论文发表。',
      '',f'完成审计：{audit["status"]}。检查新说话人与旧项目隔离、每人4句、配对训练引用一致、完整种子组、原模型冻结、输入输出哈希、各阶段报告先于下一阶段、确认集不用于选择，以及试听单通道链接。',
      '', '- [G5A数据协议](G5A_配对旋转数据与新语句协议.md)、[G5B训练报告](G5B_整场景旋转训练报告.md)、[G5C信号报告](G5C_新说话人信号指标报告.md)。',
      '- [汇总](../artifacts/g5_rotation_v1/evaluation/summary.json)、[新说话人试听](../artifacts/g5_rotation_v1/review/index.html)、[审计](../artifacts/g5_rotation_v1/completion_audit.json)。',
      '', '来源仍为官方[LibriSpeech](https://www.openslr.org/12)、[DEMAND](https://zenodo.org/records/1227121)、[McNet](https://github.com/Audio-WestlakeU/McNet)、[loralib](https://github.com/microsoft/LoRA)与[Whisper](https://github.com/openai/whisper)。',
      '', '生成时间UTC：'+datetime.now(timezone.utc).isoformat()]
    report=ROOT/'docs/G5_整场景旋转与新说话人确认报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rt.archive('completion',report,[OUT/'completion_audit.json',OUT/'paired_data_audit.json',EVAL/'summary.json',EVAL/'asr_provenance.json',OUT/'review/selection.json',ROOT/'scripts/g5_evaluate.py',Path(__file__)])
    rt.dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'stage':'G5','completed_utc':datetime.now(timezone.utc).isoformat(),'report':str(report.relative_to(ROOT))});print(json.dumps({'stage':'G5','status':'complete','systems':len(systems),'pairs':len(signals),'audit':audit['status']}),flush=True)
