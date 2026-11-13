#!/usr/bin/env python3
"""G4 audit, factual report, figures and deterministic counterfactual listening."""
import hashlib,html,json
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
from g4_run import ROOT,OUT,EVAL,read,dump,sha,aggregate,archive

def make_review(cases,signals,asrs,systems):
    ai={(r['system'],r['sample_id']):r for r in asrs};si={(r['system'],r['sample_id']):r for r in signals}
    bysp=defaultdict(set)
    for r in cases:bysp[r['speaker_id']].add(r['source_id'])
    first={sorted(v)[0] for v in bysp.values()};chosen=sorted([r for r in cases if r['source_id'] in first and r['condition']=='normal' and r['noise_id']=='OOFFICE' and r['noise_azimuth']==-60 and r['snr_db']==0],key=lambda r:(r['source_id'],r['target_azimuth']))
    assert len(chosen)==12
    originals={};peak=0
    for r in chosen:
        sid=r['sample_id'];originals[(sid,'reference')]=sf.read(r['reference_path'],dtype='float32')[0]
        for system in systems:originals[(sid,system['system'])]=sf.read(si[(system['system'],sid)]['output_path'],dtype='float32')[0]
    peak=max(float(np.abs(a).max()) for a in originals.values());gain=np.float32(.95/peak if peak>1 else 1.)
    folder=OUT/'review';folder.mkdir(exist_ok=False);cards=[];audio_count=0
    for r in chosen:
        sid=r['sample_id'];players=[]
        for role in ['reference']+[s['system'] for s in systems]:
            a=originals[(sid,role)];rf=folder/'audio'/role;rf.mkdir(parents=True,exist_ok=True);p=rf/(sid+'.wav');sf.write(p,a*gain,16000,subtype='FLOAT');audio_count+=1
            score='' if role=='reference' else f'；ASR错误 {ai[(role,sid)]["score"]["errors"]}；'+html.escape(ai[(role,sid)]['hypothesis'])
            players.append(f'<p>{html.escape(role)}{score}<br><audio controls preload="none" src="audio/{html.escape(role)}/{sid}.wav"></audio></p>')
        cards.append(f'<section><h2>{r["source_id"]}：目标{r["target_azimuth"]}°，噪声−60°</h2><p>{html.escape(r["transcript"])}</p>'+''.join(players)+'</section>')
    page='<!doctype html><meta charset="utf-8"><title>G4 方向变化对照</title><style>body{max-width:1000px;margin:30px auto;font:16px system-ui;line-height:1.6}section{border:1px solid #bbb;padding:20px;margin:20px 0}audio{width:480px;max-width:100%}</style><h1>G4 同一语音的方向变化</h1><p>每位说话人按ID固定选一句，三种目标方向；非结果挑选、非盲听。对应方向组的第5麦带噪和干净参考完全相同，其他通道空间关系不同。固定噪声角度时，相对夹角也会改变，不能单独证明过拟合；整体旋转对照见报告。全部试听角色共用衰减 '+f'{float(gain):.6f}'+'，不参与计分。没有新的人工听感判定。</p>'+''.join(cards)
    (folder/'index.html').write_text(page,encoding='utf-8');dump(folder/'selection.json',{'rule':'first source per test speaker, OOoffice/noise-60/SNR0, all three target directions','sample_ids':[r['sample_id'] for r in chosen],'common_gain_all_players':float(gain),'audio_count':audio_count})

def finish_report(cases,systems):
    signals=read(EVAL/'signals.jsonl');asrs=read(EVAL/'asr_scores.jsonl');ai={(r['system'],r['sample_id']):r for r in asrs}
    summary={'coverage':{'signal_rows':len(signals),'asr_rows':len(asrs),'sources':len({r['source_id'] for r in cases}),'speakers':len({r['speaker_id'] for r in cases}),'cases_per_system':len(cases)},'systems':{},'directional_drops':{},'fault_deltas':{}}
    def subset(system,condition,ta=None):return [r for r in signals if r['system']==system and r['condition']==condition and (ta is None or r['target_azimuth']==ta)]
    for sy in systems:
        name=sy['system'];groups={}
        for condition in ('normal','rotation','gain','latency','clean'):
            groups[condition]={'all':aggregate(subset(name,condition),ai)}
            if condition in ('normal','rotation','clean'):
                groups[condition]['by_target']={str(ta):aggregate(subset(name,condition,ta),ai) for ta in (-30,20,60)}
        summary['systems'][name]=groups;summary['directional_drops'][name]={}
        for condition in ('normal','rotation','clean'):
            g=groups[condition]['by_target'];center=g['20'];summary['directional_drops'][name][condition]={str(ta):{
                'si_sdr_change_from_target20':g[str(ta)]['metrics']['si_sdr_db']-center['metrics']['si_sdr_db'],
                'wer_change_from_target20':g[str(ta)]['asr']['wer']-center['asr']['wer']} for ta in (-30,60)}
        base=[r for r in subset(name,'normal',20) if r['snr_db']==0];bg=aggregate(base,ai)
        summary['fault_deltas'][name]={kind:{'si_sdr_delta':groups[kind]['all']['metrics']['si_sdr_db']-bg['metrics']['si_sdr_db'],
             'wer_delta':groups[kind]['all']['asr']['wer']-bg['asr']['wer']} for kind in ('gain','latency')}
    dump(EVAL/'summary.json',summary)
    controls=defaultdict(set);inputok=True
    for r in cases:
        inputok &= sha(r['noisy_path'])==r['input_sha256'] and sha(r['reference_path'])==r['reference_sha256']
        x,sr=sf.read(r['noisy_path'],dtype='float32',always_2d=True);s,ssr=sf.read(r['reference_path'],dtype='float32');assert sr==ssr==16000
        controls[r['counterfactual_group']].add((hashlib.sha256(x[:,4].tobytes()).hexdigest(),hashlib.sha256(s.tobytes()).hexdigest()))
    prep=json.loads((OUT/'stages/preparation/archive_manifest.json').read_text());sigstage=json.loads((OUT/'stages/signal/archive_manifest.json').read_text())
    prov=json.loads((EVAL/'provenance.json').read_text());ap=json.loads((EVAL/'asr_provenance.json').read_text())
    keys={(r['system'],r['sample_id']) for r in signals};expected={(s['system'],r['sample_id']) for s in systems for r in cases}
    checks={'all_signal_and_asr_pairs_present':keys==expected=={(r['system'],r['sample_id']) for r in asrs} and len(signals)==len(asrs)==len(expected),
       'all_signal_metrics_ok':all(r['status']=='ok' for r in signals),'input_reference_hashes_unchanged':bool(inputok),
       'raw_reference_samples_identical_within_counterfactual_groups':all(len(v)==1 for v in controls.values()),
       'output_hashes':all(sha(r['output_path'])==r['output_sha256'] for r in signals),
       'preparation_archived_before_inference':datetime.fromisoformat(prep['saved_at_utc'])<datetime.fromisoformat(prov['started_utc']),
       'signal_report_archived_before_asr':datetime.fromisoformat(sigstage['saved_at_utc'])<datetime.fromisoformat(ap['started_utc']),
       'adapter_checkpoints_unchanged':all(s['kind']!='adapter' or sha(ROOT/s['path'])==s['checkpoint_sha256'] for s in systems),
       'no_training':prov['training_performed'] is False}
    import runpy
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g4_audit');repo=ROOT/'third_party/McNet';rt['verify_upstream'](repo,repo/rt['OFFLINE_CHECKPOINT']);checks['official_base_unchanged']=True
    make_review(cases,signals,asrs,systems)
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            if tag=='audio':self.paths.append(dict(attrs)['src'])
    parser=Links();parser.feed((OUT/'review/index.html').read_text());checks['review_84_mono_links']=len(parser.paths)==84 and all(sf.info(OUT/'review'/p).channels==1 and sf.info(OUT/'review'/p).samplerate==16000 for p in parser.paths)
    audit={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'counterfactual_groups':len(controls),'coverage':summary['coverage']};dump(OUT/'completion_audit.json',audit)
    if audit['status']!='ok':raise RuntimeError('G4 audit failed; evidence retained')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=EVAL/'figures';figures.mkdir();fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    ordinary='ordinary_s20261112'
    for ax,method in zip(axes,('complete_s20261112','without_clean_identity_s20261112')):
        z=[]
        for ta in (-30,20,60):
            vals=[]
            for relative in (-80,25):
                def score(name):return np.mean([r['metrics']['si_sdr_db'] for r in subset(name,'rotation',ta) if r['relative_angle']==relative])
                vals.append(score(method)-score(ordinary))
            z.append(vals)
        z=np.array(z);im=ax.imshow(z,cmap='coolwarm',vmin=-max(1,float(np.abs(z).max())),vmax=max(1,float(np.abs(z).max())))
        ax.set_xticks([0,1],['-80','+25']);ax.set_yticks([0,1,2],['-30','20 (training)','60']);ax.set_xlabel('Relative noise azimuth (deg)');ax.set_ylabel('Target azimuth (deg)');ax.set_title(method.replace('_s20261112',''))
        for i in range(3):
            for j in range(2):ax.text(j,i,f'{z[i,j]:+.2f}',ha='center',va='center',color='black')
        fig.colorbar(im,ax=ax,label='SI-SDR minus ordinary (dB)')
    fig.suptitle('Rigid scene rotations, fixed relative separation; 8 reused sources, one seed');fig.savefig(figures/'rotation_deltas.png',dpi=170);plt.close(fig)
    lines=['# G4 空间泛化压力测试报告','','## 结论与边界','','本阶段对冻结权重进行方向和通道误差压力测试，没有训练或调参。它使用G3保留测试中的8句、4位说话人（1/3/2/2句），不是独立新语句，也不是实测阵列验证。所有方法只使用预先固定的种子20261112，不能据此声称跨种子稳定。',
      '', '共6系统×472例=2,832个信号及识别对。288例将目标方向、噪声方向、SNR和噪声类型完全交叉；96例保持相对方位、整体旋转场景；64例增益/固定延迟扰动；24例干净输入。每个空间反事实组的参考第5麦与干净参考都逐样本完全相同，审计通过。这使模型输出差异不能归因于该麦录到的词语内容改变。','',
      '但固定噪声方向时，相对夹角会随目标变化；整体旋转虽保持夹角，阵列各向异性仍改变空间可分性。因此结果证明方向敏感性，不单独证明训练方向过拟合的因果机制，也不能把ASR退步当成真实吞字。','',
      '## 固定噪声角度，目标方向完全交叉','','单元格为SI-SDR dB / WER；噪声、SNR、源句均配对。','','|系统|目标−30°|训练方向20°|目标60°|','|---|---:|---:|---:|']
    for s in systems:
        g=summary['systems'][s['system']]['normal']['by_target'];lines.append('|'+s['system']+'|'+ '|'.join(f'{g[str(t)]["metrics"]["si_sdr_db"]:.2f} / {g[str(t)]["asr"]["wer"]:.1%}' for t in (-30,20,60))+'|')
    lines+=['','## 保持相对方向，整体旋转对照','','相对方位−80/+25°，SNR0，两种噪声。此处不与上一表混合计平均。','','|系统|目标−30°|目标20°|目标60°|','|---|---:|---:|---:|']
    for s in systems:
        g=summary['systems'][s['system']]['rotation']['by_target'];lines.append('|'+s['system']+'|'+ '|'.join(f'{g[str(t)]["metrics"]["si_sdr_db"]:.2f} / {g[str(t)]["asr"]["wer"]:.1%}' for t in (-30,20,60))+'|')
    lines+=['','![旋转对照](../artifacts/g4_counterfactual_v1/evaluation/figures/rotation_deltas.png)','','## 通道误差与干净输入','','增益为各通道±1.5dB内，延迟为固定0/±1/±2样本，参考第5麦均不扰动。下表是扰动减对应未扰动条件，SI-SDR负值与WER正值表示退步。','','|系统|增益ΔSI / ΔWER百分点|延迟ΔSI / ΔWER百分点|','|---|---:|---:|']
    for s in systems:
        name=s['system'];d=summary['fault_deltas'][name];lines.append(f'|{name}|{d["gain"]["si_sdr_delta"]:+.2f} / {100*d["gain"]["wer_delta"]:+.2f}|{d["latency"]["si_sdr_delta"]:+.2f} / {100*d["latency"]["wer_delta"]:+.2f}|')
    lines+=['','|干净输入系统|目标−30° SI / WER|目标20° SI / WER|目标60° SI / WER|','|---|---:|---:|---:|']
    for s in systems:
        g=summary['systems'][s['system']]['clean']['by_target'];lines.append('|'+s['system']+'|'+ '|'.join(f'{g[str(t)]["metrics"]["si_sdr_db"]:.2f} / {g[str(t)]["asr"]["wer"]:.1%}' for t in (-30,20,60))+'|')
    lines+=['','## 解释与下一步','','以上为探索性小样本诊断，不作总体显著性声明。若适配收益依赖固定目标方向，下一步优先把随机目标方向的数据增强当作强对照，再讨论新模块；这属于成熟思路，不是创新点。若开展后续训练，应另行登记验证与新的、说话人数更均衡的确认语句，不能把G3/G4发现集回用为独立确认。','',
      '本阶段指标仍为SI-SDR、真实宽带PESQ、标准STOI与固定Whisper tiny.en。主推理FP32，计分读取保存的原幅度FLOAT WAV；ASR不接收参考文本提示。已有ASR缓存只按完全相同的float32样本及固定模型/解码来源复用。试听副本全页共用衰减，不参与指标。','',
      '未覆盖：真实录音、混响、近场距离变化、时钟漂移、其他阵列及其他识别器。4位说话人且一人只有一句，无新增人工听感结论。','',
      '## 留档入口','', '- [G4A数据与协议](G4A_数据与压力测试协议.md)、[G4B信号评估](G4B_冻结模型信号评估报告.md)。',
      '- [逐组机器汇总](../artifacts/g4_counterfactual_v1/evaluation/summary.json)、[同一句的方向变化试听](../artifacts/g4_counterfactual_v1/review/index.html)。',
      '- [完成审计](../artifacts/g4_counterfactual_v1/completion_audit.json)、`artifacts/g4_counterfactual_v1/stages/`阶段快照。','',f'审计状态：{audit["status"]}；生成时间UTC：{datetime.now(timezone.utc).isoformat()}。']
    report=ROOT/'docs/G4_空间泛化压力测试报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('completion',report,[OUT/'completion_audit.json',EVAL/'summary.json',EVAL/'asr_provenance.json',OUT/'review/selection.json',Path(__file__)])
    dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'stage':'G4','completed_utc':datetime.now(timezone.utc).isoformat(),'report':str(report.relative_to(ROOT))})
    print(json.dumps({'stage':'G4','status':'complete','pairs':len(signals),'audit':audit['status']}),flush=True)
