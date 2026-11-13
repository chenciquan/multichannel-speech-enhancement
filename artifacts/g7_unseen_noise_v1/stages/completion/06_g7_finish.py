#!/usr/bin/env python3
"""Stratified frozen-model transfer, integrity checks and stage archive."""
import hashlib,html,json,runpy
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
import g4_run as rt
from g5_finish import compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g7_unseen_noise_v1';BASE=ROOT/'data/g7_v1';EVAL=OUT/'evaluation'

def make_review(cases,signals,asrs,systems):
    seed=20261112;names=['noisy','mcnet']+[s['system'] for s in systems if s.get('seed')==seed];idx={(r['system'],r['sample_id']):r for r in signals};ai={(r['system'],r['sample_id']):r for r in asrs}
    first=min(r['source_id'] for r in cases)
    selected=sorted([r for r in cases if r['source_id']==first and r['condition']=='point' and r['relative_angle']==-80 and r['snr_db']==0],key=lambda r:(r['noise_id'],r['target_azimuth']));assert len(selected)==6
    audio={}
    for r in selected:
        sid=r['sample_id'];audio[(sid,'reference')]=sf.read(r['reference_path'],dtype='float32')[0]
        for name in names:audio[(sid,name)]=sf.read(idx[(name,sid)]['output_path'],dtype='float32')[0]
    peak=max(float(np.abs(a).max()) for a in audio.values());gain=np.float32(.95/peak if peak>1 else 1.);folder=OUT/'review';folder.mkdir();cards=[]
    for r in selected:
        sid=r['sample_id'];players=[]
        for name in ['reference']+names:
            p=folder/'audio'/name/(sid+'.wav');p.parent.mkdir(parents=True,exist_ok=True);sf.write(p,audio[(sid,name)]*gain,16000,subtype='FLOAT')
            text='' if name=='reference' else f'；ASR错误{ai[(name,sid)]["score"]["errors"]}；'+html.escape(ai[(name,sid)]['hypothesis'])
            players.append(f'<p>{name}{text}<br><audio controls preload="none" src="audio/{name}/{sid}.wav"></audio></p>')
        cards.append(f'<section><h2>{r["noise_id"]}，目标{r["target_azimuth"]}°</h2><p>{html.escape(r["transcript"])}</p>'+''.join(players)+'</section>')
    page='<!doctype html><meta charset="utf-8"><title>G7 新噪声对照</title><style>body{max-width:1000px;margin:30px auto;font:16px system-ui;line-height:1.6}section{padding:20px;border:1px solid #bbb;margin:20px 0}audio{width:500px;max-width:100%}</style><h1>G7 新噪声对照</h1><p>按ID固定选择第一源句，仅一位说话人，不代表全部12句；两个新噪声、三个方向，0dB、点声源相对−80°。固定首种子，不按结果选择。非盲听、无人为听感标签。全页共同试听衰减'+f'{float(gain):.6f}'+'，不用于计分；WER不是声音丢失百分比。</p>'+''.join(cards)
    (folder/'index.html').write_text(page,encoding='utf-8');rt.dump(folder/'selection.json',{'rule':'first lexical source ID, both noise IDs,all3 targets,SNR0,point relative-80,seed20261112','source_id':first,'sample_ids':[r['sample_id'] for r in selected],'common_gain':float(gain),'audio_count':42})

def finish_report(cases,systems):
    rt.OUT=OUT;rt.EVAL=EVAL;signals=rt.read(EVAL/'signals.jsonl');asr=rt.read(EVAL/'asr_scores.jsonl');ai={(r['system'],r['sample_id']):r for r in asr};plan=json.loads((OUT/'frozen_plan.json').read_text())
    cells={f'{g}_{v}':[s['system'] for s in systems if s.get('geometry')==g and s.get('variant')==v] for g in ('fixed','rotated') for v in ('without_clean_identity','complete')}
    assert all(len(v)==3 for v in cells.values())
    def choose(names,noise=None,snr=None,target=None,field=None):return [r for r in signals if r['system'] in names and (noise is None or r['noise_id']==noise) and (snr is None or r['snr_db']==snr) and (target is None or r['target_azimuth']==target) and (field is None or r['condition']==field)]
    summary={'coverage':{'signal_rows':len(signals),'asr_rows':len(asr),'sources':12,'speakers':3,'systems':14,'cases_per_system':648,'seeds':[20261112,20261113,20261114]},'cells':{},'baselines':{},'comparisons':{},'recognizer_warnings':{}}
    for cell,names in cells.items():
        summary['cells'][cell]={noise:{str(ta):rt.aggregate(choose(names,noise=noise,target=ta),ai) for ta in (-40,20,70)} for noise in plan['noise_ids']}
    for baseline in ('noisy','mcnet'):summary['baselines'][baseline]={noise:{str(ta):rt.aggregate(choose([baseline],noise=noise,target=ta),ai) for ta in (-40,20,70)} for noise in plan['noise_ids']}
    for variant in ('without_clean_identity','complete'):
        for noise in plan['noise_ids']:
            for target in (-40,20,70):
                subset=[r for r in signals if r['noise_id']==noise and r['target_azimuth']==target]
                key=f'{variant}|{noise}|{target}';summary['comparisons'][key]=compare(subset,asr,cells['rotated_'+variant],cells['fixed_'+variant],False)
    summary['per_snr_direction']={}
    for cell,names in cells.items():
        summary['per_snr_direction'][cell]={noise:{str(snr):{str(ta):rt.aggregate(choose(names,noise,snr,ta),ai) for ta in (-40,20,70)} for snr in (-5,0,5)} for noise in plan['noise_ids']}
    summary['per_field']={cell:{field:rt.aggregate(choose(names,field=field),ai) for field in ('point','multi8')} for cell,names in cells.items()}
    warning_candidates=[]
    for system in systems:
        name=system['system'];items=[r for r in asr if r['system']==name];noisy={r['sample_id']:r for r in asr if r['system']=='noisy'}
        summary['recognizer_warnings'][name]={'cases':len(items),'WER_over100_cases':sum(r['score']['wer']>1 for r in items),'insertions_exceed_reference_words':sum(r['score']['I']>r['score']['N'] for r in items),
          'more_errors_than_noisy_cases':sum(r['score']['errors']>noisy[r['sample_id']]['score']['errors'] for r in items),'more_deletions_than_noisy_cases':sum(r['score']['D']>noisy[r['sample_id']]['score']['D'] for r in items)}
        for r in sorted(items,key=lambda r:(-r['score']['I']/r['score']['N'],-r['score']['wer'],r['sample_id']))[:3]:warning_candidates.append({'system':name,**r,'notice':'Recognizer diagnostic only, not acoustic word-deletion label'})
    rt.dump(EVAL/'summary.json',summary);rt.dump(EVAL/'ASR_warning_candidates.json',warning_candidates)
    manifest=json.loads((OUT/'stages/preparation/archive_manifest.json').read_text());inputok=rt.sha(BASE/'manifest.jsonl')==manifest['manifest_sha256'];canonical=defaultdict(set)
    for r in cases:
        inputok &= rt.sha(r['noisy_path'])==r['input_sha256'] and rt.sha(r['reference_path'])==r['reference_sha256']
        x,sr=sf.read(r['noisy_path'],dtype='float32',always_2d=True);s,rr=sf.read(r['reference_path'],dtype='float32');assert sr==rr==16000
        canonical[r['counterfactual_group']].add((hashlib.sha256(x[:,4].tobytes()).hexdigest(),hashlib.sha256(s.tobytes()).hexdigest()))
    previous_noise=json.loads((ROOT/'data/g3_v1/sources/noise_manifest.json').read_text());new_noise=json.loads((OUT/'noise_sources.json').read_text())
    expected={(s['system'],r['sample_id']) for s in systems for r in cases};prov=json.loads((EVAL/'provenance.json').read_text());ap=json.loads((EVAL/'asr_provenance.json').read_text());sigarchive=json.loads((OUT/'stages/signal/archive_manifest.json').read_text())
    checks={'all9072_signal_and_asr_pairs':len(signals)==len(asr)==9072 and {(r['system'],r['sample_id']) for r in signals}==expected=={(r['system'],r['sample_id']) for r in asr},
       'all_metric_rows_ok':all(r['status']=='ok' for r in signals),'input_hashes_match':bool(inputok),'counterfactual_raw_and_reference_equal':all(len(v)==1 for v in canonical.values()),
       'all_output_hashes_match':all(rt.sha(r['output_path'])==r['output_sha256'] for r in signals),'frozen_adapter_hashes_match':all(s['kind']!='adapter' or rt.sha(ROOT/s['path'])==s['checkpoint_sha256'] for s in systems),
       'new_noise_ids_and_files_not_previously_used':not {r['noise_id'] for r in new_noise}&{r['noise_id'] for r in previous_noise} and not {r['sha256'] for r in new_noise}&{r['sha256'] for r in previous_noise},
       'bounded_noise_member_downloads':all(r['range_bytes_received']<=20*1024**2 and not r['full_archive_hash_verified'] for r in new_noise),
       'data_archived_before_inference':datetime.fromisoformat(manifest['saved_utc'])<datetime.fromisoformat(prov['started_utc']),
       'signal_report_archived_before_ASR':datetime.fromisoformat(sigarchive['saved_at_utc'])<datetime.fromisoformat(ap['started_utc']),
       'no_training_or_new_selection':prov['training_performed'] is False}
    bridge=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g7_audit');repo=ROOT/'third_party/McNet';bridge['verify_upstream'](repo,repo/bridge['OFFLINE_CHECKPOINT']);checks['official_base_unchanged']=True
    make_review(cases,signals,asr,systems)
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            if tag=='audio':self.paths.append(dict(attrs)['src'])
    parser=Links();parser.feed((OUT/'review/index.html').read_text());checks['42_mono_review_links']=len(parser.paths)==42 and all(sf.info(OUT/'review'/p).channels==1 and sf.info(OUT/'review'/p).samplerate==16000 for p in parser.paths)
    audit={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'coverage':summary['coverage'],'counterfactual_groups':len(canonical)};rt.dump(OUT/'completion_audit.json',audit)
    if audit['status']!='ok':raise RuntimeError('G7 audit failed; preserve evidence')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=EVAL/'figures';figures.mkdir();fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for column,noise in enumerate(plan['noise_ids']):
        for variant,color in [('without_clean_identity','#4769a8'),('complete','#b54c35')]:
            for geometry,style in [('fixed','--'),('rotated','-')]:
                values=summary['cells'][geometry+'_'+variant][noise];x=[-40,20,70]
                axes[0,column].plot(x,[values[str(t)]['metrics']['si_sdr_db'] for t in x],style+'o',color=color,label=geometry+' / '+variant)
                axes[1,column].plot(x,[values[str(t)]['asr']['wer']*100 for t in x],style+'o',color=color)
        axes[0,column].set_title(noise);axes[0,column].set_ylabel('SI-SDR (dB)');axes[1,column].set_ylabel('WER (%)')
        for ax in axes[:,column]:ax.set_xlabel('Target azimuth (deg)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('New noise recordings, reused12 G5 sources;3 seeds, all SNRs');fig.savefig(figures/'new_noise_direction_profiles.png',dpi=170);plt.close(fig)
    lines=['# G7 新噪声迁移与方向权衡报告','','## 范围与结果表','','G6资源和接口报告归档后开展，未训练或重新选择权重。固定G5全部12个适配器（三种子四单元）、原McNet和带噪参考。12句/3位说话人来自已查看的G5确认集：这是同源新噪声压力测试，不是新的独立说话人确认。',
       '', 'NPARK/PSTATION此前未在本项目使用；官方文件和许可已核实，每个仅有界提取ch01，成员CRC及本地SHA验证，不声称完整ZIP的MD5已核实。名义阵列模拟，不是实测六通道语音。原始预训练模型是否见过类似环境未知。',
       '', '三个目标方向×三个SNR×两种新噪声×两种点声源相对方位/八源叠加×12句，共648例/系统，14系统9,072个信号及ASR对。点声源占2/3，八源占1/3；三方向共享片段与参考麦克风样本，SNR标定前噪声固定40Hz高通。',
       '', '下表汇总全部SNR与空间条件；单元格SI-SDR dB / WER，适配器合并三种子。原始分噪声/SNR/方向/结构结果保留，不能只拿非训练方向平均值宣称全面改善。','','|新噪声|系统组|−40°|20°|70°|','|---|---|---:|---:|---:|']
    for noise in plan['noise_ids']:
        for name in ('noisy','mcnet')+tuple(cells):
            values=summary['baselines'][name][noise] if name in ('noisy','mcnet') else summary['cells'][name][noise]
            lines.append('|'+noise+'|'+name+'|'+'|'.join(f'{values[str(t)]["metrics"]["si_sdr_db"]:.2f} / {values[str(t)]["asr"]["wer"]:.1%}' for t in (-40,20,70))+'|')
    lines+=['','![新噪声方向曲线](../artifacts/g7_unseen_noise_v1/evaluation/figures/new_noise_direction_profiles.png)','','## 旋转减固定：分噪声与方向','','以下区间以12句为簇、先合并条件与配对种子，再bootstrap2,000次。只有3位说话人，多组比较仅探索性描述，未作正式多重比较推断。WER差为百分点，负值更好；SI-SDR正值更好。','','|约束|噪声|方向|ΔSI-SDR [区间]|ΔWER百分点 [区间]|','|---|---|---:|---|---|']
    for variant in ('without_clean_identity','complete'):
        for noise in plan['noise_ids']:
            for ta in (-40,20,70):
                c=summary['comparisons'][f'{variant}|{noise}|{ta}']['conditional_source_bootstrap_2000'];s=c['si_sdr_db'];w=c['wer']
                lines.append(f'|{variant}|{noise}|{ta}|{s["delta"]:+.2f} [{s["ci95"][0]:+.2f},{s["ci95"][1]:+.2f}]|{w["delta"]*100:+.1f} [{w["ci95"][0]*100:+.1f},{w["ci95"][1]*100:+.1f}]|')
    lines+=['','## 识别异常与解释边界','','微平均WER使用全部编辑次数/参考词数，不裁剪大于100%的值。额外记录WER>100%、插入数超过参考词数、相对带噪错误/删除增加的条件数；它们是识别退化筛查，不是真实吞字标签。源句/条件重复，不能把这些计数当成独立受试次数。','','|系统|WER>100%条件|插入>N条件|错误多于带噪条件|','|---|---:|---:|---:|']
    for name,w in summary['recognizer_warnings'].items():lines.append(f'|{name}|{w["WER_over100_cases"]}|{w["insertions_exceed_reference_words"]}|{w["more_errors_than_noisy_cases"]}|')
    lines+=['','主识别器仍为固定Whisper tiny.en，无参考文本提示，无ASR反传或测试调参。即使方向效应延续，也不能由单ASR确认人声细节。下一研究优先用独立CTC识别器交叉核查已有音频，先核实官方模型来源/许可，再登记协议；不把语言模型幻觉自动解读为增强器删词。',
       '', '本阶段没有新增干净输入证据，沿用原G5相关分析；也没有新的人工听感结论。仍缺真实混响、硬件阵列、更多独立说话人及更广噪声录音。所有方法组件为已有技术，不能由这里宣称创新已成立或全条件优势。',
       '', f'完成审计：{audit["status"]}。数据报告先于模型推理、信号报告先于ASR；原官方模型和所有选定适配器哈希未变，输入/输出/参考组及完整覆盖通过核对。',
       '', '- [数据协议](G7A_新噪声迁移协议与数据.md)、[信号指标报告](G7B_新噪声信号指标报告.md)。',
       '- [机器汇总与逐SNR结果](../artifacts/g7_unseen_noise_v1/evaluation/summary.json)、[固定源句试听](../artifacts/g7_unseen_noise_v1/review/index.html)、[审计](../artifacts/g7_unseen_noise_v1/completion_audit.json)。',
       '', '数据来源与许可依据：[DEMAND官方Zenodo记录](https://zenodo.org/records/1227121)。仅提取16k版本ch01；未将完整多通道噪声录音当作本项目实测混合语音。',
       '', '生成UTC：'+datetime.now(timezone.utc).isoformat()]
    report=ROOT/'docs/G7_新噪声迁移与方向权衡报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rt.archive('completion',report,[OUT/'completion_audit.json',EVAL/'summary.json',EVAL/'asr_provenance.json',OUT/'review/selection.json',ROOT/'scripts/g7_evaluate.py',Path(__file__)])
    rt.dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'stage':'G7','completed_utc':datetime.now(timezone.utc).isoformat(),'report':str(report.relative_to(ROOT)),'pairs':9072});print(json.dumps({'stage':'G7','status':'complete','pairs':9072,'audit':audit['status']}),flush=True)
