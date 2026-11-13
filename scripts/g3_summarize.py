#!/usr/bin/env python3
"""Summarize all completed systems, matched comparisons and review candidates."""
import hashlib,html,json,os,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.asr_scoring import corpus_score
OUT=ROOT/'artifacts/g3_overnight_v1';EVAL=OUT/'evaluation'
METRICS=('si_sdr_db','pesq_wb','stoi')
def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def aggregate(items,asr_index):
    good=[r for r in items if r['status']=='ok'];answer={'cases':len(items),'successful':len(good),'failures':len(items)-len(good),
        'sources':len({r['source_id'] for r in items}),'speakers':len({r['speaker_id'] for r in items}),
        'metrics':{k:float(np.mean([r['metrics'][k] for r in good])) if good else None for k in METRICS}}
    scores=[asr_index[(r['system'],r['sample_id'])]['score'] for r in good if (r['system'],r['sample_id']) in asr_index]
    answer['asr']=corpus_score(scores);answer['asr']['scored_cases']=len(scores)
    return answer

def comparison(signals,asrs,systems_a,systems_b,label):
    assert len(systems_a)==len(systems_b)
    sig={(r['system'],r['sample_id']):r for r in signals if r['status']=='ok' and r['split']=='test'}
    asr={(r['system'],r['sample_id']):r for r in asrs if r['split']=='test'}
    groups=defaultdict(list);speaker={}
    for (system,sid),r in sig.items():
        if system!=systems_a[0]:continue
        vals={k:[] for k in METRICS};error_differences=[];word_counts=[]
        for a,b in zip(systems_a,systems_b):
            ar,br=sig[(a,sid)],sig[(b,sid)]
            for k in METRICS:vals[k].append(ar['metrics'][k]-br['metrics'][k])
            aa,bb=asr[(a,sid)]['score'],asr[(b,sid)]['score'];assert aa['N']==bb['N']
            error_differences.append(aa['errors']-bb['errors']);word_counts.append(aa['N'])
        groups[r['source_id']].append({**{k:float(np.mean(v)) for k,v in vals.items()},
            'error_delta':float(np.mean(error_differences)),'words':float(np.mean(word_counts))})
        speaker[r['source_id']]=r['speaker_id']
    source_ids=sorted(groups);data=np.array([[*[np.mean([r[k] for r in groups[s]]) for k in METRICS],
        sum(r['error_delta'] for r in groups[s]),sum(r['words'] for r in groups[s])] for s in source_ids])
    def calc(d):return np.r_[d[:,:3].mean(axis=0),d[:,3].sum()/d[:,4].sum()]
    rng=np.random.default_rng(2026111204);draws=np.array([calc(data[rng.integers(0,len(data),len(data))]) for _ in range(2000)])
    point=calc(data);lo,hi=np.quantile(draws,[.025,.975],axis=0)
    result={'label':label,'a':systems_a,'b':systems_b,'delta_definition':'A minus B; SI/PESQ/STOI higher is better; WER lower is better',
        'source_clusters':len(data),'speaker_clusters':len(set(speaker.values())),
        'source_paired_bootstrap_2000':{k:{'delta':float(point[i]),'ci95':[float(lo[i]),float(hi[i])]} for i,k in enumerate((*METRICS,'wer'))},
        'caution':'Conditional exploratory intervals over source utterances, only four test speakers and two noise recordings. Multiple conditions/seeds do not create independent sources. No broad significance or novelty claim.'}
    speaker_values={sp:calc(data[[i for i,s in enumerate(source_ids) if speaker[s]==sp]]).tolist() for sp in sorted(set(speaker.values()))}
    result['per_speaker_deltas_order_si_pesq_stoi_wer']=speaker_values
    macro=np.mean(list(speaker_values.values()),axis=0)
    result['speaker_equal_weight_sensitivity']={k:float(macro[i]) for i,k in enumerate((*METRICS,'wer'))}
    return result

def review(signals,asrs):
    systems=['noisy','mcnet','ordinary_s20261112','complete_s20261112'];idx={(r['system'],r['sample_id']):r for r in signals if r['status']=='ok' and r['split']=='test'}
    ai={(r['system'],r['sample_id']):r for r in asrs if r['split']=='test'};candidates=[]
    for (system,sid),r in idx.items():
        if system!='complete_s20261112':continue
        b=idx[('ordinary_s20261112',sid)];aa=ai[(system,sid)]['score'];bb=ai[('ordinary_s20261112',sid)]['score']
        candidates.append({'id':sid,'source':r['source_id'],'speaker':r['speaker_id'],
             'si_delta':r['metrics']['si_sdr_db']-b['metrics']['si_sdr_db'],'errors_delta':aa['errors']-bb['errors']})
    selected=[]
    def add(rows,reason,n):
        count=0
        for r in rows:
            if any(s['id']==r['id'] for s in selected):continue
            selected.append({**r,'reason':reason});count+=1
            if count>=n:break
    add(sorted([r for r in candidates if r['errors_delta']>0 or r['si_delta']<0],key=lambda r:(-r['errors_delta'],r['si_delta'],r['id'])),'ASR退步或SI-SDR较差候选，非人声误删结论',3)
    add(sorted([r for r in candidates if r['errors_delta']<0 or r['si_delta']>0],key=lambda r:(r['errors_delta'],-r['si_delta'],r['id'])),'至少一项指标改善的候选，可能仍有权衡',3)
    for sp in sorted({r['speaker'] for r in candidates}):
        add(sorted([r for r in candidates if r['speaker']==sp],key=lambda r:r['id']),'每位测试说话人的固定顺序抽查，不预判听感',1)
    folder=OUT/'review';folder.mkdir(exist_ok=False);(folder/'noisy_mic5').mkdir();cards=[]
    def src(path):return html.escape(os.path.relpath(path,folder).replace(os.sep,'/'),quote=True)
    for selected_case in selected:
        sid=selected_case['id'];r=idx[('noisy',sid)];x,sr=sf.read(r['input_path'],dtype='float32',always_2d=True)
        role_audio={'reference':sf.read(r['reference_path'],dtype='float32')[0],'noisy':x[:,4].copy()}
        for system in systems:
            if system!='noisy':role_audio[system]=sf.read(idx[(system,sid)]['output_path'],dtype='float32')[0]
        peak=max(float(np.abs(a).max()) for a in role_audio.values());gain=np.float32(.95/peak if peak>1 else 1.)
        selected_case['listening_common_gain']=float(gain);selected_case['original_role_peaks']={k:float(np.abs(a).max()) for k,a in role_audio.items()}
        listen_paths={}
        for role,a in role_audio.items():
            rolefolder=folder/('noisy_mic5' if role=='noisy' else 'audio/'+role);rolefolder.mkdir(parents=True,exist_ok=True)
            path=rolefolder/(sid+'.wav');expected=(a*gain).astype('float32');sf.write(path,expected,sr,subtype='FLOAT')
            actual,_=sf.read(path,dtype='float32');assert np.array_equal(actual,expected);listen_paths[role]=path
        players=[f'<p>同例共同试听衰减系数：{float(gain):.6f}；不用于计分。</p><p>干净参考 <audio controls preload="none" src="{src(listen_paths["reference"])}"></audio></p>']
        for system in systems:
            item=idx[(system,sid)];ar=ai[(system,sid)];path=listen_paths[system];m=item['metrics']
            players.append(f'<p><b>{html.escape(system)}</b> SI-SDR {m["si_sdr_db"]:.2f} / PESQ {m["pesq_wb"]:.3f} / STOI {m["stoi"]:.4f} / ASR错误 {ar["score"]["errors"]}<br><audio controls preload="none" src="{src(path)}"></audio><br>{html.escape(ar["hypothesis"])}</p>')
        cards.append(f'<section><h2>{html.escape(sid)}</h2><p>{html.escape(selected_case["reason"])}；{html.escape(r["noise_id"])} / {r["noise_field"]} / SNR {r["snr_db"]} dB</p><p>原文：{html.escape(r["transcript"])}</p>'+''.join(players)+'</section>')
    page='<!doctype html><meta charset="utf-8"><title>G3 对照试听</title><style>body{max-width:1000px;margin:30px auto;font:16px system-ui;line-height:1.6}section{padding:20px;border:1px solid #bbb;margin:20px 0}audio{width:460px;max-width:100%}</style><h1>G3 对照试听</h1><p>本页含改善、退步候选和每位说话人的固定顺序抽查。候选由客观结果筛选，不是盲听；ASR漏词不等于真实声音被删除。带噪只播放第5麦。计分使用原幅度输出；若试听存在越界峰值，只对同例全部角色施加共同衰减，保留相对幅度，系数逐例列出。尚未赋予人工听感结论。</p>'+''.join(cards)
    (folder/'index.html').write_text(page,encoding='utf-8');dump(folder/'selection.json',selected)

def main():
    assert (EVAL/'ASR_COMPLETE.json').exists()
    signals=read(EVAL/'signals.jsonl');asrs=read(EVAL/'asr_scores.jsonl');ai={(r['system'],r['sample_id']):r for r in asrs}
    systems=json.loads((EVAL/'systems.json').read_text());summary={'test':{},'clean_test':{},'subgroups':{},'comparisons':[]}
    for sy in systems:
        name=sy['system']
        for split in ('test','clean_test'):summary[split][name]=aggregate([r for r in signals if r['system']==name and r['split']==split],ai)
        summary['subgroups'][name]={}
        for field in ('noise_id','noise_field','snr_db','speaker_id','noise_azimuth'):
            groups=defaultdict(list)
            for r in signals:
                if r['system']==name and r['split']=='test':groups[str(r[field])].append(r)
            summary['subgroups'][name][field]={k:aggregate(v,ai) for k,v in groups.items()}
    names={s['system'] for s in systems};seeds=[s for s in (20261112,20261113,20261114) if f'complete_s{s}' in names and f'ordinary_s{s}' in names]
    comparisons=[(['mcnet'],['noisy'],'原始McNet减带噪'),(['ordinary_s20261112'],['mcnet'],'普通适配减原始McNet'),
                 (['complete_s20261112'],['ordinary_s20261112'],'首种子完整方案减普通适配'),
                 ([f'complete_s{s}' for s in seeds],[f'ordinary_s{s}' for s in seeds],'配对种子平均：完整方案减普通适配'),
                 ([f'complete_s{s}' for s in seeds],[f'without_stratification_s{s}' for s in seeds],'空间分层消融：三种子，后两种子为验证触发扩展'),
                 ([f'complete_s{s}' for s in seeds],[f'without_clean_identity_s{s}' for s in seeds],'干净约束消融：三种子，后两种子为验证触发扩展'),
                 ([f'without_clean_identity_s{s}' for s in seeds],[f'ordinary_s{s}' for s in seeds],'无干净约束时的空间分层效果：三种子，验证触发扩展')]
    for a,b,label in comparisons:
        if all(s in names for s in a+b):summary['comparisons'].append(comparison(signals,asrs,a,b,label))
    summary['coverage']={'signal_rows':len(signals),'asr_rows':len(asrs),'signal_status':dict(Counter(r['status'] for r in signals)),
                         'test_sources':30,'test_speakers':4,'test_mixtures_per_system':360,'clean_controls_per_system':30,'paired_seeds':seeds}
    sp_sources=defaultdict(set)
    for r in signals:
        if r['system']=='mcnet' and r['split']=='test':sp_sources[r['speaker_id']].add(r['source_id'])
    summary['coverage']['test_source_counts_per_speaker']={k:len(v) for k,v in sp_sources.items()}
    summary['limitations']=['Short 1–3.8s English clean-source utterances, four held-out speakers, two test noise recordings.',
       'No room reverberation or real microphone recordings; nominal far-field plane-wave simulation only.',
       'Point-source test angle is coupled with SNR: +45 degrees only at 0 dB, -60 degrees at +/-5 dB; angle-specific causal conclusions are invalid.',
       'Multi8 is an unconditioned eight independent-segment sum, not a measured diffuse field.',
       'Single ASR is recognizer-specific and cannot prove acoustic word deletion.',
       'Test source counts per speaker are 10,10,9,1; the fourth speaker has only one utterance. Speaker-macro sensitivity is secondary and highly variable.',
       'LoRA, balancing, replay, and clean identity constraints are known ideas. Novelty is not established by beating original McNet.',
       'Source bootstrap intervals are exploratory given only four speakers; ablation seeds 2 and 3 were registered after first-seed validation, before any final-test evaluation.']
    dump(EVAL/'summary.json',summary);review(signals,asrs)
    lines=['# G3 指标速览','','所有指标均为本次实际计算；WER按编辑次数/参考词数汇总。','','|系统|SI-SDR dB|PESQ|STOI|WER|成功/总数|','|---|---:|---:|---:|---:|---:|']
    for name,r in summary['test'].items():
        m=r['metrics'];lines.append(f'|{name}|{m["si_sdr_db"]:.3f}|{m["pesq_wb"]:.3f}|{m["stoi"]:.4f}|{r["asr"]["wer"]:.2%}|{r["successful"]}/{r["cases"]}|')
    lines+=['','## 完整方案减普通适配（配对种子平均）','']
    c=next(r for r in summary['comparisons'] if r['label'].startswith('配对种子'))
    for k,v in c['source_paired_bootstrap_2000'].items():lines.append(f'- {k}: {v["delta"]:.6f}，探索性95%区间 [{v["ci95"][0]:.6f}, {v["ci95"][1]:.6f}]')
    lines+=['','## 限制','']+['- '+s for s in summary['limitations']]
    (EVAL/'metrics_overview.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(summary['coverage']))
if __name__=='__main__':main()
