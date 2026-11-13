"""Approved G35-P1 B/C and signal curves only; no ASR or training."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='1'
import argparse,csv,gc,hashlib,json,math,shutil,time,traceback
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g35_azimuth_scan_v1';DATA=ROOT/'data/g35_azimuth_scan_v1'
ANGLES=list(range(-90,91,10));METRICS=['si_sdr_db','pesq_wb','stoi'];SEEDS=[20261112,20261113,20261114]
SYSTEMS=['noisy','mcnet']+[f'g5/fixed_without_clean_identity_s{s}' for s in SEEDS]+[f'g15/rigid_without_clean_identity_s{s}' for s in SEEDS]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def history_guard():
    before=json.loads((ROOT/'artifacts/g35_admission_v1/protected_before.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in before.items()),'Protected historical file changed'
def guard(start,limit):
    if time.monotonic()-start>limit-120:raise TimeoutError('Frozen stage time reserve reached')
    if shutil.disk_usage(ROOT).free<6*1024**3:raise RuntimeError('Remaining storage reserve insufficient')
def csv_write(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def build():
    from mcse.geometry import rotate_azimuth,azimuth_degrees,wrap_degrees
    from mcse.spatial_noise import direction,highpass40
    from mcse.simulation import render_plane_wave,geometric_delays,mix_at_snr,speech_activity_mask,snr_db
    start=time.monotonic();assert not OUT.exists() and not DATA.exists(),'Never overwrite existing scan'
    assert json.loads((ROOT/'artifacts/g35_admission_v1/result.json').read_text())['status']=='passed'
    history_guard();assert shutil.disk_usage(ROOT).free>=16*1024**3
    OUT.mkdir();DATA.mkdir();(DATA/'noisy').mkdir();(DATA/'reference').mkdir()
    sources=read(ROOT/'data/g17_v1/sources/speech_manifest.jsonl');old=read(ROOT/'data/g17_v1/manifest.jsonl')
    assert len(sources)==8
    pos=np.array(json.loads((ROOT/'configs/g1_geometry.json').read_text())['microphone_positions_m'])
    noises={};noise_sources=json.loads((ROOT/'artifacts/g17_fresh_confirmation_v1/noise_sources.json').read_text())
    for r in noise_sources:
        assert sha(r['path'])==r['sha256'];a,sr=sf.read(r['path'],dtype='float64');assert sr==16000 and a.ndim==1
        noises[r['noise_id']]=highpass40(a)
    plan={'stage':'B_C_signal_only','authorization':'User explicitly approved B,C after A; prioritize three signal curves; no ASR/training',
          'protocol_sha256':sha(ROOT/'docs/G35_前方180度方位扫描冻结协议_待审核.md'),
          'admission_sha256':sha(ROOT/'artifacts/g35_admission_v1/result.json'),'script_sha256':sha(__file__),
          'angles':ANGLES,'systems':SYSTEMS,'seeds':SEEDS,'expected_inputs':1064,'expected_outputs':7448,'expected_signal_rows':8512,
          'source_manifest_sha256':sha(ROOT/'data/g17_v1/sources/speech_manifest.jsonl'),
          'parent_manifest_sha256':sha(ROOT/'data/g17_v1/manifest.jsonl'),'max_B_seconds':1200,'max_C_seconds':5400,'max_plot_seconds':1800}
    dump(OUT/'frozen_plan.json',plan)
    records=[];group_reports=[]
    with (DATA/'manifest.jsonl').open('x') as log:
        for source in sources:
            path=ROOT/'data/g17_v1/sources'/source['path'];assert sha(path)==source['sha256']
            mono,sr=sf.read(path,dtype='float64');assert sr==16000 and len(mono)==source['samples']
            speech={theta:render_plane_wave(mono,geometric_delays(pos,direction(theta,10))) for theta in ANGLES}
            bases=[r for r in old if r['source_id']==source['source_id'] and r['target_azimuth']==20];assert len(bases)==7
            for base in bases:
                guard(start,1200);variants=[];expected_gain=None
                if base['condition']!='clean':
                    clips=[noises[base['noise_id']][lo:hi] for lo,hi in base['noise_intervals']]
                    assert all(len(clip)==len(mono) for clip in clips)
                for theta in ANGLES:
                    s=speech[theta]
                    if base['condition']=='clean':x=s;gain=0.
                    else:
                        nf=np.zeros_like(s);k=len(clips)
                        for clip,u in zip(clips,base['base_noise_directions']):
                            moved=rotate_azimuth(u,theta-20)
                            assert abs(moved@direction(theta,10)-np.asarray(u)@direction(20,10))<=1e-12
                            if k==1:assert abs(wrap_degrees(azimuth_degrees(moved)-theta-base['relative_angle']))<=1e-10
                            delay=geometric_delays(pos,moved);assert 32+math.ceil(float(np.abs(delay).max()))+1<=64
                            nf+=render_plane_wave(clip,delay)/np.sqrt(k)
                        x,_,gain=mix_at_snr(s,nf,0,speech_activity_mask(s[:,4]))
                    if expected_gain is None:expected_gain=gain
                    assert gain==expected_gain,'Cross-angle gain differs; do not repair channels'
                    variants.append((theta,x,s,gain))
                scale=min(1.,.95/max(float(np.abs(a).max()) for _,x,s,_ in variants for a in (x,s)))
                ref_hashes=set();target_hashes=set();snr_errors=[];maxpeak=0.
                for theta,x,s,gain in variants:
                    sid=f'g35_{len(records):04d}_{source["source_id"]}_{theta:+04d}'
                    xp=DATA/'noisy'/(sid+'.wav');rp=DATA/'reference'/(sid+'.wav')
                    with xp.open('xb') as f:sf.write(f,x*scale,16000,format='WAV',subtype='PCM_24')
                    with rp.open('xb') as f:sf.write(f,s[:,4]*scale,16000,format='WAV',subtype='PCM_24')
                    xx,xsr=sf.read(xp,dtype='float32',always_2d=True);ss,ssr=sf.read(rp,dtype='float32')
                    assert xsr==ssr==16000 and xx.shape==(len(mono)+128,6) and ss.shape==(len(xx),)
                    assert np.isfinite(xx).all() and np.isfinite(ss).all() and np.abs(xx).max()<=.95+2**-23
                    assert np.abs(ss).max()<=.95+2**-23
                    ref_hashes.add(hashlib.sha256(xx[:,4].tobytes()).hexdigest());target_hashes.add(hashlib.sha256(ss.tobytes()).hexdigest())
                    if base['condition']=='clean':assert np.array_equal(xx[:,4],ss)
                    else:
                        error=abs(snr_db(ss,xx[:,4]-ss,speech_activity_mask(ss)))
                        assert error<=.002;snr_errors.append(error)
                    maxpeak=max(maxpeak,float(np.abs(xx).max()))
                    # Fixed band-SNR diagnostic: no signal modification, frequency-band energy on saved reference.
                    f=np.fft.rfftfreq(len(ss),1/16000);band=(f>=100)&(f<=8000)
                    bsnr=None
                    if base['condition']!='clean':
                        sp=np.fft.rfft(ss.astype(float));npow=np.fft.rfft((xx[:,4]-ss).astype(float))
                        bsnr=float(10*np.log10(np.sum(np.abs(sp[band])**2)/np.sum(np.abs(npow[band])**2)))
                    row={**base,'sample_id':sid,'split':'viewed_development_19_angle','target_azimuth':theta,
                        'noise_azimuth':theta+base['relative_angle'] if base['relative_angle'] is not None else None,
                        'actual_relative_azimuth':base['relative_angle'],'geometry_protocol':'rigid_rotation',
                        'noisy_path':str(xp),'reference_path':str(rp),'input_sha256':sha(xp),'reference_sha256':sha(rp),
                        'common_scale':scale,'noise_gain':gain,'band_SNR_100_8000_whole_utterance_db':bsnr,
                        'parent_G17_sample_id':base['sample_id'],'counterfactual_group':base['sample_id'],
                        'actual_noise_directions':[rotate_azimuth(u,theta-20).tolist() for u in base['base_noise_directions']]}
                    records.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                assert len(ref_hashes)==len(target_hashes)==1,'Natural reference differs; stop, no replacement'
                group_reports.append({'parent':base['sample_id'],'angles':19,'reference_unique_hashes':len(ref_hashes),
                    'target_unique_hashes':len(target_hashes),'gain':expected_gain,'scale':scale,
                    'saved_SNR_max_error_db':max(snr_errors,default=0),'input_max_peak':maxpeak})
                dump(OUT/'status.json',{'stage':'B','groups_done':len(group_reports),'rows':len(records),'seconds':time.monotonic()-start})
    assert len(records)==1064 and sum(r['condition']!='clean' for r in records)==912 and len(group_reports)==56
    assert len({r['sample_id'] for r in records})==1064
    assert all(sha(r['noisy_path'])==r['input_sha256'] and sha(r['reference_path'])==r['reference_sha256'] for r in records)
    history_guard();dump(OUT/'data_group_audit.json',group_reports)
    done={'status':'passed','rows':1064,'noisy':912,'clean':152,'groups':56,'manifest_sha256':sha(DATA/'manifest.jsonl'),
          'seconds':time.monotonic()-start,'natural_reference_all_groups_exact':True,'all_saved_SNR_within_0p002dB':True}
    dump(OUT/'DATA_COMPLETE.json',done)
    report=ROOT/'docs/G35B_19角输入与几何验收报告.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G35B 19角输入与几何验收报告\n\n'+json.dumps(done,ensure_ascii=False,indent=2)+'\n\n56组均正确共同旋转、组内19角共用尺度；参考自然一致，无通道替换。保留原片段/方向/完整语句，PCM24读回及SNR全量核查；仅旧G17开发来源，不是独立最终集。\n\n[逐组原始记录](../artifacts/g35_azimuth_scan_v1/data_group_audit.json)；[新清单](../data/g35_azimuth_scan_v1/manifest.jsonl)。\n')
    print(json.dumps(done),flush=True)

_evaluator=None
def score(row):
    global _evaluator
    from mcse.evaluation import MetricEvaluator,waveform_diagnostics
    if _evaluator is None:_evaluator=MetricEvaluator()
    s,sr=sf.read(row['reference_path'],dtype='float32');y,yr=sf.read(row['output_path'],dtype='float32')
    assert sr==yr==16000
    row['metrics']=_evaluator.evaluate(s,y);row['waveform_diagnostics']=waveform_diagnostics(s,y)
    row['status']='ok' if all(v['status']=='ok' for v in row['metrics']['status'].values()) else 'metric_failure'
    return row

def signals():
    import torch
    from mcse.runtime import load_system,enhance_array
    start=time.monotonic();assert not (OUT/'signals.jsonl').exists(),'No implicit rerun/resume'
    done=json.loads((OUT/'DATA_COMPLETE.json').read_text());assert done['status']=='passed' and sha(DATA/'manifest.jsonl')==done['manifest_sha256']
    history_guard();rows=read(DATA/'manifest.jsonl');assert len(rows)==1064
    torch.set_num_threads(2);torch.manual_seed(2026113501)
    torch.backends.cudnn.enabled=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    assert str(torch.__version__)=='2.5.1+cu121' and not torch.are_deterministic_algorithms_enabled() and torch.cuda.is_available()
    result=[];resource=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool,(OUT/'signals.jsonl').open('x') as log:
        pending=set()
        def collect(block):
            nonlocal pending
            if not pending:return
            ready,pending=wait(pending,timeout=60 if block else 0,return_when=FIRST_COMPLETED)
            guard(start,5400)
            for future in ready:
                r=future.result();log.write(json.dumps(r,allow_nan=False)+'\n');log.flush();result.append(r)
                if r['status']!='ok':raise RuntimeError('Metric failure retained; stop without changing criteria')
        for system in SYSTEMS:
            guard(start,5400);began=time.monotonic();runtime=None
            if system!='noisy':
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();runtime=load_system(system,'cuda')
            folder=OUT/'audio'/system.replace('/','_');folder.mkdir(parents=True,exist_ok=False)
            for i,case in enumerate(rows):
                guard(start,5400);assert sha(case['noisy_path'])==case['input_sha256'] and sha(case['reference_path'])==case['reference_sha256']
                x,sr=sf.read(case['noisy_path'],dtype='float32',always_2d=True);assert sr==16000
                if runtime is None:y=x[:,4].copy();elapsed=0.
                else:y,elapsed=enhance_array(runtime,x)
                assert y.shape==(len(x),) and np.isfinite(y).all()
                path=folder/(case['sample_id']+'.wav')
                with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
                yy,yr=sf.read(path,dtype='float32');assert yr==16000 and np.array_equal(y,yy)
                pending.add(pool.submit(score,{**case,'system':system,'output_path':str(path),'output_sha256':sha(path),'inference_seconds':elapsed}))
                while len(pending)>=8:collect(True)
                if (i+1)%100==0:dump(OUT/'status.json',{'stage':'C','system':system,'input_done':i+1,'scored':len(result),'seconds':time.monotonic()-start})
            while pending:collect(True)
            resource.append({'system':system,'rows':1064,'seconds_with_metrics':time.monotonic()-began,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated() if runtime else None,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved() if runtime else None,'runtime':runtime.metadata if runtime else None})
            if runtime:del runtime;gc.collect();torch.cuda.empty_cache()
            dump(OUT/'resources.json',resource);print('completed '+system+' '+str(len(result)),flush=True)
    assert len(result)==8512 and Counter(r['system'] for r in result)==Counter({s:1064 for s in SYSTEMS})
    assert len({(r['system'],r['sample_id']) for r in result})==8512
    history_guard()
    done={'status':'passed','rows':8512,'enhancement_outputs':7448,'system_count':8,'seconds':time.monotonic()-start,
          'signals_sha256':sha(OUT/'signals.jsonl'),'all_metrics_ok':True,'ASR_run':False,'training_run':False}
    dump(OUT/'SIGNAL_COMPLETE.json',done);print(json.dumps(done),flush=True)

def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    start=time.monotonic();complete=json.loads((OUT/'SIGNAL_COMPLETE.json').read_text())
    assert complete['status']=='passed' and sha(OUT/'signals.jsonl')==complete['signals_sha256']
    rows=read(OUT/'signals.jsonl');assert len(rows)==8512 and all(r['status']=='ok' for r in rows)
    dest=OUT/'curves';dest.mkdir(exist_ok=False)
    font='/mnt/c/Windows/Fonts/msyh.ttc';assert Path(font).exists()
    font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','font.size':10})
    def family(s):return 'fixed' if s.startswith('g5/') else 'rigid' if s.startswith('g15/') else s
    def mean(items,k):return float(np.mean([r['metrics'][k] for r in items]))
    noisy=[r for r in rows if r['condition']!='clean'];clean=[r for r in rows if r['condition']=='clean']
    cells=defaultdict(list);bygroup=defaultdict(list)
    for r in noisy:
        cells[(r['system'],r['target_azimuth'])].append(r)
        bygroup[(r['system'],r['target_azimuth'],r['noise_id'],r['condition'],str(r['relative_angle']))].append(r)
    profiles=[]
    for (system,theta),items in sorted(cells.items()):
        assert len(items)==48
        profiles.append({'system':system,'family':family(system),'angle':theta,'n':48,**{k:mean(items,k) for k in METRICS}})
    csv_write(dest/'per_seed_19_angles.csv',profiles)
    csv_write(dest/'noise_field_groups.csv',[{'system':s,'angle':a,'noise':n,'field':f,'relative':rel,'n':len(v),**{k:mean(v,k) for k in METRICS}} for (s,a,n,f,rel),v in sorted(bygroup.items())])
    FAMILIES=['noisy','mcnet','fixed','rigid'];labels={'noisy':'带噪参考麦','mcnet':'官方 McNet','fixed':'单方位适配 LoRA','rigid':'几何保持式多方位 LoRA'}
    means={}
    for fam in FAMILIES:
        for k in METRICS:
            means[(fam,k)]=np.array([np.mean([p[k] for p in profiles if p['family']==fam and p['angle']==theta]) for theta in ANGLES])
    csv_write(dest/'mean_19_angles.csv',[{'family':fam,'angle':theta,**{k:float(means[(fam,k)][i]) for k in METRICS}} for fam in FAMILIES for i,theta in enumerate(ANGLES)])
    summary=[];inside=np.array([(-45<=a<=75) for a in ANGLES])
    for fam in FAMILIES:
        for k in METRICS:
            v=means[(fam,k)];summary.append({'family':fam,'metric':k,'mean_19':float(v.mean()),'worst':float(v.min()),
                'worst_angles':','.join(str(ANGLES[i]) for i in np.flatnonzero(v==v.min())),
                'best':float(v.max()),'best_angles':','.join(str(ANGLES[i]) for i in np.flatnonzero(v==v.max())),
                'range':float(np.ptp(v)),'angle_std_population':float(np.std(v)),
                'inside_mean':float(v[inside].mean()),'inside_worst':float(v[inside].min()),
                'outside_mean':float(v[~inside].mean()),'outside_worst':float(v[~inside].min())})
    csv_write(dest/'summary_19.csv',summary)
    per_seed=[]
    for system in SYSTEMS:
        for k in METRICS:
            v=np.array([p[k] for p in sorted([p for p in profiles if p['system']==system],key=lambda p:p['angle'])])
            per_seed.append({'system':system,'metric':k,'mean':float(v.mean()),'worst':float(v.min()),'range':float(np.ptp(v))})
    csv_write(dest/'per_seed_summary.csv',per_seed)
    clean_table=[{'system':system,'n':152,**{k:mean([r for r in clean if r['system']==system],k) for k in METRICS},
        'RMS_gain':float(np.mean([r['waveform_diagnostics']['estimate_rms']/np.sqrt(np.mean(sf.read(r['reference_path'],dtype='float64')[0]**2)) for r in clean if r['system']==system]))} for system in SYSTEMS]
    csv_write(dest/'clean_controls.csv',clean_table)
    # All paired individual differences, including unfavorable samples and every seed.
    lookup={(r['system'],r['sample_id']):r for r in noisy};pairs=[]
    for seed in SEEDS:
        fixed=f'g5/fixed_without_clean_identity_s{seed}';rigid=f'g15/rigid_without_clean_identity_s{seed}'
        for r in [r for r in noisy if r['system']==fixed]:
            other=lookup[(rigid,r['sample_id'])]
            pairs.append({'seed':seed,'sample_id':r['sample_id'],'angle':r['target_azimuth'],'speaker':r['speaker_id'],
                'noise':r['noise_id'],'field':r['condition'],'relative':r['relative_angle'],**{k:other['metrics'][k]-r['metrics'][k] for k in METRICS}})
    assert len(pairs)==2736;csv_write(dest/'all_paired_differences.csv',pairs)
    units=['SI-SDR (dB)','PESQ-WB','STOI'];colors={'noisy':'#555555','mcnet':'#111111','fixed':'#2467b2','rigid':'#d35a17'}
    fig,axes=plt.subplots(3,1,figsize=(10,11),sharex=True)
    for ax,k,unit in zip(axes,METRICS,units):
        ax.axvspan(-45,75,color='#afc4b1',alpha=.15,label='多方位训练覆盖区间' if k==METRICS[0] else None)
        ax.axvline(20,color='#777',ls=':',lw=1)
        for fam in FAMILIES:
            if fam in ('fixed','rigid'):
                for sys in [s for s in SYSTEMS if family(s)==fam]:
                    values=[next(p[k] for p in profiles if p['system']==sys and p['angle']==a) for a in ANGLES]
                    ax.plot(ANGLES,values,color=colors[fam],alpha=.22,lw=.8)
            ax.plot(ANGLES,means[(fam,k)],'-o',color=colors[fam],ms=3,lw=1.8,label=labels[fam])
        ax.set_ylabel(unit);ax.grid(alpha=.2);ax.set_xlim(-90,90)
    axes[0].legend(ncol=2,fontsize=9);axes[0].set_title('前方180°离散方位开发诊断（粗线：三种子均值；细线：全部种子）')
    axes[-1].set_xticks(ANGLES);axes[-1].set_xlabel('目标水平方位（度；俯仰固定 +10°）')
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(dest/('three_metrics_19_angles.'+ext),dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for ax,k,unit in zip(axes,METRICS,units):
        ax.axvspan(-45,75,color='#afc4b1',alpha=.15);ax.axhline(0,color='black',lw=.8);ax.axvline(20,color='#777',ls=':')
        for seed in SEEDS:ax.plot(ANGLES,[np.mean([p[k] for p in pairs if p['seed']==seed and p['angle']==a]) for a in ANGLES],color=colors['rigid'],alpha=.3,lw=1)
        ax.plot(ANGLES,means[('rigid',k)]-means[('fixed',k)],'-o',color=colors['rigid'],ms=3)
        ax.set_ylabel('Δ '+unit);ax.grid(alpha=.2)
    axes[0].set_title('几何保持式多方位 − 单方位 LoRA（同种子、同输入配对）')
    axes[-1].set_xticks(ANGLES);axes[-1].set_xlabel('目标水平方位（度）');fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(dest/('paired_differences_19_angles.'+ext),dpi=300)
    plt.close(fig)
    report=ROOT/'docs/G35C_19角三项信号指标与方向权衡报告.md'
    lines=['# G35C 19角三项信号指标与方向权衡报告','',
        '用户补充批准A通过后执行B/C，优先交三项曲线。A已通过；B1064输入与C8512行信号评分完整，含7448增强输出。无训练、ASR或G34恢复。',
        '', '8句/4人旧G17开发来源，19角、两噪声、三场型。带噪主比较与干净控制分开；全部三个种子，所有不利记录保留。每角48基础条件、每人两句且各条件等量，等权聚合等价于先话人/句子/条件再种子平均。',
        '', '![三项曲线](../artifacts/g35_azimuth_scan_v1/curves/three_metrics_19_angles.png)',
        '', '![配对差值](../artifacts/g35_azimuth_scan_v1/curves/paired_differences_19_angles.png)',
        '', '|系统|指标|19角均值|最差(角)|最好(角)|波动范围|区间内均值|区间外均值|',
        '|---|---|---:|---|---|---:|---:|---:|']
    for r in summary:lines.append(f"|{labels[r['family']]}|{r['metric']}|{r['mean_19']:.4f}|{r['worst']:.4f} ({r['worst_angles']})|{r['best']:.4f} ({r['best_angles']})|{r['range']:.4f}|{r['inside_mean']:.4f}|{r['outside_mean']:.4f}|")
    lines+=['', '区间内12点（−40至+70），外7点；全体按19点等权。表为平均曲线最差值，逐种子最差值另存；不能把曲线更平等同更好，也不能外推连续、360°、三维或真实阵列。',
        '', '当前只评价信号质量和干净保真，不提供新WER或人工听评结论。未做按话人bootstrap，本报告不作显著性主张。原始逐例与按噪声/场型/种子的分组CSV完整保存。',
        '', '## 产物与验证','',
        '[完整逐例信号](../artifacts/g35_azimuth_scan_v1/signals.jsonl)、[19点数据](../artifacts/g35_azimuth_scan_v1/curves/mean_19_angles.csv)、[逐种子数据](../artifacts/g35_azimuth_scan_v1/curves/per_seed_19_angles.csv)、[噪声场型分组](../artifacts/g35_azimuth_scan_v1/curves/noise_field_groups.csv)、[全部配对差](../artifacts/g35_azimuth_scan_v1/curves/all_paired_differences.csv)、[干净控制](../artifacts/g35_azimuth_scan_v1/curves/clean_controls.csv)。',
        '', '新共同尺度及显式TF32许可与旧G17口径不同，不用跨轮分差宣称方法收益；所有模型在本轮共用同一输入、参考及尺度。不得根据本开发成绩自动重训或择固定方向。',
        '', '完成后停止，ASR及后续训练仍需另行批准。']
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    dump(OUT/'SUMMARY_COMPLETE.json',{'status':'passed','seconds':time.monotonic()-start,'summary':summary,
        'files':{str(p.relative_to(ROOT)):sha(p) for p in dest.iterdir() if p.is_file()},'report_sha256':sha(report)})
    print('signal curves complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['B','C','curves']);args=parser.parse_args()
    try:
        if args.stage=='B':build()
        elif args.stage=='C':signals()
        else:summarize()
    except BaseException:
        if OUT.exists():
            p=OUT/(args.stage+'_FAILED.json')
            with p.open('x') as f:json.dump({'status':'failed','traceback':traceback.format_exc()},f,indent=2)
        raise
