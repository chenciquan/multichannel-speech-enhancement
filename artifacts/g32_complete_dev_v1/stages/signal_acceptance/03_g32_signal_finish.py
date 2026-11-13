"""Accept all frozen full-utterance outputs before any ASR scores exist."""
from collections import defaultdict
import numpy as np
import soundfile as sf
from g32_common import ROOT,OUT,BASE,sha,read,dump,now,meta,archive,status
from g32_signal import guard
from mcse.evaluation import si_sdr


def finish():
    p=guard();done=meta(OUT/'SIGNAL_COMPLETE.json');assert done['status']=='ok' and sha(OUT/'signals.jsonl')==done['signals_sha256']
    cases={r['sample_id']:r for r in read(BASE/'manifest.jsonl')};rows=read(OUT/'signals.jsonl')
    expected={(s,m,c) for s in ('mcnet','cirm','cirm_mask','cirm_shape') for m in ('direct','mvdr','mwf') for c in cases}|{('noisy','direct',c) for c in cases}
    assert len(rows)==len(expected)==468;seen=set();refs={}
    for r in rows:
        key=r['source_system'],r['method'],r['sample_id'];assert key in expected and key not in seen;seen.add(key)
        c=cases[r['sample_id']];assert all(r[k]==v for k,v in c.items())
        assert r['status']=='ok' and not r['oracle_clean_information'] and r['full_source'] and r['ASR_scoring_allowed']
        assert sha(r['output_path'])==r['output_sha256'] and sha(r['reference_path'])==r['reference_sha256']
        if r['reference_path'] not in refs:refs[r['reference_path']]=sf.read(r['reference_path'],dtype='float32')[0]
        y,sr=sf.read(r['output_path'],dtype='float32');s=refs[r['reference_path']]
        assert sr==16000 and y.shape==s.shape==(r['frames'],) and np.isfinite(y).all()
        assert all(np.isfinite(r['metrics'][m]) and r['metrics']['status'][m]['status']=='ok' for m in ('si_sdr_db','pesq_wb','stoi'))
        assert abs(si_sdr(s,y)-r['metrics']['si_sdr_db'])<1e-8
        if r['condition']=='clean':
            a=s.astype(float);b=y.astype(float);power=float(a@a)
            assert np.isclose((b-a)@(b-a)/power,r['clean_levels']['reference_normalized_squared_error'],rtol=1e-12,atol=1e-14)
            assert np.isclose(np.sqrt((b@b)/power),r['clean_levels']['rms_gain_ratio'],rtol=1e-12,atol=1e-14)
    assert seen==expected
    def aggregate(rs):
        a={'rows':len(rs),'unique_sources':len({r['source_id'] for r in rs}),**{m:float(np.mean([r['metrics'][m] for r in rs])) for m in ('si_sdr_db','pesq_wb','stoi')}}
        if all(r['condition']=='clean' for r in rs):a['clean_levels']={m:float(np.mean([r['clean_levels'][m] for r in rs])) for m in ('rms_gain_ratio','reference_normalized_squared_error')}
        return a
    dimensions={'overall':('source_system','method','clean'),'by_rt':('source_system','method','requested_rt60','clean'),
        'by_speaker':('source_system','method','speaker_id','clean'),'by_source':('source_system','method','source_id','clean'),
        'by_field':('source_system','method','noise_field','requested_rt60')};summary={}
    for section,fields in dimensions.items():
        buckets=defaultdict(list)
        for r in rows:
            values={**r,'clean':r['condition']=='clean'};buckets[tuple(values[k] for k in fields)].append(r)
        summary[section]=[{**dict(zip(fields,k)),**aggregate(rs)} for k,rs in buckets.items()]
    dump(OUT/'signal_summary.json',summary,True)
    by={(r['source_system'],r['method'],r['clean']):r for r in summary['overall']}
    report=ROOT/'docs/G32_完整开发语句信号与干净电平报告.md'
    lines=['# G32 完整开发语句信号与干净电平','','468条保存音频及36条件覆盖通过；全部SI-SDR与干净RMS/NMSE从WAV独立重算。以下只汇总完整旧开发语句，不是新独立说话人或G31主比较替代。此刻尚无本轮ASR评分。','',
        '|系统|输出|带噪SI-SDR|PESQ|STOI|干净RMS增益比|干净NMSE|','|---|---|---:|---:|---:|---:|---:|']
    for (s,m,clean),r in sorted(by.items()):
        if clean:continue
        c=by[s,m,True]['clean_levels'];lines.append(f"|{s}|{m}|{r['si_sdr_db']:.4f}|{r['pesq_wb']:.4f}|{r['stoi']:.4f}|{c['rms_gain_ratio']:.4f}|{c['reference_normalized_squared_error']:.5f}|")
    lines+=['','每个系统带噪24条件、干净12条件，反复使用四条完整语句/两人，原转写共16词token。按RT、说话人、语句、场型的细分完整保存；源方向/噪声类型轮换耦合，不作单因素因果解释。原电平保留，SI-SDR可忽略整体增益，不能单独证明字词保留。与G31中心裁剪没有完全配对噪声/尺度，不能将跨阶段差异归因于长度。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    marker={'status':'ok','utc':now(),'rows':468,'cases':36,'SI_recomputed':468,'signal_summary_sha256':sha(OUT/'signal_summary.json'),
        'signals_sha256':sha(OUT/'signals.jsonl'),'report_sha256':sha(report),'ASR_performed':False}
    dump(OUT/'SIGNAL_ANALYSIS_COMPLETE.json',marker,True)
    archive('signal_analysis',report,[OUT/'SIGNAL_ANALYSIS_COMPLETE.json',OUT/'signal_summary.json',OUT/'SIGNAL_COMPLETE.json',ROOT/'scripts/g32_signal_finish.py'])


def accept():
    guard();done=meta(OUT/'SIGNAL_ANALYSIS_COMPLETE.json');assert done['status']=='ok' and meta(OUT/'power_signal_helper.json')['status']=='released'
    assert sha(OUT/'signals.jsonl')==done['signals_sha256'] and sha(OUT/'signal_summary.json')==done['signal_summary_sha256']
    stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    audit={'status':'ok','utc':now(),'rows':468,'cases':36,'SI_recomputed':468,'power_released':True,'stages':stages,
        'signals_sha256':sha(OUT/'signals.jsonl'),'ASR_performed':False,'training_performed':False}
    dump(OUT/'SIGNAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G32_完整开发语句信号阶段验收.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G32 完整开发语句信号阶段验收\n\n全部468行覆盖、保存波形与SI/干净指标独立重算通过，模型参数/数据/代码与阶段快照通过哈希核对。临时唤醒已释放，无新增训练或本轮ASR。完整强对照保留；下一步固定双ASR的原始及电平条件分别评分，并增加原干声参考，不用新成绩改判G31。\n')
    archive('signal_acceptance',report,[OUT/'SIGNAL_ACCEPTANCE_AUDIT.json',OUT/'power_signal_helper.json',ROOT/'scripts/g32_signal_finish.py'])
    status(stage='signal_accepted_ASR_not_started')
