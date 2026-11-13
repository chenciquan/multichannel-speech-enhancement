"""Complete paired development accounting and apply the already pinned gate."""
from collections import defaultdict
import numpy as np
import soundfile as sf
from g31_common import ROOT,OUT,sha,read,dump,now,meta,archive,guard,status
from mcse.evaluation import si_sdr


def finish():
    plan=guard();protocol=plan['protocol'];done=meta(OUT/'SIGNAL_COMPLETE.json')
    assert done['status']=='ok' and done['rows']==1404 and sha(OUT/'signals.jsonl')==done['signals_sha256']
    cases={r['sample_id']:r for r in meta(OUT/'cases.json')};assert len(cases)==108
    rows=read(OUT/'signals.jsonl');seen=set();reference={}
    expected={(s,m,c) for s in ('mcnet','cirm','cirm_mask','cirm_shape') for m in ('direct','mvdr','mwf') for c in cases}
    expected|={('noisy','direct',c) for c in cases};assert len(expected)==1404
    for r in rows:
        key=r['source_system'],r['method'],r['sample_id'];assert key in expected and key not in seen;seen.add(key)
        c=cases[r['sample_id']]
        assert all(r[k]==c[k] for k in ('index','source_id','speaker_id','field','condition','requested_rt60','target_azimuth','reference_path','reference_sha256'))
        assert r['ASR_scoring_allowed'] is False and not r['oracle_clean_information'] and r['status']=='ok'
        assert sha(r['output_path'])==r['output_sha256']
        if r['reference_path'] not in reference:
            assert sha(r['reference_path'])==r['reference_sha256'];reference[r['reference_path']]=sf.read(r['reference_path'],dtype='float32')[0]
        y,sr=sf.read(r['output_path'],dtype='float32');ref=reference[r['reference_path']]
        assert sr==16000 and y.shape==ref.shape==(23679,) and np.isfinite(y).all()
        assert all(np.isfinite(r['metrics'][m]) and r['metrics']['status'][m]['status']=='ok' for m in ('si_sdr_db','pesq_wb','stoi'))
        assert abs(si_sdr(ref,y)-r['metrics']['si_sdr_db'])<1e-8
        if r['condition']=='clean':
            a=ref.astype(float);b=y.astype(float);power=float(a@a)
            assert np.isclose((b-a)@(b-a)/power,r['clean_levels']['reference_normalized_squared_error'],atol=1e-14,rtol=1e-12)
            assert np.isclose(np.sqrt((b@b)/power),r['clean_levels']['rms_gain_ratio'],atol=1e-14,rtol=1e-12)
    assert seen==expected
    def aggregate(rs):
        result={'rows':len(rs),'unique_sources':len({r['source_id'] for r in rs}),
            **{m:float(np.mean([r['metrics'][m] for r in rs])) for m in ('si_sdr_db','pesq_wb','stoi')}}
        if all(r['condition']=='clean' for r in rs):result['clean_levels']={m:float(np.mean([r['clean_levels'][m] for r in rs])) for m in ('rms_gain_ratio','reference_normalized_squared_error')}
        return result
    dimensions={'overall':('source_system','method','clean'),'by_rt':('source_system','method','requested_rt60','clean'),
        'by_speaker':('source_system','method','speaker_id','clean'),'by_source':('source_system','method','source_id','clean'),
        'by_field':('source_system','method','field','requested_rt60'),'by_noise':('source_system','method','noise_id','requested_rt60')}
    summary={}
    for section,fields in dimensions.items():
        buckets=defaultdict(list)
        for r in rows:
            values={**r,'clean':r['condition']=='clean'};buckets[tuple(values[f] for f in fields)].append(r)
        summary[section]=[{**dict(zip(fields,k)),**aggregate(v)} for k,v in buckets.items()]
    by={(r['source_system'],r['method'],r['clean']):r for r in summary['overall']}
    candidate=by['cirm_shape','mvdr',False];clean=by['cirm_shape','mvdr',True];comparisons=[];rules=protocol['extension_gate']
    assert candidate['rows']==72 and clean['rows']==36
    for comparator in protocol['primary_comparators']:
        b=by[comparator,'mvdr',False];c=by[comparator,'mvdr',True]
        delta={m:candidate[m]-b[m] for m in ('si_sdr_db','pesq_wb','stoi')}
        nmse=clean['clean_levels']['reference_normalized_squared_error']-c['clean_levels']['reference_normalized_squared_error']
        tests={'SI':delta['si_sdr_db']>=rules['mean_SI_SDR_delta_vs_each_comparator_min_db'],
            'PESQ':delta['pesq_wb']>=rules['mean_PESQ_delta_vs_each_comparator_min'],
            'STOI':delta['stoi']>=rules['mean_STOI_delta_vs_each_comparator_min'],
            'clean_NMSE':nmse<=rules['clean_mean_NMSE_increase_vs_each_comparator_max']}
        comparisons.append({'comparator':comparator,'deltas':delta,'clean_NMSE_increase':nmse,'checks':tests,'passed':all(tests.values())})
    speakers=[];bs={(r['source_system'],r['method'],r['speaker_id'],r['clean']):r for r in summary['by_speaker']}
    for sp in sorted({r['speaker_id'] for r in rows}):
        delta=bs['cirm_shape','mvdr',sp,False]['si_sdr_db']-bs['cirm','mvdr',sp,False]['si_sdr_db']
        speakers.append({'speaker':sp,'SI_delta_vs_cirm':delta,'passed':delta>=rules['each_development_speaker_mean_SI_delta_vs_cirm_min_db']})
    decision={'status':'ok','utc':now(),'extension_gate_passed':all(r['passed'] for r in comparisons+speakers),
        'comparisons':comparisons,'speaker_checks':speakers,'protocol_sha256':sha(ROOT/'configs/g31_development_protocol.json'),
        'scope':'single-seed development spending gate, not statistical significance/novelty/content retention proof'}
    dump(OUT/'summary.json',summary,True);dump(OUT/'primary_decision.json',decision,True)
    report=ROOT/'docs/G31_同预算目标开发信号与保真对照报告.md'
    lines=['# G31 同预算目标的开发信号与保真对照','','完整1404条信号评分覆盖108个开发条件；12条旧开发语句、两位说话人，约1秒裁剪加完整RIR尾，非完整句子识别或新说话人最终确认。训练统一登记1500步，未按此处成绩改checkpoint。', '',
        '|系统|输出|带噪SI-SDR|PESQ|STOI|干净RMS增益比|干净NMSE|','|---|---|---:|---:|---:|---:|---:|']
    for (system,method,clean_flag),r in sorted(by.items()):
        if clean_flag:continue
        c=by[system,method,True]['clean_levels'];lines.append(f"|{system}|{method}|{r['si_sdr_db']:.4f}|{r['pesq_wb']:.4f}|{r['stoi']:.4f}|{c['rms_gain_ratio']:.4f}|{c['reference_normalized_squared_error']:.6f}|")
    lines+=['','## 预登记主比较：统计辅助的MVDR','','|对照|ΔSI-SDR|ΔPESQ|ΔSTOI|干净NMSE增量|单项检查|','|---|---:|---:|---:|---:|---|']
    for r in comparisons:lines.append(f"|{r['comparator']}|{r['deltas']['si_sdr_db']:.5f}|{r['deltas']['pesq_wb']:.5f}|{r['deltas']['stoi']:.5f}|{r['clean_NMSE_increase']:.6f}|{r['checks']}|")
    lines+=['',f"预登记扩展门槛：{'通过，仅表示值得继续确认' if decision['extension_gate_passed'] else '未通过，不据此扩大统计辅助方案或宣称优于强对照'}。每位开发说话人的相对SI检查：{speakers}。", '',
        '主门槛早于训练冻结，未根据结果更换为直接输出或MWF。全部按RT、场型、噪声、说话人及源语句细分保存，无删除条件或选择有利种子。开发点源噪声/方向按语句轮换，因素并非完全独立，不将分组差异解释为单独因果效应。', '',
        'SI-SDR不约束全局增益；PESQ/STOI不是逐词保留证据。干净RMS/NMSE保留原电平，不能以较低损失或信号指标代替WER、内容保真、正式听测。此处明确禁用整句WER。统计目标标签仍是真值掩码加权的混合统计，非真实源SCM；已有监督思想的组合不认定原创。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    complete={'status':'ok','utc':now(),'rows':1404,'reference_cases':108,'SI_recomputed':1404,
        'summary_sha256':sha(OUT/'summary.json'),'decision_sha256':sha(OUT/'primary_decision.json'),
        'report_sha256':sha(report),'extension_gate_passed':decision['extension_gate_passed'],'ASR_performed':False}
    dump(OUT/'ANALYSIS_COMPLETE.json',complete,True)
    archive('analysis',report,[OUT/'ANALYSIS_COMPLETE.json',OUT/'summary.json',OUT/'primary_decision.json',OUT/'SIGNAL_COMPLETE.json',ROOT/'scripts/g31_finish.py'])


def accept():
    plan=guard();complete=meta(OUT/'ANALYSIS_COMPLETE.json');done=meta(OUT/'SIGNAL_COMPLETE.json')
    assert complete['status']==done['status']=='ok' and complete['SI_recomputed']==1404
    assert meta(OUT/'power_helper.json')['status']=='released'
    assert sha(OUT/'signals.jsonl')==done['signals_sha256'] and sha(OUT/'summary.json')==complete['summary_sha256']
    assert sha(OUT/'primary_decision.json')==complete['decision_sha256']
    rows=read(OUT/'signals.jsonl');assert len(rows)==len({(r['system'],r['sample_id']) for r in rows})==1404
    stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    audit={'status':'ok','utc':now(),'rows':1404,'development_cases':108,'old_development_sources':12,'speakers':2,
        'ASR_performed':False,'training_performed':False,'extension_gate_passed':complete['extension_gate_passed'],
        'analysis_sha256':sha(OUT/'ANALYSIS_COMPLETE.json'),'stages':stages,'power_released':True}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G31_完整开发评价验收.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G31 完整开发评价验收\n\n1404条评分覆盖108个固定开发条件，完整波形/源数据/代码/参数与阶段快照核对，1404条SI-SDR及干净RMS/NMSE已独立重算。保持唤醒释放，无新训练、ASR或下载。主门槛在训练前冻结，所有组及不利结果保留。\n\n[开发对照报告](G31_同预算目标开发信号与保真对照报告.md)中的门槛只决定是否扩展研究，不证明统计显著、原创、人声内容保留或真实阵列泛化。\n')
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',OUT/'power_helper.json',ROOT/'scripts/g31_finish.py'])
    status(stage='accepted',extension_gate_passed=complete['extension_gate_passed'])
