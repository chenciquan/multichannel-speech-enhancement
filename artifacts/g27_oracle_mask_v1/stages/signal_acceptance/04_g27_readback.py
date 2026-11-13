"""Accept and compare the frozen oracle-mask signal diagnostic."""
from collections import defaultdict
import numpy as np
import soundfile as sf
from g27_signal import ROOT,OUT,SOURCE,sha,read,dump,now,meta,archive


def main():
    plan=meta(OUT/'frozen_plan.json');complete=meta(OUT/'SIGNAL_COMPLETE.json')
    assert complete['status']=='ok' and complete['rows']==864 and complete['cases']==432
    assert meta(OUT/'power_helper.json')['status']=='released'
    assert sha(OUT/'signals.jsonl')==complete['signals_sha256'] and sha(OUT/'summary.json')==complete['summary_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h,n
    stage_hashes={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:
            assert sha(stage/r['snapshot'])==sha(ROOT/r['source'])==r['sha256']
        stage_hashes[stage.name]=sha(stage/'archive_manifest.json')
    manifest={r['sample_id']:r for r in read(ROOT/'data/g22_v1/manifest.jsonl') if r['condition']!='clean'}
    rows=read(OUT/'signals.jsonl');assert len(rows)==len({(r['system'],r['sample_id']) for r in rows})==864
    groups=defaultdict(list);rt0=defaultdict(dict)
    for r in rows:
        assert r['sample_id'] in manifest and r['oracle_clean_information'] and not r['deployable']
        for k,v in manifest[r['sample_id']].items():assert r[k]==v
        assert sha(r['output_path'])==r['output_sha256']
        method=r['system'].split('_')[-1];groups[r['room_id'],r['requested_rt60'],method].append(r)
        if r['requested_rt60']==0:rt0[r['system'],r['matched_room_pair']][r['room_id']]=r
    assert len(rt0)==144
    file_equal=0
    for r in rt0.values():
        a,b=r['reference'],r['new']
        file_equal+=int(a['output_sha256']==b['output_sha256'])
        x,sr=sf.read(a['output_path'],dtype='float32');y,rr=sf.read(b['output_path'],dtype='float32')
        assert sr==rr==16000 and np.array_equal(x,y)
    old={ (r['system'],r['sample_id']):r for r in read(SOURCE/'evaluation/combined_signals.jsonl') }
    compare=[];details=[]
    for (room,rt,method),rs in sorted(groups.items()):
        assert len(rs)==72
        a=[old['mcnet_mask_'+method,r['sample_id']] for r in rs]
        b=[old['truth_covariance_'+method,r['sample_id']] for r in rs]
        means={name:{m:float(np.mean([r['metrics'][m] for r in records])) for m in ('si_sdr_db','pesq_wb','stoi')} for name,records in (('McNet_mask',a),('truth_ref_power_mask',rs),('truth_covariance',b))}
        delta={name:{m:float(np.mean([r['metrics'][m]-c['metrics'][m] for r,c in zip(rs,records)])) for m in ('si_sdr_db','pesq_wb','stoi')} for name,records in (('McNet_mask',a),('truth_covariance',b))}
        compare.append({'room':room,'rt':rt,'method':method,'cases':72,'means':means,'oracle_mask_minus':delta,
            'fraction_SI_better_than_McNet_mask':float(np.mean([r['metrics']['si_sdr_db']>c['metrics']['si_sdr_db'] for r,c in zip(rs,a)])),
            'covariance_shape':{k:float(np.mean([r['mask_diagnostics'][k] for r in rs])) for k in rs[0]['mask_diagnostics']}})
        for r,c,t in zip(rs,a,b):details.append({'sample_id':r['sample_id'],'room':room,'rt':rt,'method':method,
            'delta_SI_vs_McNet':r['metrics']['si_sdr_db']-c['metrics']['si_sdr_db'],
            'delta_SI_vs_truth_covariance':r['metrics']['si_sdr_db']-t['metrics']['si_sdr_db'],
            'speech_NMSE':r['components']['speech_error_nmse'],'noise_reduction_db':r['components']['noise_reduction_db']})
    dump(OUT/'paired_comparison.json',compare,True);dump(OUT/'paired_deltas.json',details,True)
    report=ROOT/'docs/G27_真值掩码信号验收与瓶颈读回.md'
    lines=['# G27 真值掩码信号验收与瓶颈读回','','864条输出和原始数据、代码、阶段快照、波形哈希通过检查；144对无反射输出逐位一致。6条件12输出的预检与未修改滤波函数逐位一致。临时保持唤醒已释放，无新增训练或识别。', '',
           '|房间|请求RT|滤波|例数|McNet掩码SI-SDR|真值功率比掩码SI-SDR|真值协方差SI-SDR|真值掩码SI优于McNet比例|','|---|---:|---|---:|---:|---:|---:|---:|']
    for r in compare:
        m=r['means'];lines.append(f"|{r['room']}|{r['rt']}|{r['method']}|72|{m['McNet_mask']['si_sdr_db']:.3f}|{m['truth_ref_power_mask']['si_sdr_db']:.3f}|{m['truth_covariance']['si_sdr_db']:.3f}|{r['fraction_SI_better_than_McNet_mask']:.1%}|")
    lines+=['','|房间|请求RT|语音协方差形状误差：McNet / 真值掩码|噪声协方差形状误差：McNet / 真值掩码|','|---|---:|---:|---:|']
    for r in compare:
        if r['method']!='mvdr':continue
        c=r['covariance_shape'];lines.append(f"|{r['room']}|{r['rt']}|{c['McNet_mask_speech_shape_error']:.5f} / {c['true_mask_speech_shape_error']:.5f}|{c['McNet_mask_noise_shape_error']:.5f} / {c['true_mask_noise_shape_error']:.5f}|")
    lines+=['', '## 能说明与不能说明的事情', '',
        '这些是同四句、同噪声、同场景的配对结果，条件不是独立语句。形状误差先逐频率按真值能量加权，再逐例平均；没有对最有利频率或系统挑选。该量不是MISD，也未证明与WER单调相关。', '',
        '若真值功率比掩码改善输出，它表明这一具体真值掩码在现有统计估计/求解流程中有可用空间，尚不证明McNet或LoRA可学到它。若它仍不如真值协方差，说明这一功率比构造尚未实现真值统计的效果；不能由一个掩码失败推出所有标量掩码都无法实现。两类诊断均不是理论上界。', '',
        'MVDR和MWF除了求解器，还保留不同的协方差归一化约定；不做单一求解器因果结论。真值功率比基于全混响参考，噪声包括量化差额，不能将结果解读成已解决去混响。', '',
        '## 下一阶段门槛', '',
        '继续比较固定两个识别器的原始与电平控制评分，分别复用G22/G26对应条件的原有成绩与精确输入缓存，保留两类参考评分；新波形必须实际解码或有严格同PCM/同模型/同预处理缓存。新分数不能用于调整本掩码或宣称新方法。先完成这项诊断的归档，再决定是否做同预算的混响训练与成熟协方差监督目标对照。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    audit={'status':'ok','utc':now(),'rows':864,'cases':432,'RT0_exact_pairs':144,'RT0_same_file_hash_pairs':file_equal,'previous_stages':stage_hashes,
        'signal_sha256':sha(OUT/'signals.jsonl'),'paired_comparison_sha256':sha(OUT/'paired_comparison.json'),
        'report_sha256':sha(report),'power_released':True,'ASR_performed':False,'training_performed':False}
    dump(OUT/'SIGNAL_ACCEPTANCE_AUDIT.json',audit,True)
    archive('signal_acceptance',report,[OUT/'SIGNAL_ACCEPTANCE_AUDIT.json',OUT/'paired_comparison.json',OUT/'paired_deltas.json',ROOT/'scripts/g27_readback.py',OUT/'power_helper.json'])
    print(json_summary(compare))


def json_summary(rows):
    return [{k:r[k] for k in ('room','rt','method','means')} for r in rows]


if __name__=='__main__':main()
