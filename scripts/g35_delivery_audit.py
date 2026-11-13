"""Read saved B/C artifacts, verify coverage/hashes/curve values; no forwards."""
import csv,hashlib,json,subprocess,time
from pathlib import Path
from collections import Counter
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g35_azimuth_scan_v1'
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def csv_write(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    start=time.monotonic();assert not (OUT/'DELIVERY_AUDIT.json').exists()
    A=json.loads((ROOT/'artifacts/g35_admission_v1/result.json').read_text());B=json.loads((OUT/'DATA_COMPLETE.json').read_text());C=json.loads((OUT/'SIGNAL_COMPLETE.json').read_text());S=json.loads((OUT/'SUMMARY_COMPLETE.json').read_text())
    assert A['status']==B['status']==C['status']==S['status']=='passed'
    plan=json.loads((OUT/'frozen_plan.json').read_text());assert sha(ROOT/'scripts/g35_scan_signals.py')==plan['script_sha256']
    inputs=read(ROOT/'data/g35_azimuth_scan_v1/manifest.jsonl');rows=read(OUT/'signals.jsonl')
    assert len(inputs)==1064 and len(rows)==8512 and A['forward_attempts']==len(A['forwards'])==14
    assert sha(OUT/'signals.jsonl')==C['signals_sha256'] and sha(ROOT/'data/g35_azimuth_scan_v1/manifest.jsonl')==B['manifest_sha256']
    assert len({(r['system'],r['sample_id']) for r in rows})==8512
    cases={r['sample_id']:r for r in inputs}
    for r in inputs:
        assert sha(r['noisy_path'])==r['input_sha256'] and sha(r['reference_path'])==r['reference_sha256']
    clean=[]
    for r in rows:
        assert r['status']=='ok' and all(v['status']=='ok' for v in r['metrics']['status'].values())
        assert sha(r['output_path'])==r['output_sha256']
        case=cases[r['sample_id']]
        for key in ['noisy_path','reference_path','input_sha256','reference_sha256','common_scale','noise_gain','target_azimuth']:
            assert r[key]==case[key]
        if r['condition']=='clean':
            s,sr=sf.read(r['reference_path'],dtype='float64');y,yr=sf.read(r['output_path'],dtype='float64')
            clean.append({'system':r['system'],'sample_id':r['sample_id'],'angle':r['target_azimuth'],
                'NMSE':float(np.sum((y-s)**2)/np.sum(s*s)),'RMS_gain':float(np.sqrt(np.sum(y*y)/np.sum(s*s)))})
    for system in plan['systems']:
        for angle in plan['angles']:
            cell=[r for r in rows if r['system']==system and r['target_azimuth']==angle]
            assert len(cell)==56 and sum(r['condition']=='clean' for r in cell)==8
    nois=[r for r in rows if r['condition']!='clean']
    def family(s):return 'fixed' if s.startswith('g5/') else 'rigid' if s.startswith('g15/') else s
    def csv_read(p):
        with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    means=csv_read(OUT/'curves/mean_19_angles.csv');profiles=csv_read(OUT/'curves/per_seed_19_angles.csv')
    assert len(means)==76 and len(profiles)==152
    max_error=0.
    for p in means:
        cell=[r for r in nois if family(r['system'])==p['family'] and r['target_azimuth']==int(p['angle'])]
        for key in ['si_sdr_db','pesq_wb','stoi']:
            error=abs(float(np.mean([r['metrics'][key] for r in cell]))-float(p[key]));max_error=max(max_error,error)
            assert error<=1e-12
    assert len(csv_read(OUT/'curves/all_paired_differences.csv'))==2736
    clean_summary=[{'system':system,'n':152,'NMSE_mean':float(np.mean([r['NMSE'] for r in clean if r['system']==system])),
        'RMS_gain_mean':float(np.mean([r['RMS_gain'] for r in clean if r['system']==system]))} for system in plan['systems']]
    csv_write(OUT/'curves/clean_fidelity_independent.csv',clean_summary)
    before=json.loads((ROOT/'artifacts/g35_admission_v1/protected_before.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in before.items())
    registry=json.loads((ROOT/'artifacts/g35_admission_v1/selected_registry.json').read_text())
    assert all(sha(ROOT/r['path'])==r['checkpoint_sha256'] for k,r in registry.items() if k!='mcnet')
    official=ROOT/'third_party/McNet/model_checkpoints/offline/epoch494_criteria18.78_sdr18.78.ckpt'
    assert sha(official)=='24b8c556cc52990e4d7db83da4909b7e8a1dc4a0c14880b98381bb62818beecf'
    total_bytes=sum(p.stat().st_size for folder in [OUT,ROOT/'data/g35_azimuth_scan_v1'] for p in folder.rglob('*') if p.is_file())
    assert total_bytes<=10*1024**3
    git=subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True)
    resources=json.loads((OUT/'resources.json').read_text())
    audit={'status':'passed','verification_seconds':time.monotonic()-start,'A_forwards':14,'C_forwards':7448,
        'inputs':1064,'rows':8512,'all_models_all_19_angles_present':True,'all_input_reference_output_hashes_verified':True,
        'curve_independent_max_error':max_error,'protected_files_unchanged':True,'all_7_weights_unchanged':True,
        'disk_bytes':total_bytes,'git_status':git,'ASR_executed':False,'training_executed':False,
        'B_seconds':B['seconds'],'C_seconds':C['seconds'],'curves_seconds':S['seconds'],
        'C_peak_allocated_bytes':max(r['peak_allocated_bytes'] or 0 for r in resources),
        'C_peak_reserved_bytes':max(r['peak_reserved_bytes'] or 0 for r in resources),'clean_fidelity':clean_summary,
        'delivery_script_sha256':sha(__file__)}
    dump(OUT/'DELIVERY_AUDIT.json',audit)
    labels={'noisy':'带噪参考麦','mcnet':'官方McNet','fixed':'单方位LoRA','rigid':'几何保持式多方位LoRA'}
    G=json.loads((ROOT/'artifacts/g35_admission_v1/geometry.json').read_text())
    lines=['# G35 准入及19角信号扫描交付报告','',
        '2026-11-13。按用户补充授权：A通过后执行B、C，优先输出19角SI-SDR、PESQ、STOI曲线。本轮全部完成，无训练、ASR、G34恢复或Git提交。',
        '', '## A 准入：实际通过及失败','',
        f"{len(A['checks'])}项记录全部通过，失败0项；七模型×两条旧G17输入=14次前向，墙时{A['elapsed_seconds']:.3f}秒。",
        f"原主干3328706参数，LoRA新增3584（总3332290）；四个低秩矩阵形状为[4,256]/[64,4]和[4,512]/[64,4]。七模型主干摘要完全一致，推理前后参数摘要不变。",
        f"点积最大误差{G['dot_max_error']:.3e}，相对角最大误差{G['relative_azimuth_max_error_deg']:.3e}°，独立TDOA误差{G['TDOA_max_error_samples']:.3e}样本，FIR所需保护{G['required_guard_samples']}≤64。容差未放宽。",
        f"A阶段Torch峰值分配{max(r['peak_allocated_bytes'] for r in A['models'])/1024**3:.3f} GiB，峰值保留{max(r['peak_reserved_bytes'] for r in A['models'])/1024**3:.3f} GiB。FP32、TF32两许可关闭；单通道同长、有限、FLOAT保存回读均通过。",
        '', '[A完整检查表](G35A_几何与模型准入报告.md)、[参数/来源/逐次耗时及显存](../artifacts/g35_admission_v1/result.json)。A报告是在A边界记载未执行B/C，本交付记录用户追加授权后的B/C完成状态，保留原报告。',
        '', '## B/C 完成范围','',
        f"B：1064输入（912带噪、152干净），56配对组，耗时{B['seconds']:.2f}秒；所有组自然参考一致，未复制/替换通道，保存后活动SNR误差全部≤0.002dB。",
        f"C：7448增强输出、8512行信号指标，耗时{C['seconds']:.2f}秒；全部八系统、19点、三个种子完整。绘图{S['seconds']:.2f}秒。C峰值分配/保留{audit['C_peak_allocated_bytes']/1024**3:.3f}/{audit['C_peak_reserved_bytes']/1024**3:.3f} GiB。",
        '', '![19角三项曲线](../artifacts/g35_azimuth_scan_v1/curves/three_metrics_19_angles.png)',
        '', '![LoRA配对差值](../artifacts/g35_azimuth_scan_v1/curves/paired_differences_19_angles.png)',
        '', '|系统|指标|19角平均|最差值/角|最好值/角|极差|', '|---|---|---:|---|---|---:|']
    for r in S['summary']:
        lines.append(f"|{labels[r['family']]}|{r['metric']}|{r['mean_19']:.4f}|{r['worst']:.4f} / {r['worst_angles']}°|{r['best']:.4f} / {r['best_angles']}°|{r['range']:.4f}|")
    lines+=['', '## 解释边界与数据完整性','',
        '这是旧G17八句/四人的开发画像，不是新的独立最终确认，也不是连续角度、360°、三维或真实阵列证明。不同指标可有不同排序，不根据曲线自动改训练方位/采样/模型。区间内12点、外7点，完整分组和所有种子均保留。',
        f"独立重算平均曲线与CSV最大差{max_error:.3e}；输入、目标、全部输出SHA均复核。历史关键源码/协议和七份权重未变；新数据/产物约{total_bytes/1024**3:.3f}GiB，未超10GiB。没有用有限检查或平均曲线代替全覆盖验证。",
        '[19角数据与主报告](G35C_19角三项信号指标与方向权衡报告.md)、[完整交付核验](../artifacts/g35_azimuth_scan_v1/DELIVERY_AUDIT.json)、[干净NMSE/RMS保真](../artifacts/g35_azimuth_scan_v1/curves/clean_fidelity_independent.csv)。',
        '', '## 安全保存当前版本建议（未执行）','',
        '根项目仍无首次Git提交。先经批准在另一个日期命名目录只追加备份，明确路径/容量/复制范围；源码、配置、文档与数据、权重、日志分别生成SHA清单，保留根Git及官方子仓库来源。随后经批准按明确清单建立首次源码提交；不要git add -f纳入全部数据/环境，也不要以提交替代被忽略产物备份。此次没有提交、推送、移动、删除或覆盖历史成果。',
        '', '当前授权任务结束即停止。识别评分、训练候选和最终独立确认仍未执行，须另行批准。']
    report=ROOT/'docs/G35_准入及19角信号扫描交付报告.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
