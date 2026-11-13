#!/usr/bin/env python3
"""Persist stage report and immutable evidence snapshot before downstream work."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g3_overnight_v1'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['training','signal','completion'],required=True);a=p.parse_args()
    folder=OUT/'stages'/a.stage;folder.mkdir(parents=True,exist_ok=False);files=[];references=[]
    if a.stage=='training':
        selected=json.loads((OUT/'selected_checkpoints_extended.json').read_text());results=[]
        for s in selected['systems']:results.append(json.loads((OUT/'train'/s['system']/'result.json').read_text()))
        files=[ROOT/'configs/g3_overnight_plan.json',ROOT/'configs/g3_execution_supplement.json',ROOT/'configs/g3_factorial_extension.json',
               OUT/'selected_checkpoints.json',OUT/'selected_checkpoints_extended.json',OUT/'fp32_amendment.json',ROOT/'data/g3_v1/dataset/frozen_protocol.json',
               ROOT/'requirements-adaptation.lock.txt',ROOT/'scripts/g3_train.py',ROOT/'src/mcse/adaptation.py',ROOT/'scripts/g3_build.py']
        for s in selected['systems']:
            path=ROOT/s['path'];references.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'role':'selected_adapter_checkpoint'})
        for name in ('train_x.npy','train_y.npy','clean_x.npy','train_catalog.json'):
            path=ROOT/'data/g3_v1/dataset'/name;references.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'role':'shared_training_pool'})
        lines=['# G3A 训练阶段报告','','训练阶段已结束，检查点全部按验证集选定，尚未打开最终测试。原主对照为普通适配与完整方案；根据首种子验证结果补充登记两个消融的种子复验，构成三种子2×2组件对照。','','|训练系统|有效更新|选定步数|验证SI-SDR dB|验证STOI|原权重冻结|','|---|---:|---:|---:|---:|---|']
        for r in results:
            s=r['selected'];lines.append(f'|{r["variant"]}_s{r["seed"]}|{r["completed_steps"]}|{s["step"]}|{s["si_sdr_db"]:.3f}|{s["stoi"]:.4f}|{r["base_unchanged"]}|')
        lines+=['',f'共{len(results)}次正式训练，合计{sum(r["completed_steps"] for r in results)}次有效更新。每次均为3,584个可训练参数、同一数据池、3,000次更新与六次检查点选择机会。干净约束组额外进行600次干净输入计算，并非等总算力。',
                '',f'最大Torch实际分配显存{max(r["peak_allocated_bytes"] for r in results)/2**30:.3f}GiB，最大保留缓存{max(r["peak_reserved_bytes"] for r in results)/2**30:.3f}GiB。',
                '', '首次AMP普通适配出现非有限梯度，失败产物保留；正式对照全部以FP32从原权重重启。55项测试通过；逐次训练记录包含冻结基线前后摘要。',
                '', '本阶段的验证数值不能当成独立测试结论，也不能证明人声误删或创新成立。下一步在选定权重固定后，对360个独立保留混合条件及30句干净输入统一评估。']
        report=ROOT/'docs/G3A_训练阶段报告.md'
        files += [OUT/'train'/s['system']/'result.json' for s in selected['systems']]
    elif a.stage=='signal':
        rows=[json.loads(s) for s in (OUT/'evaluation/signals.jsonl').read_text().splitlines()];systems=sorted({r['system'] for r in rows})
        files=[OUT/'evaluation/SIGNAL_COMPLETE.json',OUT/'evaluation/protocol.json',OUT/'evaluation/systems.json',OUT/'evaluation/metric_backends.json',ROOT/'scripts/g3_evaluate.py']
        references.append({'path':str((OUT/'evaluation/signals.jsonl').relative_to(ROOT)),'sha256':sha(OUT/'evaluation/signals.jsonl'),'role':'all_signal_scores_and_waveform_hashes'})
        lines=['# G3B 信号指标阶段报告','','选定权重的信号评估已结束，识别评估尚未开始。保存后的FLOAT WAV是计分来源，无逐句峰值归一化、重采样或自动时移拟合。','','|系统|有效/总例数|SI-SDR dB|PESQ|STOI|','|---|---:|---:|---:|---:|']
        for system in systems:
            items=[r for r in rows if r['system']==system and r['split']=='test'];good=[r for r in items if r['status']=='ok']
            means=[sum(r['metrics'][k] for r in good)/len(good) if good else float('nan') for k in ('si_sdr_db','pesq_wb','stoi')]
            lines.append(f'|{system}|{len(good)}/{len(items)}|{means[0]:.3f}|{means[1]:.3f}|{means[2]:.4f}|')
        lines+=['','每系统另有30句干净输入对照，完整逐例数据保留于`evaluation/signals.jsonl`。360个混合条件来自30句、4位测试说话人，不能当作360句独立语音。',
                '', 'SI-SDR衡量波形误差，PESQ估计感知质量，STOI估计可懂度；均不能单独证明词语、音素完整。下一阶段使用固定Whisper tiny.en计算WER，不用识别结果返回调参。']
        report=ROOT/'docs/G3B_信号指标阶段报告.md'
    else:
        report=ROOT/'docs/G3_五小时实验验收报告.md';assert report.is_file()
        files=[OUT/'completion_audit.json',OUT/'selected_checkpoints_extended.json',OUT/'evaluation/summary.json',OUT/'evaluation/asr_provenance.json',OUT/'review/selection.json',OUT/'review/index.html',
               ROOT/'scripts/g3_summarize.py',ROOT/'scripts/g3_audit.py',ROOT/'scripts/g3_report.py']
        for name in ('asr_scores.jsonl','asr_cache.jsonl','signals.jsonl'):
            path=OUT/'evaluation'/name;references.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'role':'completed_evaluation_evidence'})
    if a.stage!='completion':
        if report.exists():raise RuntimeError('Refuse overwriting archived stage report')
        report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files.append(report)
    entries=[]
    for i,path in enumerate(files):
        target=folder/f'{i:02d}_{path.name}';shutil.copy2(path,target);entries.append({'source':str(path.relative_to(ROOT)),'snapshot':target.name,'sha256':sha(path)})
    manifest={'stage':a.stage,'saved_at_utc':datetime.now(timezone.utc).isoformat(),'report':str(report.relative_to(ROOT)),'files':entries,'referenced_artifacts':references,
              'large_audio_checkpoints_copied':False,'large_outputs':'Kept in original per-run directories; hashes and exact checkpoint paths in manifests, no evidence deleted.'}
    (folder/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'stage':a.stage,'status':'archived','report':manifest['report']}))
if __name__=='__main__':main()
