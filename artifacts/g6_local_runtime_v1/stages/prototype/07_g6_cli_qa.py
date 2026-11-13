#!/usr/bin/env python3
"""Actual CLI parity against saved G5 waveforms, plus archived unit-test result."""
import json,subprocess,sys,time,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.runtime import sha256
OUT=ROOT/'artifacts/g6_local_runtime_v1';PLAN=json.loads((ROOT/'configs/g6_cli_plan.json').read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    assert (OUT/'stages/benchmark/archive_manifest.json').exists();folder=OUT/'prototype';folder.mkdir(exist_ok=False)
    with (folder/'unit_tests.log').open('x') as log:p=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    if p.returncode:raise RuntimeError('Unit tests failed; do not continue silently')
    cases=[json.loads(s) for s in (ROOT/'data/g5_v1/confirmation/manifest.jsonl').read_text().splitlines()];case=sorted([r for r in cases if r['condition']!='clean'],key=lambda r:r['sample_id'])[0]
    prior={(r['system'],r['sample_id']):r for r in [json.loads(s) for s in (ROOT/'artifacts/g5_rotation_v1/evaluation/signals.jsonl').read_text().splitlines()]}
    results=[]
    for index,(system,device) in enumerate([(s,'cuda') for s in PLAN['primary_QA_systems']]+[('mcnet','cpu')]):
        name=system.replace('/','_')+'_'+device;output=folder/(name+'.wav');report=folder/(name+'.json');start=time.perf_counter()
        with (folder/(name+'.log')).open('x') as log:
            p=subprocess.run([sys.executable,str(ROOT/'scripts/enhance_registered.py'),'--input',case['noisy_path'],'--output',str(output),'--report',str(report),'--system',system,'--device',device],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=PLAN['per_subprocess_timeout_seconds'])
        if p.returncode:raise RuntimeError(f'CLI failed {system}/{device}')
        wall=time.perf_counter()-start;y,sr=sf.read(output,dtype='float32');base_system=system.split('/',1)[-1];expected,_=sf.read(prior[(base_system,case['sample_id'])]['output_path'],dtype='float32');assert y.shape==expected.shape and sr==16000
        error=float(np.max(np.abs(y-expected)));tolerance=PLAN['GPU_saved_output_maxabs_tolerance'] if device=='cuda' else .001
        record={'system':system,'device':device,'max_abs_error_against_saved_G5':error,'tolerance':tolerance,'passed':error<=tolerance,'subprocess_wall_seconds':wall,'output_sha256':sha256(output),'report':str(report.relative_to(ROOT))}
        results.append(record);dump(folder/'status.json',{'completed':index+1,'total':6,'last':record});print(json.dumps(record),flush=True)
        if not record['passed']:raise RuntimeError('Numerical parity failed; preserve evidence')
    # Verify a real CLI second invocation refuses overwrite and preserves bytes.
    target=folder/'mcnet_cuda.wav';before=sha256(target)
    p=subprocess.run([sys.executable,str(ROOT/'scripts/enhance_registered.py'),'--input',case['noisy_path'],'--output',str(target),'--report',str(folder/'must_not_exist.json')],cwd=ROOT,capture_output=True,text=True,timeout=180)
    overwrite_ok=p.returncode!=0 and sha256(target)==before and not (folder/'must_not_exist.json').exists()
    assert overwrite_ok
    result={'status':'ok','completed_utc':datetime.now(timezone.utc).isoformat(),'case_id':case['sample_id'],'input_sha256':sha256(case['noisy_path']),'tests_log':'unit_tests.log','results':results,'real_overwrite_refusal':overwrite_ok,'new_quality_claim':False}
    dump(folder/'qa_results.json',result)
    report=ROOT/'docs/G6B_本地推理命令与一致性验证.md';lines=['# G6B 本地推理命令与一致性验证','','G6A资源报告归档后开展。命令复用原McNet、原解码器和现有loralib，不另写增强模型或安装依赖；不启动服务或公开发布。','',
      '输入为六个真实麦克风通道、16kHz、512—64000样本；参考第5麦。不得把单麦复制六份。输出单通道FLOAT WAV，保留幅度与长度，不自动重采样、归一化或分块。只加载由G3/G5归档清单登记且SHA匹配的适配器；未知/篡改权重拒绝加载。','',
      '## 命令','', '在项目WSL目录中执行：','','```bash','.venv/bin/python scripts/enhance_registered.py --list-systems',
      '.venv/bin/python scripts/enhance_registered.py --input /path/to/six_channel.wav --output artifacts/my_run/enhanced.wav --report artifacts/my_run/provenance.json --system g5/rotated_without_clean_identity_s20261112 --device cuda','```','',
      '示例系统只说明调用方式，不代表全条件最佳。默认`--system mcnet`为原始基线；也可显式选择CPU。输出与报告必须是不同的新路径，已有文件会在模型加载前被拒绝。超过4秒不静默分块，较长录音尚需另立上下文/拼接验证方案。','',
      '## 实际验证','','固定选择G5非干净确认集的第一条ID，没有按结果挑选。独立CLI进程输出与已有G5保存波形比较：','','|系统|设备|最大绝对差|允许值|独立进程墙钟秒|','|---|---|---:|---:|---:|']
    for r in results:lines.append(f'|{r["system"]}|{r["device"]}|{r["max_abs_error_against_saved_G5"]:.9g}|{r["tolerance"]:g}|{r["subprocess_wall_seconds"]:.2f}|')
    lines+=['','全部通过。GPU允许最大绝对差1e-5，CPU跨后端1e-3仅作数值兼容界限；不是音质改善幅度。真实二次调用验证了拒绝覆盖且原字节不变。单测还检查错误采样率/通道/长度、非有限与溢出/复数、输出路径冲突及注册清单篡改。完整单测日志和逐次CLI日志保留。',
      '', '独立进程墙钟包含Python/库导入、GPU初始化、核验、加载、IO和计算，因此明显高于G6A纯计算耗时，不能将两者混报。命令没有自动运行ASR或质量指标，也不保证真实房间效果。原模型为离线双向结构，不能因此宣称流式部署完成。',
      '', '接口代码为`scripts/enhance_registered.py`、`src/mcse/runtime.py`；测试/QA和注册模型哈希见`artifacts/g6_local_runtime_v1/prototype/`。用户使用许可仍按既有本地研究范围，不推定公开再分发授权。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8');archive=OUT/'stages/prototype';archive.mkdir(parents=True);entries=[]
    for i,p in enumerate([report,folder/'qa_results.json',folder/'unit_tests.log',ROOT/'configs/g6_cli_plan.json',ROOT/'src/mcse/runtime.py',ROOT/'scripts/enhance_registered.py',ROOT/'tests/test_runtime_interface.py',Path(__file__)]):
        dst=archive/f'{i:02d}_{p.name}';shutil.copy2(p,dst);entries.append({'source':str(p.relative_to(ROOT)),'snapshot':dst.name,'sha256':sha256(p)})
    dump(archive/'archive_manifest.json',{'saved_utc':datetime.now(timezone.utc).isoformat(),'files':entries});dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'stage':'G6','completed_utc':datetime.now(timezone.utc).isoformat(),'report':str(report.relative_to(ROOT)),'resource_report':'docs/G6A_本地推理资源实测.md','audit':'prototype/qa_results.json'})
if __name__=='__main__':main()
