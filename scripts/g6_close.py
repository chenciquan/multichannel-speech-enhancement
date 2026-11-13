#!/usr/bin/env python3
"""Read completed G6 evidence and archive final stage acceptance."""
import json,hashlib,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g6_local_runtime_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=json.loads((OUT/'prototype/qa_results.json').read_text());bench=json.loads((OUT/'benchmark/raw_timings.json').read_text());checks={
      'CLI_QA_passed':result['status']=='ok' and all(r['passed'] for r in result['results']) and result['real_overwrite_refusal'],
      'ten_benchmark_configurations':len(bench['measurements'])==10,
      'twenty_valid_calls_per_configuration':all(len(r['calls_seconds'])==20 and all(np.isfinite(x) and x>0 for x in r['calls_seconds']) for r in bench['measurements']),
      'full62_tests_passed':'Ran 62 tests' in (OUT/'prototype/unit_tests.log').read_text() and '\nOK' in (OUT/'prototype/unit_tests.log').read_text(),
      'focused9_contract_tests_passed':'Ran 9 tests' in (OUT/'verification/unit_tests.log').read_text() and '\nOK' in (OUT/'verification/unit_tests.log').read_text()}
    snapshot_ok=True
    for mp in (OUT/'stages').glob('*/archive_manifest.json'):
        for r in json.loads(mp.read_text())['files']:snapshot_ok &= sha(mp.parent/r['snapshot'])==r['sha256']
    checks['prior_stage_snapshot_hashes_match']=bool(snapshot_ok)
    for r in result['results']:
        p=ROOT/r['report'];meta=json.loads(p.read_text());wave=Path(meta['output']['path']);info=sf.info(wave)
        checks[r['system']+'_'+r['device']+'_saved_output']=sha(wave)==r['output_sha256'] and info.samplerate==16000 and info.channels==1 and info.frames==meta['input']['frames']
    archive_time=json.loads((OUT/'stages/benchmark/archive_manifest.json').read_text())['saved_utc'];checks['benchmark_archived_before_CLI_QA']=datetime.fromisoformat(archive_time)<datetime.fromisoformat(result['completed_utc'])
    verdict={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'closed_utc':datetime.now(timezone.utc).isoformat(),'no_new_training':True,'no_new_quality_claim':True}
    if verdict['status']!='ok':raise RuntimeError(json.dumps(verdict))
    report=ROOT/'docs/G6_本地推理阶段验收.md';report.write_text('# G6 本地推理阶段验收\n\nG6A资源与G6B独立CLI均已完成并分阶段归档，原G3—G5结果不改。\n\n- FP32、batch1下，1秒纯GPU计算约31—35ms，3秒约83—94ms，RTF约0.03；峰值分配显存3秒约1.37GiB，Torch缓存约2.7GiB，系统开销另计。这不是流式低延迟结论。\n- 五个GPU CLI结果与G5保存波形逐样本一致，最大差0；CPU原基线最大差约2.79e-4，小于预定1e-3兼容界限。\n- 独立进程墙钟约25—29秒，包含库导入、初始化、核验、加载和IO；不得用纯计算数值代替使用者等待时间。\n- 全套62项单测通过；随后接口9项定向复核通过，其中2项新增，覆盖未知系统与篡改权重在加载前拒绝。两日志有重叠，不相加称为71项。\n- 实际二次CLI验证拒绝覆盖；音频/报告分路径、形状、采样率、有限值、长度、注册清单与模型哈希均检查。\n\n入口：`scripts/enhance_registered.py --list-systems`。推理需当前WSL环境`.venv/bin/python`，六通道16kHz、最多4秒；不能把单麦复制六份，长文件/流式尚未实现。默认原McNet，也可选择注册G3/G5系统。示例系统不是全场景赢家。\n\n[资源报告](G6A_本地推理资源实测.md)、[命令与逐次QA](G6B_本地推理命令与一致性验证.md)。原始时间、逐次CLI输出和审计位于`artifacts/g6_local_runtime_v1/`。本阶段不产生新的质量/创新结论。下一阶段冻结模型在新噪声录音上验证G5权衡是否保持，先核实来源和登记协议。\n',encoding='utf-8')
    folder=OUT/'stages/acceptance';folder.mkdir(exist_ok=False);entries=[]
    for i,p in enumerate([report,OUT/'verification/unit_tests.log',ROOT/'tests/test_runtime_interface.py',Path(__file__)]):
        dst=folder/f'{i:02d}_{p.name}';shutil.copy2(p,dst);entries.append({'source':str(p.relative_to(ROOT)),'snapshot':dst.name,'sha256':sha(p)})
    (OUT/'FINAL_COMPLETION_AUDIT.json').write_text(json.dumps(verdict,indent=2)+'\n');(folder/'archive_manifest.json').write_text(json.dumps({'saved_utc':verdict['closed_utc'],'files':entries},indent=2)+'\n');print(json.dumps({'status':verdict['status'],'report':str(report.relative_to(ROOT))}))
if __name__=='__main__':main()
