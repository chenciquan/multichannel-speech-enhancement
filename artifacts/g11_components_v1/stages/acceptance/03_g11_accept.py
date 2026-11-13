#!/usr/bin/env python3
"""Close G11 after full-suite verification; retain descriptive component readback."""
import json
from pathlib import Path
import numpy as np
from g11_common import ROOT,OUT,now,sha,dump,archive
from mcse.asr_scoring import corpus_score


def main():
    result=json.loads((OUT/'analysis/summary.json').read_text());log=OUT/'verification/unit_tests.log';text=log.read_text()
    checks={'G11_audit_ok':json.loads((OUT/'completion_audit.json').read_text())['status']=='ok',
      'full86_tests_passed':'Ran 86 tests' in text and '\nOK' in text,
      'temporary_awake_released':json.loads((OUT/'power_helper.json').read_text())['status']=='released'}
    valid=True;n=0
    for p in (OUT/'stages').glob('*/archive_manifest.json'):
        for r in json.loads(p.read_text())['files']:valid &= sha(p.parent/r['snapshot'])==r['sha256'];n+=1
    checks['all_prior_G11_snapshots_match']=bool(valid)
    verdict={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'snapshot_files_checked':n,'completed_utc':now()}
    if verdict['status']!='ok':raise RuntimeError(json.dumps(verdict))
    lines=['# G11 验收与分量诊断简明读回','','完整4,032条条件分解、全部三种子、源/旧输出/权重哈希及线性检查通过；86项完整测试通过，阶段快照核验与临时保持唤醒释放已完成。未训练或重跑ASR，原报告不改。',
      '', '下表合并四种录音，仅是便于阅读的描述均值，不替代分噪声/场型/说话人结果。目标失真NMSE小者好；目标增益更接近0仅代表总能量较接近，不能保证细节；噪声衰减大者好。所有指标针对冻结MVDR条件滤波器，不能据此断言非线性直接输出删词。','','|前端|方向|目标NMSE|目标增益dB|噪声衰减dB|分量SNR dB|旧Whisper/CTC WER|','|---|---:|---:|---:|---:|---:|---:|']
    for group in ('original','fixed','rotated'):
        for t in (-40,20,70):
            v=[r[group][str(t)] for r in result['cells'].values()];m={k:float(np.mean([r['mean'][k] for r in v])) for k in ('speech_error_nmse','target_gain_db','noise_reduction_db','component_output_snr_db')}
            wer={engine:corpus_score(r[engine] for r in v)['wer'] for engine in ('Whisper','CTC')}
            lines.append(f'|{group}|{t}|{m["speech_error_nmse"]:.3f}|{m["target_gain_db"]:.2f}|{m["noise_reduction_db"]:.2f}|{m["component_output_snr_db"]:.2f}|{wer["Whisper"]:.1%} / {wer["CTC"]:.1%}|')
    lines+=['','这些分量属于可重建的模拟场。若语音失真与噪声压低同时变化，只能说明这批条件下的滤波器权衡，不能据同向WER证明人声词语完整。G10的新语音已被查看，后续调参与验证必须区分开发集和新确认条件。',
      '', '下一项高价值风险核查是加入成熟房间响应生成器的混响压力测试，而非先增加恢复模块。正式开展前须核实依赖/许可与Python资源、固定含无混响对照的协议、保留完整语句和混响尾部，并明确采用参考麦克风混响语音作为降噪目标。当前只完成公开文档核查，未安装或运行该候选。',
      '', '[G11主报告](G11_固定MVDR目标语音与残余噪声诊断报告.md)、[机器汇总](../artifacts/g11_components_v1/analysis/summary.json)、[完整测试](../artifacts/g11_components_v1/verification/unit_tests.log)。','', 'UTC：'+now()]
    report=ROOT/'docs/G11_验收与分量诊断简明读回.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',verdict,True)
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',log,Path(__file__)]);print(json.dumps(verdict))


if __name__=='__main__':main()
