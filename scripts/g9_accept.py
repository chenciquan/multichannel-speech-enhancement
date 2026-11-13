#!/usr/bin/env python3
"""Archive full-suite verification and descriptive G9 readback before G10."""
import json
from pathlib import Path
from g9_common import ROOT,OUT,now,sha,dump,archive


def main():
    result=json.loads((OUT/'evaluation/summary.json').read_text());log=OUT/'verification/unit_tests.log';text=log.read_text()
    checks={'G9_audit_ok':json.loads((OUT/'completion_audit.json').read_text())['status']=='ok',
      'full75_tests_passed':'Ran 75 tests' in text and '\nOK' in text,
      'temporary_awake_state_released':json.loads((OUT/'power_helper.json').read_text())['status']=='released'}
    n=0;valid=True
    for p in (OUT/'stages').glob('*/archive_manifest.json'):
        for entry in json.loads(p.read_text())['files']:valid &= sha(p.parent/entry['snapshot'])==entry['sha256'];n+=1
    checks['all_G9_prior_snapshots_match']=bool(valid)
    verdict={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'snapshot_files_checked':n,'completed_utc':now()}
    if verdict['status']!='ok':raise RuntimeError(json.dumps(verdict))
    lines=['# G9 验收与后续研究决策','','G9报告、双ASR与完整审计均已归档；75项完整测试通过，所有阶段快照哈希及临时保持唤醒释放核对通过。原报告不重写。',
       '', '下表读回成熟估计掩码MVDR减原McNet的点估计：SI-SDR正值好，WER百分点负值好。12个分组复用同12句/3人，不作独立显著性推断，也不把oracle结果当部署承诺。','','|数据/噪声|方向|ΔSI-SDR dB|ΔWhisper WER百分点|ΔCTC WER百分点|','|---|---:|---:|---:|---:|']
    for ds,noises in result['cells'].items():
        for noise,systems in noises.items():
            for t in (-40,20,70):
                a=systems['mcnet_mask_mvdr'][str(t)];b=systems['mcnet'][str(t)]
                lines.append(f'|{ds}/{noise}|{t}|{a["metrics"]["si_sdr_db"]-b["metrics"]["si_sdr_db"]:+.2f}|{(a["Whisper"]["wer"]-b["Whisper"]["wer"])*100:+.1f}|{(a["CTC"]["wer"]-b["CTC"]["wer"])*100:+.1f}|')
    lines+=['','下一步优先补充独立新说话人，并冻结“直接波形输出/标准MVDR输出×固定/旋转适配”的对照。这样同时针对样本重复和成熟对照不足，暂不追加优化器训练或自称新恢复模块。需在取数与新输出前登记具体系统/条件、源排除规则和资源上限。',
       '', 'G10只考虑现有无干净约束分支的固定/旋转全部三种子，是基于已查看结果收窄机制问题；不得宣称因此证明干净约束普遍无效。新说话人验证该收窄后的假设，仍需面对四位说话人、模拟阵列和既有噪声录音的限制。',
       '', '辅助mask_mvdr函数只知道“带噪输入+所提供估计”，不能自行判断估计是不是oracle；本轮调用方已剔除函数通用提示字段，并按实际来源保存oracle标记。主/诊断的真实信息来源以冻结调用流程和逐例provenance为准。',
       '', '[G9总报告](G9_成熟波束形成对照与机制诊断报告.md)、[完整测试日志](../artifacts/g9_beamforming_v1/verification/unit_tests.log)。','', 'UTC：'+now()]
    report=ROOT/'docs/G9_验收与后续研究决策.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',verdict,True)
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',log,ROOT/'configs/g10_fresh_speaker_candidate.json',Path(__file__)]);print(json.dumps(verdict))


if __name__=='__main__':main()
