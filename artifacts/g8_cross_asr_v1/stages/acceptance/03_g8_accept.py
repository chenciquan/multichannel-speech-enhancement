#!/usr/bin/env python3
"""Archive post-run full-suite verification without rewriting frozen G8 report."""
import json,re
from pathlib import Path
from g8_common import ROOT,OUT,now,sha,dump,archive


def main():
    log=OUT/'verification/unit_tests.log';text=log.read_text();match=re.search(r'Ran (\d+) tests',text)
    checks={'G8_audit_ok':json.loads((OUT/'completion_audit.json').read_text())['status']=='ok',
      'full70_tests_passed':bool(match and int(match.group(1))==70 and '\nOK' in text),
      'temporary_awake_state_released':json.loads((OUT/'power_helper.json').read_text())['status']=='released'}
    total=0;valid=True
    for run in ('g7_unseen_noise_v1','g8_cross_asr_v1'):
        for path in (ROOT/'artifacts'/run/'stages').glob('*/archive_manifest.json'):
            for entry in json.loads(path.read_text())['files']:
                valid &= sha(path.parent/entry['snapshot'])==entry['sha256'];total+=1
    checks['all_prior_G7_G8_snapshots_match']=bool(valid)
    verdict={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'snapshot_files_checked':total,'completed_utc':now()}
    if verdict['status']!='ok':raise RuntimeError(json.dumps(verdict))
    report=ROOT/'docs/G8_验收补充记录.md';report.write_text('# G8 验收补充记录\n\nG8双识别器报告与审计已保存，原报告/快照不重写。新增完整测试套件70项通过；G7/G8此前所有快照哈希核验通过，临时保持唤醒已释放。\n\n24个依赖条件中23个效应方向一致，只是探索性描述，不是正确率或显著性比例。−40°改善、20°/70°代价大致被第二识别器复核；不存在全条件赢家，也未确认稳定声学吞字。\n\n原报告“干净参考的识别下限”应理解为干净参考识别基准，不是数学上有保证的WER下界。含噪音频偶尔可能因识别器偏差或语言规律获得更低WER；参考缩放/重复不能增加独立源句数量。G5/G7仍合计只有同12句/3人，参考不同缩放的统计不是新语料。\n\nG8的source-cluster bootstrap固定种子20261112，旧G5/G7区间各自沿用其冻结程序；不要求不同bootstrap实现的有限次Monte Carlo端点完全相同。点估计使用相同微平均定义，不能按哪个区间更有利来选报告。\n\n下一阶段加入成熟的估计掩码Souden MVDR对照。若增加真DOA/真参考诊断，必须单独标为oracle，不能当可部署的公平主基线。本阶段不再训练、不改旧证据。\n\n[双识别器报告](G8_双识别器交叉核查报告.md)、[完整测试日志](../artifacts/g8_cross_asr_v1/verification/unit_tests.log)。\n\nUTC：'+now()+'\n',encoding='utf-8')
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',verdict,True);archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',log,Path(__file__)])
    print(json.dumps(verdict))


if __name__=='__main__':main()
