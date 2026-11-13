#!/usr/bin/env python3
import json
from collections import Counter
from g16_common import ROOT,OUT,now,sha,dump,read,archive


def main():
    assert (OUT/'stages/runtime/archive_manifest.json').exists();plan=json.loads((OUT/'frozen_plan.json').read_text());g14=ROOT/'artifacts/g14_corrected_geometry_v1'
    assert sha(ROOT/'data/g14_v1/manifest.jsonl')==plan['G14_manifest_sha256'] and sha(g14/'evaluation/combined_signals.jsonl')==plan['G14_signals_sha256']
    cases=[r for r in read(ROOT/'data/g14_v1/manifest.jsonl') if r['geometry_protocol']=='rigid_rotation'];rows=[r for r in read(g14/'evaluation/combined_signals.jsonl') if r['geometry_protocol']=='rigid_rotation'];assert len(cases)==168 and len(rows)==2520
    assert len({r['system'] for r in rows})==15 and all(v==168 for v in Counter(r['system'] for r in rows).values())
    files={}
    for r in rows:
        for p,h in ((r['noisy_path'],r['input_sha256']),(r['reference_path'],r['reference_sha256']),(r['output_path'],r['output_sha256'])):
            if p in files:assert files[p]==h
            files[p]=h
    assert all(sha(p)==h for p,h in files.items())
    for name,items in [('case_manifest.jsonl',cases),('baseline_signals.jsonl',rows)]:
        with (OUT/name).open('x') as f:
            for r in items:f.write(json.dumps(r)+'\n')
    dump(OUT/'PREPARATION_COMPLETE.json',{'status':'ok','completed_utc':now(),'cases':168,'reused_baseline_rows':2520,'manifest_sha256':sha(OUT/'case_manifest.jsonl'),'baseline_rows_sha256':sha(OUT/'baseline_signals.jsonl'),'all_baseline_audio_hashes_match':True},True)
    report=ROOT/'docs/G16B_正确几何开发诊断输入与基线复用.md';report.write_text('# G16B 正确几何开发诊断输入与基线复用\n\nG15训练及新注册接口QA归档后，复用G14全部rigid_rotation条件：8旧句/4人，144带噪+24干净=168条件；两条旧噪声、0dB、三目标角及三空间场。没有重新选择有利句子/方位/噪声，没有新混合或语料。\n\n15个既有系统2520条保存输出与输入/参考SHA全核验，原文件不复制覆盖。三个新刚性训练模型和对应MVDR新增1008条，总21系统3528条/后端。旧counter模型继续按其真实训练含义标记。所有模型均为3000步、三种子，成熟MVDR规则固定。\n\n这是已查看语句的开发诊断，不能作为独立最终验证。主要比较新刚性减旧反向、减固定，并查看20°代价、全部方向/噪声/种子及干净控制；两个ASR不替代人工词尾标签。新语音确认必须另行冻结协议。\n\n本报告归档后才生成新模型直接输出；随后MVDR、Whisper、CTC逐阶段归档。\n\nUTC：'+now()+'\n',encoding='utf-8')
    archive('preparation',report,[OUT/'frozen_plan.json',OUT/'PIPELINE_READY.json',OUT/'systems.json',OUT/'PREPARATION_COMPLETE.json',OUT/'case_manifest.jsonl',OUT/'baseline_signals.jsonl']+[ROOT/n for n in json.loads((OUT/'PIPELINE_READY.json').read_text())['files']])


if __name__=='__main__':main()
