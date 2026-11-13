"""Independent integrity gate after all matched-room signal phases end."""
import json
from g22_common import ROOT,OUT,BASE,sha,read,dump,now,archive,metadata,guard,status


def main():
    ready=guard();ev=OUT/'evaluation';done=metadata(ev/'SIGNAL_COMPLETE.json')
    assert done['status']=='ok' and done['rows']==16488 and done['RT0_pairs']==2748
    power=metadata(OUT/'signal_power_helper.json');assert power['status']=='released'
    assert sha(ev/'combined_signals.jsonl')==done['combined_sha256']
    assert sha(ev/'signal_summary.json')==done['summary_sha256']
    manifests={}
    for directory in sorted((OUT/'stages').iterdir()):
        manifest=metadata(directory/'archive_manifest.json')
        for r in manifest['files']:
            assert sha(directory/r['snapshot'])==r['sha256'] and sha(ROOT/r['source'])==r['sha256']
        manifests[directory.name]=sha(directory/'archive_manifest.json')
    for c in read(BASE/'manifest.jsonl'):
        assert sha(c['noisy_path'])==c['input_sha256'] and sha(c['reference_path'])==c['reference_sha256']
    verified=metadata(OUT/'verified_files.json')
    for p,h in verified.items():assert sha(p)==h
    for r in metadata(OUT/'RIR_manifest.json'):assert sha(r['path'])==r['sha256']
    # Finish already read every output hash; here verify its full concatenation anchors.
    with (ev/'combined_signals.jsonl').open('rb') as full:
        for phase in ('direct','blind','truth'):
            with (ev/(phase+'_signals.jsonl')).open('rb') as part:
                for block in iter(lambda:part.read(1024*1024),b''):assert full.read(len(block))==block
        assert not full.read(1)
    resources=metadata(ev/'runtime_resources.json');assert len(resources)==10
    audit={'status':'ok','utc':now(),'signal_rows':16488,'stages':manifests,'RT0_pairs':2748,
           'RT0_exact_pairs':done['RT0_exact_pairs'],'RT0_max_abs':done['RT0_max_abs'],
           'max_allocated_bytes':max(r['peak_allocated_bytes'] for r in resources),
           'max_reserved_bytes':max(r['peak_reserved_bytes'] for r in resources),
           'combined_sha256':done['combined_sha256'],'summary_sha256':done['summary_sha256'],
           'source_inputs_rechecked':504,'ASR_complete':False,'power_helper':power}
    dump(OUT/'SIGNAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G22_信号阶段独立验收与资源读回.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G22 信号阶段独立验收与资源读回\n\n完整16488条输出已通过阶段覆盖与哈希验收，三个信号清单与合并文件逐字节一致，504条输入/参考及真实RIR来源再次核对。2748对RT0输出检查通过。临时保持唤醒已释放。\n\n10个神经前端的实际Torch分配峰值最大 '+f"{audit['max_allocated_bytes']/2**30:.3f}"+' GiB，保留峰值最大 '+f"{audit['max_reserved_bytes']/2**30:.3f}"+' GiB；两者不等同整机显存或流式实时能力。无训练或新依赖。\n\n本轮新房间参考波形可能不同，后续ASR应按样本哈希复用或实际解码参考，不可直接套用旧房间参考成绩。真值统计仅作诊断，不能部署或作为上界。\n')
    archive('signal_acceptance',report,[OUT/'SIGNAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g22_signal_accept.py',OUT/'signal_power_helper.json'])
    status(stage='signal_accepted_ASR_pending')


if __name__=='__main__':main()
