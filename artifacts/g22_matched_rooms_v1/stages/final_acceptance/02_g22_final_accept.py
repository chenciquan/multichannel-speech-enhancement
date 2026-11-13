"""Final independent provenance, cache and matched-control gate for G22."""
from collections import Counter
from g22_common import ROOT,OUT,sha,read,dump,now,archive,metadata,guard,status
from g19_scores import load_cache


def main():
    guard();rec=OUT/'recognition';plan=metadata(rec/'frozen_plan.json')
    major=metadata(OUT/'MAJOR_STAGE_COMPLETE.json');done=metadata(rec/'completion_audit.json')
    assert major['status']==done['status']=='ok'
    assert done['signals']==done['ASR_rows_each']==16488 and done['paired_outputs']==8244 and done['RT0_pairs']==2748
    assert sha(ROOT/major['report'])==major['report_sha256']
    assert sha(OUT/'stages/recognition_completion/archive_manifest.json')==major['archive_sha256']
    assert sha(rec/'summary.json')==done['summary_sha256']
    assert sha(rec/'paired_room_deltas.jsonl')==done['paired_deltas_sha256']
    assert sha(OUT/'evaluation/combined_signals.jsonl')==plan['signal_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    power=metadata(OUT/'ASR_power_helper.json');assert power['status']=='released'
    assert metadata(OUT/'signal_power_helper.json')['status']=='released'
    anchors={}
    for directory in sorted((OUT/'stages').iterdir()):
        manifest=metadata(directory/'archive_manifest.json')
        for row in manifest['files']:
            assert sha(directory/row['snapshot'])==row['sha256'] and sha(ROOT/row['source'])==row['sha256']
        anchors[directory.name]=sha(directory/'archive_manifest.json')
    prior_root=ROOT/'artifacts/g21_truth_covariance_v1/recognition';backend={}
    expected={(r['system'],r['sample_id']) for r in read(OUT/'evaluation/combined_signals.jsonl')}
    for engine in ('Whisper','CTC'):
        folder=rec/engine;complete=metadata(folder/'COMPLETE.json');provenance=metadata(folder/'provenance.json')
        assert sha(folder/'cache_new.jsonl')==complete['cache_new_sha256']
        assert sha(folder/'asr_scores.jsonl')==complete['asr_scores_sha256']
        assert sha(prior_root/engine/'provenance.json')==plan['prior'][engine]['provenance_sha256']
        assert sha(prior_root/engine/'cache_new.jsonl')==plan['prior'][engine]['cache_sha256']
        prior=metadata(prior_root/engine/'provenance.json')
        assert all(provenance[k]==prior[k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
        cache,conflicts,sources=load_cache(engine,prior,prior_root/engine)
        assert not conflicts and sources==provenance['source_caches']
        added=read(folder/'cache_new.jsonl');kinds=Counter()
        for r in added:
            assert r['sample_sha256'] not in cache and r['model_id']==prior['model_id']
            assert r['first_kind'] in ('reference','output');kinds[r['first_kind']]+=1
            cache[r['sample_sha256']]=r['hypothesis']
        assert len(added)==sum(complete['new_decodes'].values()) and dict(kinds)==complete['new_decodes']
        assert len(cache)==complete['cache_entries']
        keys=set();references={}
        for r in read(folder/'asr_scores.jsonl'):
            k=r['system'],r['sample_id'];assert k not in keys;keys.add(k)
            assert r['hypothesis']==cache[r['output_samples_sha256']]
            assert r['reference_hypothesis']==cache[r['reference_samples_sha256']]
            reference=(r['reference_samples_sha256'],r['reference_hypothesis'])
            if r['sample_id'] in references:assert reference==references[r['sample_id']]
            else:references[r['sample_id']]=reference
        assert keys==expected and len(keys)==16488 and len(references)==504
        backend[engine]={'rows':len(keys),'reference_cases':len(references),'new_reference_decodes':kinds['reference'],
                         'new_output_decodes':kinds['output'],'cache_conflicts':0,
                         'score_sha256':complete['asr_scores_sha256'],'cache_sha256':complete['cache_new_sha256']}
    audit={'status':'ok','utc':now(),'signal_and_each_ASR_rows':16488,'paired_room_outputs':8244,
           'RT0_pairs':2748,'RT0_exact_PCM_pairs':done['RT0_exact_PCM_pairs'],'ASR':backend,'archives':anchors,
           'completion_audit_sha256':sha(rec/'completion_audit.json'),'power_helper':power}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G22_完整双房间阶段验收与简明读回.md'
    lines=['# G22 完整双房间阶段验收与简明读回','','16488条信号和每个识别器的16488条评分均已完成；8244对房间输出覆盖一致。完整阶段快照、冻结代码、模型注册及新评分/缓存通过哈希核验。',
           '',f"2748对RT0输出中，{done['RT0_exact_PCM_pairs']}对PCM哈希完全一致，其两个后端评分均一致；其他输出差异见信号验收，未放宽预定阈值。",'']
    for engine,c in backend.items():
        lines.append(f"- {engine}：504个参考条件逐一匹配；新增实际解码参考{c['new_reference_decodes']}条、输出{c['new_output_decodes']}条，无缓存冲突。")
    lines+=['','全部三训练种子与既定参数保留，未新增训练。真值协方差只作诊断，不作为部署方法或理论上界。参考和新房间结果均按样本哈希处理，没有按旧转写文本套用成绩。所有临时保持唤醒请求已释放。','',
            '[完整房间配对结果](G22_双房间配对信号与识别综合报告.md)包含全部策略、干净控制及分种子/方向/场型信息。只有一个新增房间和四句已用语音，不能声称广泛泛化或稳定人声误删。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('final_acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g22_final_accept.py',OUT/'ASR_power_helper.json'])
    status(stage='accepted')


if __name__=='__main__':main()
