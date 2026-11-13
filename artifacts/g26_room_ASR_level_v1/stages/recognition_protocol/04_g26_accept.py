"""Independent normalized-cache and full paired-score acceptance for G26."""
from g26_prepare import ROOT,OUT,SOURCE,sha,read,dump,now,archive
from g26_asr import meta,seed_cache,status


def main():
    rec=OUT/'recognition';plan=meta(rec/'frozen_plan.json');major=meta(OUT/'MAJOR_STAGE_COMPLETE.json');done=meta(rec/'completion_audit.json')
    assert major['status']==done['status']=='ok' and done['rows_each_ASR']==16488 and done['RT0_pairs']==2748
    assert meta(OUT/'power_helper.json')['status']=='released'
    assert sha(ROOT/major['report'])==major['report_sha256'] and sha(OUT/'stages/completion/archive_manifest.json')==major['archive_sha256']
    assert sha(rec/'summary.json')==done['summary_sha256'] and sha(rec/'raw_to_level_changes.json')==done['changes_sha256']
    assert sha(OUT/'manifest.jsonl')==plan['manifest_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    stages={}
    for directory in (OUT/'stages').iterdir():
        for r in meta(directory/'archive_manifest.json')['files']:
            assert sha(directory/r['snapshot'])==r['sha256'] and sha(ROOT/r['source'])==r['sha256']
        stages[directory.name]=sha(directory/'archive_manifest.json')
    expected={(r['system'],r['sample_id']) for r in read(OUT/'manifest.jsonl')};checks={}
    for e in ('Whisper','CTC'):
        folder=rec/e;marker=meta(folder/'COMPLETE.json');prior,cache=seed_cache(e,plan)
        assert sha(SOURCE/'recognition'/e/'asr_scores.jsonl')==plan['raw_comparators'][e]
        assert sha(folder/'asr_scores.jsonl')==marker['score_sha256'] and sha(folder/'cache_new.jsonl')==marker['cache_sha256']
        current=meta(folder/'provenance.json');assert all(current[k]==prior[k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
        assert current['ASR_regime']=='level_control_v1' and current['input_preprocessing']==prior['input_preprocessing']
        new=read(folder/'cache_new.jsonl');assert len(new)==sum(marker['new_decodes'].values())
        for r in new:
            assert r['sample_sha256'] not in cache and r['model_id']==prior['model_id']
            assert r['normalization_id']==current['input_preprocessing']['preprocessing_id']
            cache[r['sample_sha256']]=r['hypothesis']
        scores=read(folder/'asr_scores.jsonl');assert len(scores)==16488 and {(r['system'],r['sample_id']) for r in scores}==expected
        refs={}
        for r in scores:
            assert r['hypothesis']==cache[r['output_samples_sha256']] and r['reference_hypothesis']==cache[r['reference_samples_sha256']]
            reference=r['reference_samples_sha256'],r['reference_hypothesis']
            if r['sample_id'] in refs:assert refs[r['sample_id']]==reference
            else:refs[r['sample_id']]=reference
        assert len(refs)==504
        checks[e]={'rows':16488,'reference_cases':504,'compatible_seed_entries':480,'new_decodes':marker['new_decodes']}
    audit={'status':'ok','utc':now(),'ASR':checks,'stages':stages,'RT0_pairs':2748,
           'method_sign_flips':done['method_sign_flips'],'completion_audit_sha256':sha(rec/'completion_audit.json')}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G26_完整电平辅评分阶段验收.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G26 完整电平辅评分阶段验收\n\n两个后端各16488评分、504参考条件及2748对RT0一致性核验通过。冻结代码、预处理身份、规范化输入清单、G25兼容种子、新缓存、评分及阶段快照均通过哈希验收，原raw评分保持原样。临时保持唤醒已释放。\n\n[完整对照](G26_完整双房间原始与电平控制识别对照.md)分别保留raw和统一规则电平ASR，不将辅助评分当增强模型改善或测试集最优选择。所有种子与场景保留，没有新训练。\n')
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g26_accept.py',OUT/'power_helper.json'])
    status(stage='accepted')


if __name__=='__main__':main()
