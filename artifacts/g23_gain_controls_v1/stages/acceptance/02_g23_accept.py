"""Final cache/provenance verification for the fixed-gain diagnostic."""
from g23_prepare import ROOT,OUT,G22,sha,dump,read,now,archive
from g23_asr import meta,status
from g19_scores import load_cache


def main():
    rec=OUT/'recognition';plan=meta(rec/'frozen_plan.json');major=meta(OUT/'MAJOR_STAGE_COMPLETE.json')
    done=meta(rec/'completion_audit.json');power=meta(OUT/'power_helper.json')
    assert major['status']==done['status']=='ok' and power['status']=='released' and done['rows_each_ASR']==1440
    assert sha(ROOT/major['report'])==major['report_sha256']
    assert sha(OUT/'stages/completion/archive_manifest.json')==major['archive_sha256']
    assert sha(rec/'summary.json')==done['summary_sha256'] and sha(rec/'paired_changes.json')==done['paired_changes_sha256']
    assert sha(OUT/'manifest.jsonl')==plan['manifest_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    anchors={}
    for directory in (OUT/'stages').iterdir():
        for r in meta(directory/'archive_manifest.json')['files']:
            assert sha(directory/r['snapshot'])==r['sha256'] and sha(ROOT/r['source'])==r['sha256']
        anchors[directory.name]=sha(directory/'archive_manifest.json')
    backends={}
    for engine in ('Whisper','CTC'):
        folder=rec/engine;complete=meta(folder/'COMPLETE.json');prior_folder=G22/'recognition'/engine
        assert sha(prior_folder/'provenance.json')==plan['parent'][engine]['provenance_sha256']
        assert sha(prior_folder/'cache_new.jsonl')==plan['parent'][engine]['cache_sha256']
        assert sha(prior_folder/'asr_scores.jsonl')==plan['parent'][engine]['score_sha256']
        assert sha(folder/'asr_scores.jsonl')==complete['scores_sha256'] and sha(folder/'cache_new.jsonl')==complete['cache_sha256']
        prior=meta(prior_folder/'provenance.json');current=meta(folder/'provenance.json')
        assert all(current[k]==prior[k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
        cache,conflicts,sources=load_cache(engine,prior,prior_folder);assert not conflicts and sources==current['source_caches']
        for r in read(folder/'cache_new.jsonl'):
            assert r['sample_sha256'] not in cache and r['model_id']==prior['model_id'] and r['gain_db_first_seen'] in (-6,-12)
            cache[r['sample_sha256']]=r['hypothesis']
        for r in read(folder/'asr_scores.jsonl'):
            assert r['hypothesis']==cache[r['output_samples_sha256']] and r['reference_hypothesis']==cache[r['reference_samples_sha256']]
        assert complete['rows']==1440 and complete['zero_db_cache_hits']==480
        backends[engine]={'rows':1440,'baseline_cache_hits':480,'new_decodes':complete['new_decodes'],'cache_conflicts':0}
    audit={'status':'ok','utc':now(),'ASR':backends,'stages':anchors,
           'completion_audit_sha256':sha(rec/'completion_audit.json'),'power_helper':power}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G23_增益控制阶段验收.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G23 增益控制阶段验收\n\n两后端各1440记录完成，480个0dB基准全部命中原G22缓存；新解码只用于非零增益。所有来源、冻结代码、清单、分阶段快照、缓存与评分哈希通过。模型身份和解码配置未变，临时保持唤醒已释放。\n\n[配对结果](G23_固定增益与双识别器敏感性报告.md)保留全部固定增益、种子、方向和说话人，未选择最优增益。未进行增强推理、训练、TSOS实现或人工听感判定。\n')
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g23_accept.py',OUT/'power_helper.json'])
    status(stage='accepted')


if __name__=='__main__':main()
