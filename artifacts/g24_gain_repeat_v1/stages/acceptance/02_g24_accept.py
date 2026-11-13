"""Independent fresh-forward coverage, historical integrity and archive audit."""
from g24_prepare import ROOT,OUT,PARENT,sha,read,dump,now,archive
from g24_asr import meta,status
from g19_scores import load_cache


def main():
    rec=OUT/'recognition';plan=meta(rec/'frozen_plan.json');major=meta(OUT/'MAJOR_STAGE_COMPLETE.json')
    done=meta(rec/'completion_audit.json');power=meta(OUT/'power_helper.json')
    assert major['status']==done['status']=='ok' and power['status']=='released'
    assert sha(ROOT/major['report'])==major['report_sha256'] and sha(OUT/'stages/completion/archive_manifest.json')==major['archive_sha256']
    assert sha(rec/'summary.json')==done['summary_sha256'] and sha(rec/'paired_observations.json')==done['paired_observations_sha256']
    assert sha(OUT/'manifest.jsonl')==plan['manifest_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    anchors={}
    for directory in (OUT/'stages').iterdir():
        for r in meta(directory/'archive_manifest.json')['files']:
            assert sha(directory/r['snapshot'])==r['sha256'] and sha(ROOT/r['source'])==r['sha256']
        anchors[directory.name]=sha(directory/'archive_manifest.json')
    expected={(r['base_system'],r['sample_id'],r['observation']) for r in read(OUT/'manifest.jsonl')}
    checks={}
    for e in ('Whisper','CTC'):
        folder=rec/e;marker=meta(folder/'COMPLETE.json');current=meta(folder/'provenance.json');prior_folder=PARENT/'recognition'/e
        for name,field in (('provenance.json','provenance_sha256'),('asr_scores.jsonl','scores_sha256'),('cache_new.jsonl','cache_sha256')):
            assert sha(prior_folder/name)==plan['parent'][e][field]
        prior=meta(prior_folder/'provenance.json');cache,conflicts,sources=load_cache(e,prior,prior_folder)
        assert not conflicts and sources==current['source_caches'] and current['cache_skipping'] is False
        assert all(current[k]==prior[k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
        assert sha(folder/'observations.jsonl')==marker['observations_sha256']
        obs=read(folder/'observations.jsonl');assert len(obs)==1920 and marker['actual_forwards']==1920 and marker['cache_skips']==0
        assert {r['decode_index'] for r in obs}==set(range(1920))
        assert {(r['base_system'],r['sample_id'],r['observation']) for r in obs}==expected
        for r in obs:
            assert r['actual_forward'] and r['model_id']==prior['model_id']
            if r['observation'].startswith('original_'):
                assert cache[r['output_samples_sha256']]==r['historical_baseline_hypothesis']
        checks[e]={'fresh_observations':1920,'cache_skips':0,'historic_baselines_preserved':480}
    audit={'status':'ok','utc':now(),'ASR':checks,'repeatability':done['repeatability'],'stages':anchors,
           'power_helper':power,'completion_audit_sha256':sha(rec/'completion_audit.json')}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G24_独立重复验证阶段验收.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G24 独立重复验证阶段验收\n\n两个后端各1920个真实前向观察完整，解码序号无缺失，未跳过任何重复输入。历史G23/G22缓存保持原样；当前观察独立保存，未合入历史首观测缓存。模型配置、来源、清单、全部阶段与观察文件哈希均通过，临时保持唤醒已释放。\n\n[独立重复及增益结果](G24_独立重复与精确增益识别验证报告.md)分别报告重复性和缩放效应，保留全部基准和不稳定观察，不选择最优增益。无增强训练、部署或人声误删结论。\n')
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g24_accept.py',OUT/'power_helper.json'])
    status(stage='accepted')


if __name__=='__main__':main()
