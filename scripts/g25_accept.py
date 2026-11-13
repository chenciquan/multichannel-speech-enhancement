"""Independent archive/model identity and normalized observation coverage audit."""
from g25_prepare import ROOT,OUT,PARENT,sha,read,dump,now,archive
from g25_asr import meta,status


def main():
    rec=OUT/'recognition';plan=meta(rec/'frozen_plan.json');major=meta(OUT/'MAJOR_STAGE_COMPLETE.json');done=meta(rec/'completion_audit.json')
    assert major['status']==done['status']=='ok' and meta(OUT/'power_helper.json')['status']=='released'
    assert sha(ROOT/major['report'])==major['report_sha256'] and sha(OUT/'stages/completion/archive_manifest.json')==major['archive_sha256']
    assert sha(rec/'summary.json')==done['summary_sha256'] and sha(rec/'paired_observations.json')==done['paired_sha256']
    assert sha(OUT/'manifest.jsonl')==plan['manifest_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    anchors={}
    for directory in (OUT/'stages').iterdir():
        for r in meta(directory/'archive_manifest.json')['files']:
            assert sha(directory/r['snapshot'])==r['sha256'] and sha(ROOT/r['source'])==r['sha256']
        anchors[directory.name]=sha(directory/'archive_manifest.json')
    expected={(r['base_system'],r['sample_id'],r['observation']) for r in read(OUT/'manifest.jsonl')};checks={}
    for e in ('Whisper','CTC'):
        folder=rec/e;current=meta(folder/'provenance.json');prior_dir=PARENT/'recognition'/e;prior=meta(prior_dir/'provenance.json')
        assert sha(prior_dir/'provenance.json')==plan['parent'][e]['provenance_sha256']
        assert sha(prior_dir/'observations.jsonl')==plan['parent'][e]['observations_sha256']
        assert all(current[k]==prior[k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
        assert current['ASR_regime']=='level_control_v1' and current['cache_skipping'] is False
        assert current['input_preprocessing']==meta(OUT/'frozen_plan.json')['normalization']
        obs=read(folder/'observations.jsonl');marker=meta(folder/'COMPLETE.json')
        assert sha(folder/'observations.jsonl')==marker['observations_sha256'] and marker['fresh_forwards']==960 and marker['cache_skips']==0
        assert len(obs)==960 and {r['decode_index'] for r in obs}==set(range(960))
        assert {(r['base_system'],r['sample_id'],r['observation']) for r in obs}==expected
        assert all(r['actual_forward'] and r['model_id']==prior['model_id'] for r in obs)
        checks[e]={'fresh_forwards':960,'raw_parent_pairs':480,'cache_skips':0}
    audit={'status':'ok','utc':now(),'ASR':checks,'repeatability':done['repeatability'],'stages':anchors,
           'normalization_id':meta(OUT/'frozen_plan.json')['normalization']['preprocessing_id'],
           'completion_audit_sha256':sha(rec/'completion_audit.json')}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G25_识别电平控制阶段验收.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G25 识别电平控制阶段验收\n\n两后端各960次独立前向完整，无缓存跳过。原始模型和解码配置保持一致，新输入电平预处理单独标识，raw对照来源、全部冻结代码、清单、阶段快照及观察文件哈希通过。原音频与旧成绩未改，临时保持唤醒已释放。\n\n[原幅度与电平控制评分](G25_识别输入电平控制与原始评分对照报告.md)分别保存，不选择更低结果覆盖原基准，不将评估预处理当作增强模型创新。\n')
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g25_accept.py',OUT/'power_helper.json'])
    status(stage='accepted')


if __name__=='__main__':main()
