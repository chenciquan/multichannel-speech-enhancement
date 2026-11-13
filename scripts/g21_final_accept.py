"""Final file and cache readback of the two-recognizer diagnostic stage."""
import json
from g21_truth_covariance import ROOT,OUT,sha,dump,read,now,archive
from g19_scores import load_cache


def metadata(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    rec=OUT/'recognition';plan=metadata(rec/'frozen_plan.json')
    major=metadata(OUT/'MAJOR_STAGE_COMPLETE.json');done=metadata(rec/'completion_audit.json')
    power=metadata(OUT/'ASR_power_helper.json')
    assert major['status']==done['status']=='ok' and power['status']=='released'
    assert done['rows_per_ASR']==14256 and done['systems']==33 and done['new_rows_per_ASR']==864
    assert sha(ROOT/major['report'])==major['report_sha256']
    assert sha(OUT/'stages/recognition_completion/archive_manifest.json')==major['archive_sha256']
    assert sha(rec/'summary.json')==done['summary_sha256']
    assert sha(OUT/'signals.jsonl')==done['signal_sha256']==plan['signal_sha256']
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    anchors={}
    for directory in sorted((OUT/'stages').iterdir()):
        manifest=metadata(directory/'archive_manifest.json')
        for row in manifest['files']:
            assert sha(directory/row['snapshot'])==row['sha256'] and sha(ROOT/row['source'])==row['sha256']
        anchors[directory.name]=sha(directory/'archive_manifest.json')
    g18=ROOT/'artifacts/g18_room_stress_v1/evaluation';g19=ROOT/'artifacts/g19_room_mwf_v1/evaluation'
    results={}
    for engine in ('Whisper','CTC'):
        for prefix,base in (('g18_',g18),('g19_',g19)):
            for name in ('provenance.json','asr_scores.jsonl','cache_new.jsonl'):
                assert sha(base/engine/name)==plan['history_sha256'][engine][prefix+name]
        folder=rec/engine;complete=metadata(folder/'COMPLETE.json')
        for name,key in (('cache_new.jsonl','cache_new_sha256'),('asr_scores.jsonl','asr_scores_sha256')):
            assert sha(folder/name)==complete[key]
        prior=metadata(g19/engine/'provenance.json')
        provenance=metadata(folder/'provenance.json')
        assert all(provenance[k]==prior[k] for k in ('engine','model_id','decoding','normalization','reference_prompt'))
        cache,conflicts,sources=load_cache(engine,prior,g19/engine)
        assert not conflicts and sources==provenance['source_caches']
        new=read(folder/'cache_new.jsonl')
        assert len(new)==complete['new_decodes'] and len({r['sample_sha256'] for r in new})==len(new)
        for r in new:
            assert r['model_id']==prior['model_id'] and r['sample_sha256'] not in cache
            cache[r['sample_sha256']]=r['hypothesis']
        scores=read(folder/'asr_scores.jsonl');assert len(scores)==864
        for r in scores:
            assert cache[r['output_samples_sha256']]==r['hypothesis']
            assert cache[r['reference_samples_sha256']]==r['reference_hypothesis']
        results[engine]={'rows':864,'new_decodes':len(new),'conflicts':0,'score_sha256':complete['asr_scores_sha256'],'cache_sha256':complete['cache_new_sha256']}
    audit={'status':'ok','completed_utc':now(),'cases':432,'systems':33,'rows_each_ASR':14256,
           'new_rows_each_ASR':864,'ASR':results,'stages':anchors,'power_helper':power,
           'completion_audit_sha256':sha(rec/'completion_audit.json')}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G21_真值统计双识别器验收读回.md'
    summary=metadata(rec/'summary.json')
    lines=['# G21 真值统计双识别器验收读回','','33系统、432条件的14256对照/后端已完成。新增864条输出的两个识别器缓存、评分、固定解码来源、全部阶段快照与历史文件均核验通过。临时保持唤醒已释放。','',
           '|请求RT|真值统计滤波器|SI-SDR dB|Whisper WER|CTC WER|','|---:|---|---:|---:|---:|']
    for r in sorted([r for r in summary['by_rt'] if r['group']=='truth_covariance'],key=lambda r:(r['rt'],r['method'])):
        lines.append(f"|{r['rt']}|{r['method']}|{r['si_sdr_db']:.2f}|{r['Whisper']['wer']:.2%}|{r['CTC']['wer']:.2%}|")
    lines+=['','声学指标和WER分别衡量不同方面。本表仅为使用真值统计的诊断结果，不是可部署效果或理论上界。参考词数、S/D/I、所有历史前端和三种子对照见[综合报告](G21_真值统计与识别保真综合报告.md)。单房间、8句/4人及重复条件限制不变，没有新的人工逐词标签。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('final_acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g21_final_accept.py',OUT/'ASR_power_helper.json'])
    dump(OUT/'status.json',{'stage':'accepted','utc':now()})


if __name__=='__main__':main()
