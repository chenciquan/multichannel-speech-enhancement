"""Read back the oracle-statistic signal stage without rerunning filters."""
import json
import numpy as np
import soundfile as sf
from g21_truth_covariance import ROOT,OUT,sha,read,dump,now,archive


def main():
    done=json.loads((OUT/'SIGNAL_COMPLETE.json').read_text())
    assert done['status']=='ok' and done['rows']==864 and done['oracle'] and not done['deployable']
    assert json.loads((OUT/'power_helper.json').read_text(encoding='utf-8-sig'))['status']=='released'
    assert sha(OUT/'signals.jsonl')==done['signals_sha256'] and sha(OUT/'summary.json')==done['summary_sha256']
    anchors={}
    for directory in sorted((OUT/'stages').iterdir()):
        manifest=json.loads((directory/'archive_manifest.json').read_text())
        for r in manifest['files']:
            assert sha(directory/r['snapshot'])==r['sha256'] and sha(ROOT/r['source'])==r['sha256']
        anchors[directory.name]=sha(directory/'archive_manifest.json')
    for p,h in json.loads((OUT/'frozen_plan.json').read_text())['hashes'].items():assert sha(ROOT/p)==h
    for p,h in json.loads((OUT/'verified_files.json').read_text()).items():assert sha(p)==h
    cases={r['sample_id']:r for r in read(ROOT/'data/g18_v1/manifest.jsonl') if r['condition']!='clean'}
    rows=read(OUT/'signals.jsonl');index={(r['system'],r['sample_id']):r for r in rows}
    assert len(rows)==len(index)==864 and set(index)=={(n,s) for n in ('truth_covariance_mvdr','truth_covariance_mwf') for s in cases}
    for r in rows:
        assert r['oracle_clean_information'] and r['deployable'] is False
        assert all(r[k]==v for k,v in cases[r['sample_id']].items())
        assert sha(r['output_path'])==r['output_sha256']
        a,sr=sf.read(r['output_path'],dtype='float32')
        assert sr==16000 and a.shape==(r['frames'],) and np.isfinite(a).all()
        assert r['linearity_max_abs']<1e-6*(1+float(np.abs(a).max()))
    for pre in json.loads((OUT/'preflight.json').read_text()):
        full=index[pre['system'],pre['sample_id']]
        assert pre=={k:v for k,v in full.items() if k not in ('output_path','output_sha256')}
    audit={'status':'ok','completed_utc':now(),'rows':864,'cases':432,'preflight_exact_matches':6,
           'archives':anchors,'oracle_only':True,'ASR_scored':False,'outputs_read_back':864,
           'signal_sha256':done['signals_sha256'],'summary_sha256':done['summary_sha256']}
    dump(OUT/'SIGNAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G21_真值统计信号阶段验收.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G21 真值统计信号阶段验收\n\n全部432个条件、2个使用真值统计的滤波器，共864条输出通过覆盖、波形与SHA核验。6项预检与完整运行逐字段一致；所有冻结代码、来源及阶段快照哈希通过，临时保持唤醒已释放。\n\n[信号结果](G21B_真值协方差信号报告.md)仅用于已查看单房间中的诊断。它不是部署方法，也不能当作理论性能上界。本阶段尚无这些新输出的ASR成绩，不能据声学指标推断识别改善。尚待判断是否需要固定ASR补查，再做独立房间确认。\n')
    archive('signal_acceptance',report,[OUT/'SIGNAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g21_accept.py',OUT/'power_helper.json'])
    dump(OUT/'status.json',{'stage':'signal_accepted','utc':now()})


if __name__=='__main__':main()
