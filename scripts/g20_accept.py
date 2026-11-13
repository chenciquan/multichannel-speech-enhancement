"""Independent readback of archived G20 diagnostics; no numerical rerun."""
import json
from collections import Counter, defaultdict
import numpy as np
from g20_replay import ROOT, OUT, G18, sha, read, dump, now, archive


def main():
    folder = OUT/'filter'
    complete = json.loads((folder/'COMPLETE.json').read_text())
    assert complete['status']=='ok' and complete['rows']==8640
    assert json.loads((OUT/'power_helper.json').read_text(encoding='utf-8-sig'))['status']=='released'
    stages = {}
    for directory in sorted((OUT/'stages').iterdir()):
        manifest = json.loads((directory/'archive_manifest.json').read_text())
        for r in manifest['files']:
            assert sha(directory/r['snapshot'])==r['sha256'] and sha(ROOT/r['source'])==r['sha256']
        stages[directory.name] = sha(directory/'archive_manifest.json')
    for filename, key in (('components.jsonl','components_sha256'),('summary.json','summary_sha256')):
        assert sha(folder/filename)==complete[key]
    for name in ('frozen_replay_plan.json','filter/frozen_plan.json'):
        plan = json.loads((OUT/name).read_text())
        for p,h in plan.get('source_hashes',plan.get('hashes')).items(): assert sha(ROOT/p)==h
    for name in ('verified_files.json','filter/verified_files.json'):
        for p,h in json.loads((OUT/name).read_text()).items(): assert sha(p)==h
    rows = read(folder/'components.jsonl')
    cases = {r['sample_id']:r for r in read(ROOT/'data/g18_v1/manifest.jsonl') if r['condition']!='clean'}
    names = [r['system'] for r in json.loads((G18/'systems.json').read_text())['direct'] if r['kind']!='noisy']
    index = {(r['sample_id'],r['frontend'],r['method']):r for r in rows}
    assert len(index)==len(rows)==8640 and set(index)=={(s,n,m) for s in cases for n in names for m in ('mvdr','mwf')}
    for r in json.loads((folder/'preflight.json').read_text()):
        assert r==index[r['sample_id'],r['frontend'],r['method']]
    for r in rows:
        c = cases[r['sample_id']]
        assert all(r[k]==c[k] for k in ('source_id','speaker_id','requested_rt60','noise_field','noise_id','target_azimuth','relative_angle'))
        m=r['metrics']; assert all(np.isfinite(v) for v in m.values() if isinstance(v,(int,float)))
        assert r['saved_output_max_abs']<=1e-6 and m['snr_accounting_error_db']<1e-8
        assert abs(m['reconstructed_total_nmse']-m['target_error_nmse']-m['residual_noise_nmse']-m['error_cross_term'])<1e-10*(1+m['reconstructed_total_nmse'])
    measures=('speech_error_nmse','target_gain_db','noise_reduction_db','component_output_snr_db','residual_noise_nmse','error_cross_term')
    def aggregate(keys):
        buckets=defaultdict(list)
        for r in rows: buckets[tuple(r[k] for k in keys)].append(r)
        return [{**dict(zip(keys,k)),'rows':len(v), **{m:float(np.mean([r['metrics'][m] for r in v])) for m in measures}} for k,v in buckets.items()]
    stratified={label:aggregate(keys) for label,keys in {
        'per_seed':('requested_rt60','group','frontend','seed','method'),
        'per_speaker':('requested_rt60','group','speaker_id','method'),
        'per_direction':('requested_rt60','group','target_azimuth','method'),
        'per_field':('requested_rt60','group','noise_field','relative_angle','method')}.items()}
    dump(folder/'stratified_readback.json',stratified,True)
    audit={'status':'ok','completed_utc':now(),'rows':len(rows),'case_count':len(cases),'frontends':len(names),
           'stages_checked':stages,'preflight_exact_matches':60,'complete_sha256':sha(folder/'COMPLETE.json'),
           'stratified_sha256':sha(folder/'stratified_readback.json'),
           'saved_output_max_abs':max(r['saved_output_max_abs'] for r in rows),
           'linearity_max_abs':max(r['linearity_max_abs'] for r in rows),
           'new_training_or_ASR':False}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G20_分量诊断验收与简明读回.md'
    lines=['# G20 分量诊断验收与简明读回','','全部432个带噪条件、10前端、2滤波器共8640项覆盖完整。固定60项预检与正式运行结果逐字段一致；协议、还原、预检和完成归档均通过哈希核验。',
           '',f"旧混合输出最大复现误差 {audit['saved_output_max_abs']:.3g}，固定滤波器分量线性叠加最大误差 {audit['linearity_max_abs']:.3g}。语音/噪声/交叉项的平方误差核算通过。",
           '', '分种子、说话人、方向和空间场型均已保存，未把重复条件当独立语句。诊断限定于既有单房间和已用语音，不能证明吞字、掩码因果或跨房间泛化。',
           '', '[完整分量报告](G20_固定房间滤波分量诊断报告.md)', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',folder/'stratified_readback.json',ROOT/'scripts/g20_accept.py'])
    dump(OUT/'status.json',{'stage':'accepted','utc':now()})


if __name__=='__main__':main()
