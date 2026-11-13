"""Independent readback of matched-room audio and negative controls."""
import json
from collections import defaultdict
import numpy as np
import soundfile as sf
from g22_prepare import ROOT,OUT,BASE,sha,read,dump,now,archive
from mcse.simulation import snr_db


def main():
    done=json.loads((OUT/'DATA_COMPLETE.json').read_text());plan=json.loads((OUT/'frozen_plan.json').read_text())
    assert done['status']=='ok' and done['cases']==504 and sha(BASE/'manifest.jsonl')==done['manifest_sha256']
    assert json.loads((OUT/'data_power_helper.json').read_text(encoding='utf-8-sig'))['status']=='released'
    anchors={}
    for directory in sorted((OUT/'stages').iterdir()):
        manifest=json.loads((directory/'archive_manifest.json').read_text())
        for r in manifest['files']:
            assert sha(directory/r['snapshot'])==r['sha256'] and sha(ROOT/r['source'])==r['sha256']
        anchors[directory.name]=sha(directory/'archive_manifest.json')
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    for n,h in json.loads((OUT/'verified_files.json').read_text()).items():assert sha(n)==h
    for r in json.loads((OUT/'RIR_manifest.json').read_text()):assert sha(r['path'])==r['sha256']
    rows=read(BASE/'manifest.jsonl');index={r['sample_id']:r for r in rows};pairs=defaultdict(dict);groups=defaultdict(list)
    old={r['sample_id']:r for r in read(ROOT/'data/g18_v1/manifest.jsonl')}
    assert len(index)==len(rows)==504 and sum(r['condition']=='clean' for r in rows)==72
    assert len({r['source_id'] for r in rows})==len({r['speaker_id'] for r in rows})==4
    errors=[]
    for r in rows:
        prior=old[r['old_G18_sample_id']]
        assert all(r[k]==prior[k] for k in ('source_id','speaker_id','transcript','target_azimuth','requested_rt60','noise_intervals','base_noise_directions','noise_id','noise_field','relative_angle','frames'))
        for path,key in (('noisy_path','input_sha256'),('reference_path','reference_sha256')):assert sha(r[path])==r[key]
        x,sr=sf.read(r['noisy_path'],dtype='float64');s,rr=sf.read(r['reference_path'],dtype='float64')
        assert sr==rr==16000 and x.shape==(r['frames'],6) and s.shape==(r['frames'],)
        assert np.isfinite(x).all() and np.isfinite(s).all() and np.abs(x).max()<1 and np.abs(s).max()<1
        if r['condition']!='clean':
            mask=np.zeros(r['frames'],dtype=bool)
            for lo,hi in r['activity_spans_from_unquantized_reference']:mask[lo:hi]=True
            err=abs(snr_db(s,x[:,4]-s,mask));assert err<plan['SNR_tolerance_db'] and abs(err-r['saved_active_SNR_error_db'])<1e-12
            errors.append(err)
        else:assert np.array_equal(x[:,4],s)
        pair=pairs[r['matched_room_pair']];assert r['room_id'] not in pair;pair[r['room_id']]=r
        groups[r['paired_room_group']].append(r)
    assert len(pairs)==252 and all(set(v)=={'reference','new'} for v in pairs.values())
    assert len(groups)==28 and all(len(v)==18 and len({r['common_scale'] for r in v})==1 for v in groups.values())
    controls=[]
    for pair in pairs.values():
        a,b=pair['reference'],pair['new'];assert a['common_scale']==b['common_scale']
        if a['requested_rt60']!=0:continue
        x=sf.read(a['noisy_path'],dtype='float64')[0];y=sf.read(b['noisy_path'],dtype='float64')[0]
        s=sf.read(a['reference_path'],dtype='float64')[0];t=sf.read(b['reference_path'],dtype='float64')[0]
        dx=float(np.abs(x-y).max());ds=float(np.abs(s-t).max());assert max(dx,ds)<=plan['RT0_control_max_abs']
        controls.append({'old_sample_id':a['old_G18_sample_id'],'input_max_abs':dx,'reference_max_abs':ds})
    saved={r['old_sample_id']:r for r in json.loads((OUT/'RT0_controls.json').read_text())}
    assert len(controls)==84 and all(r==saved[r['old_sample_id']] for r in controls)
    audit={'status':'ok','utc':now(),'cases':504,'rooms':2,'pairs':252,'RT0_pairs_recomputed':84,
           'shared_scale_groups':28,'stages':anchors,'manifest_sha256':done['manifest_sha256'],
           'SNR_recomputed_cases':len(errors),'max_SNR_error_db':max(errors),
           'max_RT0_input_abs':max(r['input_max_abs'] for r in controls),
           'max_RT0_reference_abs':max(r['reference_max_abs'] for r in controls),'no_model_or_ASR_results':True}
    dump(OUT/'DATA_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G22_双房间数据阶段验收.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G22 双房间数据阶段验收\n\n504条输入和参考通过SHA、波形格式、来源、SNR及配对检查；252对房间条件完全匹配。28个组跨两房间及全部RT/方向共用尺度，84对RT0负控制独立回读后通过预定一个PCM24 LSB上限。99个新实际RIR及原始来源均核验，临时保持唤醒已释放。\n\n尚无模型或ASR成绩。下一步应先冻结模型注册与31个盲系统的完整评估入口，保持所有三种子；真值统计只作诊断。新房间结论不外推到真实阵列或独立说话人。\n')
    archive('data_acceptance',report,[OUT/'DATA_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g22_data_accept.py',OUT/'data_power_helper.json'])
    dump(OUT/'status.json',{'stage':'data_accepted_models_not_started','utc':now()})


if __name__=='__main__':main()
