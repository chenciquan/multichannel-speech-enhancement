"""Full coverage and exact zero-reflection output controls for matched rooms."""
import json
from collections import defaultdict
import numpy as np
import soundfile as sf
from g22_common import ROOT,OUT,BASE,sha,read,dump,now,archive,metadata,guard,status


def main():
    ready=guard();ev=OUT/'evaluation';rows=[]
    for phase in ('direct','blind','truth'):
        assert (OUT/'stages'/('signal_'+phase)/'archive_manifest.json').exists()
        marker=metadata(ev/(phase.upper()+'_COMPLETE.json'));path=ev/(phase+'_signals.jsonl')
        assert marker['status']=='ok' and sha(path)==marker['signals_sha256'];rows+=read(path)
    cases={c['sample_id']:c for c in read(BASE/'manifest.jsonl')}
    registry=metadata(OUT/'signal_systems.json');frontends=[s['system'] for s in registry['direct'] if s['kind']!='noisy']
    names=['noisy']+frontends+[n+'_mask_'+m for n in frontends for m in ('mvdr','mwf')]
    expected={(n,s) for n in names for s in cases}|{('truth_covariance_'+m,s) for s,c in cases.items() if c['condition']!='clean' for m in ('mvdr','mwf')}
    index={(r['system'],r['sample_id']):r for r in rows};assert len(rows)==len(index)==16488 and set(index)==expected
    group=defaultdict(dict)
    for r in rows:
        assert r['status']=='ok' and all(r[k]==v for k,v in cases[r['sample_id']].items())
        assert sha(r['output_path'])==r['output_sha256']
        if r['requested_rt60']==0:group[r['system'],r['matched_room_pair']][r['room_id']]=r
    controls=[]
    for (system,pair),entries in group.items():
        assert set(entries)=={'reference','new'}
        a,ar=sf.read(entries['reference']['output_path'],dtype='float32');b,br=sf.read(entries['new']['output_path'],dtype='float32')
        assert ar==br==16000 and a.shape==b.shape
        error=float(np.abs(a.astype(float)-b.astype(float)).max());assert error<=ready['RT0_output_max_abs']
        controls.append({'system':system,'matched_room_pair':pair,'max_abs':error,'exact':bool(np.array_equal(a,b))})
    assert len(controls)==2748
    with (ev/'combined_signals.jsonl').open('xb') as target:
        for phase in ('direct','blind','truth'):
            with (ev/(phase+'_signals.jsonl')).open('rb') as src:
                for block in iter(lambda:src.read(1024*1024),b''):target.write(block)
    dump(ev/'RT0_output_controls.json',controls,True)
    summary=[]
    for room in ('reference','new'):
        for rt in (0,.15,.35):
            for name in sorted({r['system'] for r in rows}):
                subset=[r for r in rows if r['room_id']==room and r['requested_rt60']==rt and r['system']==name]
                for clean in (False,True):
                    selected=[r for r in subset if (r['condition']=='clean')==clean]
                    if not selected:continue
                    record={'room_id':room,'rt':rt,'system':name,'clean':clean,'rows':len(selected),
                            **{k:float(np.mean([r['metrics'][k] for r in selected])) for k in ('si_sdr_db','pesq_wb','stoi')}}
                    if clean:record['levels']={k:float(np.mean([r['clean_levels'][k] for r in selected])) for k in ('rms_gain_ratio','reference_normalized_squared_error')}
                    summary.append(record)
    dump(ev/'signal_summary.json',summary,True)
    complete={'status':'ok','completed_utc':now(),'rows':len(rows),'RT0_pairs':len(controls),
              'RT0_exact_pairs':sum(c['exact'] for c in controls),'RT0_max_abs':max(c['max_abs'] for c in controls),
              'combined_sha256':sha(ev/'combined_signals.jsonl'),'summary_sha256':sha(ev/'signal_summary.json')}
    dump(ev/'SIGNAL_COMPLETE.json',complete,True)
    report=ROOT/'docs/G22_完整信号阶段验收报告.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G22 完整信号阶段验收报告\n\n完成16488条信号：5544直接输出、10080估计掩码滤波、864真值统计诊断。全部条件键唯一且覆盖完整，所有输出哈希通过。2748对RT0输出独立生成后检查，最大差 '+f"{complete['RT0_max_abs']:.3g}"+'，其中 '+str(complete['RT0_exact_pairs'])+' 对逐样本完全相同；预定上限为1e−5。\n\n按房间、请求RT、系统及干净/带噪分开保存信号汇总；干净控制另有RMS和NMSE。所有训练策略与三个种子保留，未调参或新增训练。真值统计不可部署，也不是理论上界。当前仅信号验收，尚未对本轮新输出完成ASR，不据此认定内容识别改善。\n')
    archive('signal_completion',report,[ev/'SIGNAL_COMPLETE.json',ev/'signal_summary.json',ev/'RT0_output_controls.json',ROOT/'scripts/g22_signal_finish.py'])
    status(stage='signal_complete_pending_readback')


if __name__=='__main__':main()
