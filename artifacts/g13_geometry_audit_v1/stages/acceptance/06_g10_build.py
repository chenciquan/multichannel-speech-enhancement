#!/usr/bin/env python3
"""Fresh speakers, reused noise recordings, matched three-angle groups."""
import hashlib,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf
from g10_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive
from mcse.simulation import render_plane_wave,geometric_delays,speech_activity_mask,mix_at_snr
from mcse.spatial_noise import highpass40,direction
from g5_prepare import rotate


def main():
    assert (OUT/'stages/sources/archive_manifest.json').exists();plan=json.loads((OUT/'frozen_plan.json').read_text());start=time.monotonic()
    for name,h in json.loads((OUT/'PIPELINE_READY.json').read_text())['files'].items():assert sha(ROOT/name)==h
    for name in ('noisy','reference'):(BASE/name).mkdir(exist_ok=False)
    sources=read(BASE/'sources/speech_manifest.jsonl');assert len(sources)==16
    geom=ROOT/'configs/g1_geometry.json';assert sha(geom)==plan['geometry_sha256'];pos=json.loads(geom.read_text())['microphone_positions_m']
    registry=json.loads((ROOT/'data/g3_v1/sources/noise_manifest.json').read_text())+json.loads((ROOT/'artifacts/g7_unseen_noise_v1/noise_sources.json').read_text())
    noise={};noise_sources=[]
    for name in plan['noise_ids']:
        r=next(r for r in registry if r['noise_id']==name);assert sha(r['path'])==r['sha256'];a,sr=sf.read(r['path'],dtype='float64')
        assert sr==16000 and a.ndim==1 and len(a)>240*16000;noise[name]=highpass40(a);noise_sources.append(r)
    dump(OUT/'noise_sources.json',noise_sources,True);rng=np.random.default_rng(plan['seed']);rows=[];canonical=defaultdict(set)
    def save(r,variants,field,noise_id,relative,intervals,directions):
        scale=min(1.,.95/max(float(np.abs(a).max()) for _,x,s,_ in variants for a in (x,s)))
        for ta,x,s,gain in variants:
            sid=f'g10_{len(rows):04d}_{r["source_id"]}';xp=BASE/'noisy'/(sid+'.wav');sp=BASE/'reference'/(sid+'.wav')
            sf.write(xp,x*scale,16000,subtype='PCM_24');sf.write(sp,s[:,4]*scale,16000,subtype='PCM_24')
            xx,_=sf.read(xp,dtype='float32',always_2d=True);ss,_=sf.read(sp,dtype='float32');xh=hashlib.sha256(xx[:,4].tobytes()).hexdigest();sh=hashlib.sha256(ss.tobytes()).hexdigest()
            key=f'{r["source_id"]}|{noise_id}|{field}|{relative}|0';canonical[key].add((xh,sh))
            rows.append({'sample_id':sid,'source_id':r['source_id'],'speaker_id':r['speaker_id'],'transcript':r['transcript'],'split':'fresh_voice_confirmation',
              'condition':field,'noise_field':field,'noise_id':noise_id,'relative_angle':relative,'target_azimuth':ta,'noise_azimuth':ta+relative if relative is not None else None,
              'snr_db':None if field=='clean' else 0,'counterfactual_group':key,'noisy_path':str(xp),'reference_path':str(sp),'input_sha256':sha(xp),'reference_sha256':sha(sp),
              'raw_mic5_sha256':xh,'reference_samples_sha256':sh,'common_scale':scale,'noise_gain':gain,'noise_intervals':intervals,'base_noise_directions':[u.tolist() for u in directions]})
    for index,r in enumerate(sources):
        assert sha(BASE/'sources'/r['path'])==r['sha256'];mono,sr=sf.read(BASE/'sources'/r['path'],dtype='float64');assert sr==16000
        speech={ta:render_plane_wave(mono,geometric_delays(pos,direction(ta,10))) for ta in plan['target_azimuth_degrees']}
        for name in plan['noise_ids']:
            for field,relative,k in [('point',-80,1),('point',25,1),('multi8',None,8)]:
                clips=[];intervals=[]
                for j in range(k):
                    lo=int((30+210*j/k)*16000);hi=int((30+210*(j+1)/k)*16000)-len(mono);offset=int(rng.integers(lo,hi+1))
                    clips.append(noise[name][offset:offset+len(mono)]);intervals.append([offset,offset+len(mono)])
                directions=[direction(20+relative,-15)] if k==1 else [rng.normal(size=3) for _ in clips]
                directions=[u/np.linalg.norm(u) for u in directions];variants=[]
                for ta,s in speech.items():
                    nf=np.zeros_like(s)
                    for clip,u in zip(clips,directions):nf+=render_plane_wave(clip,geometric_delays(pos,rotate(u,ta-20)))/np.sqrt(k)
                    x,_,gain=mix_at_snr(s,nf,0,speech_activity_mask(s[:,4]));variants.append((ta,x,s,gain))
                save(r,variants,field,name,relative,intervals,directions)
        save(r,[(ta,s,s,0.) for ta,s in speech.items()],'clean','CLEAN',None,[],[])
        status(stage='render',sources_done=index+1,total_sources=16,cases=len(rows))
        if time.monotonic()-start>plan['max_data_seconds']-120:raise TimeoutError('Data build reserve')
    assert len(rows)==624 and len(canonical)==208 and all(len(v)==1 for v in canonical.values())
    with (BASE/'manifest.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    assert sha(OUT/'systems.json')==plan['systems_sha256']
    report=ROOT/'docs/G10B_新说话人匹配模拟协议与数据.md';report.write_text('# G10B 新说话人匹配模拟协议与数据\n\n源选择报告和排除证据先保存，当前生成16句/4位全新项目说话人的匹配模拟条件。四个既有噪声OOFFICE/PCAFETER/NPARK/PSTATION，无新噪声下载。新片段不是新的独立噪声录音。\n\n每句四噪声×三空间条件×三目标方向(−40/20/70°)，0dB，共576带噪条件；另48干净输入，合计624例/系统。名义六麦自由场、40Hz高通后活动语音SNR、整场景旋转、成组共享防削波尺度；208组均断言第5麦带噪与干净参考跨方向逐样本相同。不是实测混响或硬件阵列。\n\n冻结15系统：带噪、原McNet、G5无干净约束分支固定/旋转全部三种子(共8直接输出)；原模型和六适配器各自的估计掩码Souden MVDR(7系统)。适配器按原G5验证已选，未用新语音选权重/种子，当前无训练。无oracle主对照。\n\n后续保存直接信号报告，再保存MVDR报告，再做Whisper及CTC。主问题为输出形式×训练方向分布的交互；所有方向、噪声、种子、干净对照和不利结果保留。只有4位独立说话人，按说话人作探索性bootstrap并逐人报告，不能把9,360条件记录当独立句子。\n\n来源、逐文件哈希、原权重及系统登记、代码/协议快照留存。下一步不追加依赖、训练或新权重。\n\nUTC：'+now()+'\n',encoding='utf-8')
    dump(OUT/'PREPARATION_COMPLETE.json',{'status':'ok','cases':624,'sources':16,'speakers':4,'counterfactual_groups':208,'all_equal':True,'manifest_sha256':sha(BASE/'manifest.jsonl'),'completed_utc':now()},True)
    archive('preparation',report,[OUT/'frozen_plan.json',OUT/'systems.json',OUT/'noise_sources.json',OUT/'PREPARATION_COMPLETE.json',BASE/'manifest.jsonl',OUT/'analysis_unit_tests.log',ROOT/'tests/test_g10_analysis.py',ROOT/'scripts/g10_finish.py',Path(__file__)])


if __name__=='__main__':main()
