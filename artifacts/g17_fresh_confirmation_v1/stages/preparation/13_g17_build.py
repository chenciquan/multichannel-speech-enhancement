#!/usr/bin/env python3
"""Fresh speakers, reused noise recordings, matched three-angle groups."""
import hashlib,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf
from g17_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive
from mcse.simulation import render_plane_wave,geometric_delays,speech_activity_mask,mix_at_snr
from mcse.spatial_noise import highpass40,direction
from mcse.geometry import rotate_azimuth as rotate,azimuth_degrees,wrap_degrees


def main():
    assert (OUT/'stages/sources/archive_manifest.json').exists();plan=json.loads((OUT/'frozen_plan.json').read_text());start=time.monotonic()
    for name,h in json.loads((OUT/'PIPELINE_READY.json').read_text())['files'].items():assert sha(ROOT/name)==h
    for name in ('noisy','reference'):(BASE/name).mkdir(exist_ok=False)
    sources=read(BASE/'sources/speech_manifest.jsonl');assert len(sources)==8
    geom=ROOT/'configs/g1_geometry.json';assert sha(geom)==plan['geometry_sha256'];pos=json.loads(geom.read_text())['microphone_positions_m']
    registry=json.loads((ROOT/'data/g3_v1/sources/noise_manifest.json').read_text())+json.loads((ROOT/'artifacts/g7_unseen_noise_v1/noise_sources.json').read_text())
    noise={};noise_sources=[]
    for name in plan['noise_ids']:
        r=next(r for r in registry if r['noise_id']==name);assert sha(r['path'])==r['sha256'];a,sr=sf.read(r['path'],dtype='float64')
        assert sr==16000 and a.ndim==1 and len(a)>295*16000;noise[name]=highpass40(a);noise_sources.append(r)
    dump(OUT/'noise_sources.json',noise_sources,True);rng=np.random.default_rng(plan['seed']);rows=[];canonical=defaultdict(set)
    def save(r,variants,field,noise_id,relative,intervals,directions):
        scale=min(1.,.95/max(float(np.abs(a).max()) for _,x,s,_ in variants for a in (x,s)))
        for ta,x,s,gain in variants:
            sid=f'g17_{len(rows):04d}_{r["source_id"]}';xp=BASE/'noisy'/(sid+'.wav');sp=BASE/'reference'/(sid+'.wav')
            sf.write(xp,x*scale,16000,subtype='PCM_24');sf.write(sp,s[:,4]*scale,16000,subtype='PCM_24')
            xx,_=sf.read(xp,dtype='float32',always_2d=True);ss,_=sf.read(sp,dtype='float32');xh=hashlib.sha256(xx[:,4].tobytes()).hexdigest();sh=hashlib.sha256(ss.tobytes()).hexdigest()
            key=f'{r["source_id"]}|{noise_id}|{field}|{relative}|0';canonical[key].add((xh,sh))
            rows.append({'sample_id':sid,'source_id':r['source_id'],'speaker_id':r['speaker_id'],'transcript':r['transcript'],'split':'fresh_voice_confirmation','geometry_protocol':'rigid_rotation','actual_relative_azimuth':relative,
              'condition':field,'noise_field':field,'noise_id':noise_id,'relative_angle':relative,'target_azimuth':ta,'noise_azimuth':ta+relative if relative is not None else None,
              'snr_db':None if field=='clean' else 0,'counterfactual_group':key,'noisy_path':str(xp),'reference_path':str(sp),'input_sha256':sha(xp),'reference_sha256':sha(sp),
              'raw_mic5_sha256':xh,'reference_samples_sha256':sh,'common_scale':scale,'noise_gain':gain,'noise_intervals':intervals,'base_noise_directions':[u.tolist() for u in directions]})
    for index,r in enumerate(sources):
        assert sha(BASE/'sources'/r['path'])==r['sha256'];mono,sr=sf.read(BASE/'sources'/r['path'],dtype='float64');assert sr==16000
        speech={ta:render_plane_wave(mono,geometric_delays(pos,direction(ta,10))) for ta in plan['target_angles']}
        for name in plan['noise_ids']:
            for field,relative,k in [('point',-80,1),('point',25,1),('multi8',None,8)]:
                clips=[];intervals=[]
                for j in range(k):
                    lo=int((250+45*j/k)*16000);hi=int((250+45*(j+1)/k)*16000)-len(mono);offset=int(rng.integers(lo,hi+1))
                    clips.append(noise[name][offset:offset+len(mono)]);intervals.append([offset,offset+len(mono)])
                directions=[direction(20+relative,-15)] if k==1 else [rng.normal(size=3) for _ in clips]
                directions=[u/np.linalg.norm(u) for u in directions];variants=[]
                for ta,s in speech.items():
                    nf=np.zeros_like(s)
                    for clip,u in zip(clips,directions):
                        moved=rotate(u,ta-20);assert abs(moved@direction(ta,10)-u@direction(20,10))<1e-12
                        if k==1:assert abs(wrap_degrees(azimuth_degrees(moved)-ta-relative))<1e-10
                        nf+=render_plane_wave(clip,geometric_delays(pos,moved))/np.sqrt(k)
                    x,_,gain=mix_at_snr(s,nf,0,speech_activity_mask(s[:,4]));variants.append((ta,x,s,gain))
                save(r,variants,field,name,relative,intervals,directions)
        save(r,[(ta,s,s,0.) for ta,s in speech.items()],'clean','CLEAN',None,[],[])
        status(stage='render',sources_done=index+1,total_sources=8,cases=len(rows))
        if time.monotonic()-start>plan['max_data_seconds']-120:raise TimeoutError('Data build reserve')
    assert len(rows)==168 and len(canonical)==56 and all(len(v)==1 for v in canonical.values())
    with (BASE/'manifest.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    assert sha(OUT/'systems.json')==plan['systems_sha256']
    done={'status':'ok','cases':168,'sources':8,'speakers':4,'counterfactual_groups':56,'all_reference_mic_and_target_groups_equal':True,'correct_rigid_geometry_invariants':True,'manifest_sha256':sha(BASE/'manifest.jsonl'),'completed_utc':now()}
    train=json.loads((ROOT/'data/g5_v1/rotated/train_catalog.json').read_text());overlap=0
    for r in rows:
        for lo,hi in r['noise_intervals']:
            overlap+=any(t['noise_id']==r['noise_id'] and any(max(lo,a)<min(hi,b) for a,b in t['noise_intervals']) for t in train)
    assert overlap==0
    dump(OUT/'PREPARATION_COMPLETE.json',done,True)
    dump(OUT/'noise_training_relationship.json',{'training_recording_ids':sorted({r['noise_id'] for r in train}),'current_noise_ids':plan['noise_ids'],'current_intervals':[250,295],'any_overlap_with_LoRAtraining_intervals':False,'notice':'OOFFICErecordingseeninLoRAtraining,NPARKabsentfromLoRAtrainingbutseeninprojectoldtests;originalMcNetpretrainingexposureunknown'},True)
    report=ROOT/'docs/G17B_新语音正确几何模拟协议与数据.md';report.write_text('# G17B 新语音正确几何模拟协议与数据\n\n源/排除记录先保存后，生成8句/4位项目新说话人的正确几何模拟：OOFFICE/NPARK旧录音，0dB，三目标角−40/20/70、两点源相对方位−80/+25及八源叠加；144带噪+24干净=168条件/系统，21系统3528条/后端。完整语句加128样本保护，不截字。\n\n采用已核查的rotate_azimuth；每个目标/噪声点积保持，点源实际相对角全部核验；56组第5麦带噪及干净参考跨目标方向逐样本相同。共用片段、防削波尺度；噪声40Hz高通后活动帧SNR。仅名义六麦自由场，不是房间混响或真实阵列。\n\n噪声片段固定在250—295秒、八源在互不重叠子窗选取，与G5/G15全部LoRA训练噪声片段不重叠；OOFFICE录音在适配训练中出现过，NPARK没有但已用于本项目旧测试。不是新的独立噪声录音，原McNet预训练曝光未知。\n\n冻结原模型、固定/反向/刚性各三种子(全部3000步)，并加10个对应成熟MVDR输出；不按新语音结果筛方法、种子或角度。来源源SHA/话人ID排除和方法冻结先于任何新评分；语句/条件/种子不能当数千个独立样本。后续按方向、噪声、场型、每说话人/种子及干净控制保留负结果。\n\n数据报告保存后才推理，再按直接/MVDR/Whisper/CTC逐阶段归档。\n\nUTC：'+now()+'\n',encoding='utf-8')
    archive('preparation',report,[OUT/'frozen_plan.json',OUT/'PIPELINE_READY.json',OUT/'systems.json',OUT/'noise_sources.json',OUT/'PREPARATION_COMPLETE.json',OUT/'noise_training_relationship.json',BASE/'manifest.jsonl']+[ROOT/n for n in json.loads((OUT/'PIPELINE_READY.json').read_text())['files']])



if __name__=='__main__':main()
