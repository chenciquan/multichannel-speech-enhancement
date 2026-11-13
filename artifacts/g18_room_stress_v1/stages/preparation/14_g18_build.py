#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
from collections import Counter
import numpy as np
import soundfile as sf
import rir_generator as rir
from g18_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive
from mcse.reverberation import full_speech,continuous_noise,schroeder_t20
from mcse.simulation import speech_activity_mask,mix_at_snr,snr_db
from mcse.spatial_noise import direction,highpass40
from mcse.geometry import rotate_azimuth as rotate,azimuth_degrees,wrap_degrees


def main():
    assert (OUT/'stages/physics/archive_manifest.json').exists();plan=json.loads((OUT/'frozen_plan.json').read_text());start=time.monotonic()
    for name,h in json.loads((OUT/'PIPELINE_READY.json').read_text())['files'].items():assert sha(ROOT/name)==h
    BASE.mkdir(exist_ok=False)
    for name in ('noisy','reference','rir'):(BASE/name).mkdir()
    assert sha(ROOT/'data/g17_v1/sources/speech_manifest.jsonl')==plan['G17_sources_manifest_sha256']
    sources=json.loads((OUT/'selected_sources.json').read_text());assert len(sources)==8
    pos=np.asarray(json.loads((ROOT/'configs/g1_geometry.json').read_text())['microphone_positions_m']);assert sha(ROOT/'configs/g1_geometry.json')==plan['geometry_sha256']
    ref=np.array(plan['reference_mic4_position_m']);receivers=ref+(pos-pos[4]);room=np.array(plan['room_dimensions_m']);rng=np.random.default_rng(plan['seed'])
    # One frozen eight-source field, shared by all clips and paired RT conditions.
    multi=[direction(float(rng.uniform(-180,180)),float(rng.uniform(-30,30))) for _ in range(8)]
    dump(OUT/'field_directions.json',{'base_azimuth':20,'multi8':[u.tolist() for u in multi],'notice':'Fixed once before all room scores; not a measured diffuse field'},True)
    kernels={};rir_rows=[]
    def response(rt,u):
        key=(float(rt),tuple(np.round(u,12)))
        if key not in kernels:
            source=ref+plan['source_distance_m']*u;assert ((source>0)&(source<room)).all()
            kw={'beta':np.zeros(6)} if rt==0 else {'reverberation_time':rt}
            h=rir.generate(c=343,fs=16000,r=receivers,s=source,L=room,nsample=7680,mtype=rir.mtype.omnidirectional,order=-1,dim=3,hp_filter=False,**kw)*(4*np.pi*1.5)
            assert h.shape==(7680,6) and np.isfinite(h).all()
            path=BASE/'rir'/f'{len(kernels):03d}.npy';np.save(path,h);kernels[key]=(h,str(path))
            rir_rows.append({'requested_rt60':rt,'source_direction':u.tolist(),'source_position_m':source.tolist(),'path':str(path),'sha256':sha(path),
               'measured_T20_extrapolated_RT60':schroeder_t20(h[:,4]) if rt else None,'tail_fraction20ms':float(np.sum(h[320:,4]**2)/np.sum(h[:,4]**2)),
               'truncated_480ms_last20ms_energy_fraction':float(np.sum(h[-320:,4]**2)/np.sum(h[:,4]**2))})
            if len(kernels)%12==0:status(stage='RIR_generation',responses_done=len(kernels),responses_total=99)
        return kernels[key]
    registry=json.loads((ROOT/'data/g3_v1/sources/noise_manifest.json').read_text())+json.loads((ROOT/'artifacts/g7_unseen_noise_v1/noise_sources.json').read_text())
    noise={};used=[]
    for name in plan['noise_ids']:
        r=next(r for r in registry if r['noise_id']==name);assert sha(r['path'])==r['sha256'];a,sr=sf.read(r['path'],dtype='float64');assert sr==16000 and a.ndim==1 and len(a)>295*16000
        noise[name]=highpass40(a);used.append(r)
    dump(OUT/'noise_sources.json',used,True);rows=[];snr_errors=[]
    def save(source,variants,field,nid,relative,intervals,dirs):
        scale=min(1.,.95/max(float(np.abs(a).max()) for _,_,x,s,_,_,_ in variants for a in (x,s)))
        group=f'{source["source_id"]}|{nid}|{field}|{relative}'
        for rt,angle,x,s,gain,srir,nrirs in variants:
            sid=f'g18_{len(rows):04d}_{source["source_id"]}';xp=BASE/'noisy'/(sid+'.wav');sp=BASE/'reference'/(sid+'.wav')
            with xp.open('xb') as f:sf.write(f,x*scale,16000,format='WAV',subtype='PCM_24')
            with sp.open('xb') as f:sf.write(f,s[:,4]*scale,16000,format='WAV',subtype='PCM_24')
            xx,sr=sf.read(xp,dtype='float64',always_2d=True);ss,rr=sf.read(sp,dtype='float64');assert sr==rr==16000 and xx.shape==(len(ss),6) and len(ss)<64000
            mask=None if field=='clean' else speech_activity_mask(s[:,4])
            err=None if mask is None else abs(snr_db(ss,xx[:,4]-ss,mask))
            recomputed=None if mask is None else speech_activity_mask(ss)
            recomputed_err=None if mask is None else abs(snr_db(ss,xx[:,4]-ss,recomputed))
            mask_diff=0 if mask is None else int(np.count_nonzero(mask!=recomputed))
            spans=[] if mask is None else np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))).reshape(-1,2).tolist()
            if err is not None:assert err<plan['SNR_saved_audit_tolerance_db'];snr_errors.append(err)
            rows.append({'sample_id':sid,'source_id':source['source_id'],'speaker_id':source['speaker_id'],'transcript':source['transcript'],'split':'room_stress_reused_voices',
               'condition':field,'noise_field':field,'noise_id':nid,'relative_angle':relative,'target_azimuth':angle,'requested_rt60':rt,'snr_db':None if field=='clean' else 0,
               'paired_room_group':group,'noisy_path':str(xp),'reference_path':str(sp),'input_sha256':sha(xp),'reference_sha256':sha(sp),'common_scale':scale,'noise_gain':gain,
               'frames':len(ss),'dry_frames':source['samples'],'RIR_samples':7680,'noise_preroll_samples':16000,'noise_intervals':intervals,'base_noise_directions':[u.tolist() for u in dirs],
               'speech_RIR_path':srir,'noise_RIR_paths':nrirs,'saved_active_SNR_error_db':err,'activity_spans_from_unquantized_reference':spans,'recomputed_PCM_activity_difference_samples':mask_diff,'saved_recomputed_PCM_activity_SNR_error_db':recomputed_err})
    for ix,r in enumerate(sources):
        path=ROOT/'data/g17_v1/sources'/r['path'];assert sha(path)==r['sha256'];mono,sr=sf.read(path,dtype='float64');assert sr==16000 and len(mono)==r['samples']
        L=len(mono)+7679;speech={}
        for rt in plan['requested_RT60_seconds']:
            for angle in plan['target_angles']:
                h,p=response(rt,direction(angle,10));speech[(rt,angle)]=(full_speech(mono,h),p)
        for nid in plan['noise_ids']:
            for field,relative,k in [('point',-80,1),('point',25,1),('multi8',None,8)]:
                clips=[];intervals=[]
                for j in range(k):
                    lo=int((250+45*j/k)*16000);hi=int((250+45*(j+1)/k)*16000)-(16000+L);offset=int(rng.integers(lo,hi+1))
                    clips.append(noise[nid][offset:offset+16000+L]);intervals.append([offset,offset+16000+L])
                dirs=[direction(20+relative,-15)] if k==1 else multi;variants=[]
                for (rt,angle),(s,srir) in speech.items():
                    nf=np.zeros_like(s);nrirs=[]
                    for clip,u in zip(clips,dirs):
                        moved=rotate(u,angle-20);assert abs(moved@direction(angle,10)-u@direction(20,10))<1e-12
                        if k==1:assert abs(wrap_degrees(azimuth_degrees(moved)-angle-relative))<1e-10
                        h,p=response(rt,moved);nf+=continuous_noise(clip,h,16000,L)/np.sqrt(k);nrirs.append(p)
                    x,_,gain=mix_at_snr(s,nf,0,speech_activity_mask(s[:,4]));variants.append((rt,angle,x,s,gain,srir,nrirs))
                save(r,variants,field,nid,relative,intervals,dirs)
        save(r,[(rt,a,s,s,0.,p,[]) for (rt,a),(s,p) in speech.items()],'clean','CLEAN',None,[],[])
        status(stage='render',sources_done=ix+1,total_sources=8,cases=len(rows),unique_RIRs=len(kernels))
        if time.monotonic()-start>plan['max_data_seconds']-90:raise TimeoutError('Room build reserve')
    assert len(rows)==504 and sum(r['condition']=='clean' for r in rows)==72 and len(rir_rows)==99
    assert all(n==9 for n in Counter(r['paired_room_group'] for r in rows).values())
    with (BASE/'manifest.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    dump(OUT/'RIR_manifest.json',rir_rows,True)
    done={'status':'ok','completed_utc':now(),'cases':504,'noisy_cases':432,'clean_cases':72,'reused_sources':8,'speakers':4,'RIRs':99,
        'manifest_sha256':sha(BASE/'manifest.jsonl'),'max_saved_active_SNR_error_db':max(snr_errors),'max_frames':max(r['frames'] for r in rows),'elapsed_seconds':time.monotonic()-start}
    dump(OUT/'PREPARATION_COMPLETE.json',done,True)
    mask_changes=sum(r['recomputed_PCM_activity_difference_samples']>0 for r in rows)
    lines=['# G18B 单房间反射压力数据与几何协议','','G18A物理/生成器/容量报告归档后，复用G17的完整8句/4人，保留全部有限7680点响应全卷积尾部；不是新增独立说话人。参考为第5麦混响语音，只衡量降噪，不要求干声去混响。',
      '', '单间6×5×3m，源距1.5m，正确同向转动源星座，固定墙体不转；直达源相对角/距离保持，反射场及参考波形会变。固定全局8源星座共享所有语句/噪声/RT，可计算99个响应；其星座不同G17逐组随机场。因此RT0是本轮近场无反射内部控制，不是G17平面波的精确重跑。',
      '', '请求RT0/0.15/0.35秒不等于各位置实测T60；99响应保存源位置、SHA、T20外推及末20ms能量比例。0.48秒是有限响应窗，保留整个有限滤波器卷积，不声称自然无限尾部全部保存。',
      '', 'OOFFICE/NPARK旧录音，噪声40Hz高通，250—295秒取连续段，1秒预卷积消除起始过渡，背景覆盖整个语音/尾部。两点相对角−80/+25与八源场、三目标方向均保留。跨全部RT/角度共享片段和共同防削波尺度；不误断言房间参考跨方向逐样本相等。',
      '', '活动SNR=0dB，沿用混音时未量化混响参考的固定活动帧，存下其区间；PCM24保存后用同一mask核验，最大误差'+f'{max(snr_errors):.6f}'+'dB。另重算PCM活动帧作为诊断，不静默替换原SNR约定；活动样本有变化的案例'+str(mask_changes)+'条。',
      '', '432带噪+72干净=504条件/系统，原21冻结系统=10584信号及各ASR评分，全部三种子/末步3000不变。单房间、已查看4人、旧两录音，不代表真实阵列、多房间或中文。完整数据/代码先归档，再直接/MVDR/Whisper/CTC依次评估。','', 'UTC：'+now()]
    report=ROOT/'docs/G18B_单房间反射压力数据与几何协议.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('preparation',report,[OUT/'frozen_plan.json',OUT/'systems.json',OUT/'PIPELINE_READY.json',OUT/'field_directions.json',OUT/'noise_sources.json',OUT/'selected_sources.json',OUT/'RIR_manifest.json',OUT/'PREPARATION_COMPLETE.json',BASE/'manifest.jsonl']+[ROOT/n for n in json.loads((OUT/'PIPELINE_READY.json').read_text())['files']])



if __name__=='__main__':main()
