#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import hashlib,json,time
from collections import defaultdict,Counter
import numpy as np
import soundfile as sf
from g14_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive
from mcse.geometry import rotate_azimuth,azimuth_degrees,wrap_degrees
from mcse.simulation import render_plane_wave,geometric_delays,speech_activity_mask,mix_at_snr,snr_db
from mcse.components import pcm24_field
from mcse.spatial_noise import highpass40,direction
from g5_prepare import rotate as legacy_rotate


def main():
    plan=json.loads((OUT/'frozen_plan.json').read_text());start=time.monotonic();BASE.mkdir(exist_ok=False)
    for name in ('noisy','reference'):(BASE/name).mkdir()
    assert sha(ROOT/'data/g10_v1/manifest.jsonl')==plan['G10_manifest_sha256'];assert sha(ROOT/'configs/g1_geometry.json')==plan['geometry_sha256']
    sources={r['source_id']:r for r in json.loads((OUT/'selected_sources.json').read_text())};assert len(sources)==8
    original=read(ROOT/'data/g10_v1/manifest.jsonl');selected=[r for r in original if r['source_id'] in sources and (r['noise_id'] in plan['noise_ids'] or r['condition']=='clean')];assert len(selected)==168
    groups=defaultdict(list)
    for r in selected:groups[r['counterfactual_group']].append(r)
    assert len(groups)==56 and all(len(v)==3 for v in groups.values())
    pos=json.loads((ROOT/'configs/g1_geometry.json').read_text())['microphone_positions_m'];registry=json.loads((ROOT/'artifacts/g10_fresh_voice_v1/noise_sources.json').read_text());noise={}
    for nid in plan['noise_ids']:
        r=next(r for r in registry if r['noise_id']==nid);assert sha(r['path'])==r['sha256'];a,sr=sf.read(r['path'],dtype='float64');assert sr==16000;noise[nid]=highpass40(a)
    rows=[];canonical=defaultdict(set);negative={};old_exact=0;rescaled=0
    speechcache={}
    for ix,(group,oldcases) in enumerate(groups.items()):
        r=oldcases[0];source=sources[r['source_id']]
        if r['source_id'] not in speechcache:
            p=ROOT/'data/g10_v1/sources'/source['path'];assert sha(p)==source['sha256'];mono,sr=sf.read(p,dtype='float64');assert sr==16000 and len(mono)==source['samples'];speechcache[r['source_id']]=mono
        mono=speechcache[r['source_id']];clean=r['condition']=='clean';oldmap={a['target_azimuth']:a for a in oldcases};assert set(oldmap)=={-40,20,70}
        assert len({a['common_scale'] for a in oldcases})==1;oldscale=r['common_scale'];variants=[]
        if not clean:
            clips=[noise[r['noise_id']][lo:hi] for lo,hi in r['noise_intervals']];ds=[np.array(u) for u in r['base_noise_directions']]
            assert all(np.array_equal(a['base_noise_directions'],r['base_noise_directions']) and a['noise_intervals']==r['noise_intervals'] for a in oldcases)
            assert all(len(c)==len(mono) for c in clips)
        for protocol in plan['geometry_protocols']:
            for target in plan['target_angles']:
                s=render_plane_wave(mono,geometric_delays(pos,direction(target,10)));actual=[]
                if clean:x=s;gain=0.
                else:
                    nf=np.zeros_like(s);rotate=legacy_rotate if protocol=='legacy_counterrotation' else rotate_azimuth
                    for clip,u in zip(clips,ds):
                        moved=rotate(u,target-20);actual.append(moved.tolist());nf+=render_plane_wave(clip,geometric_delays(pos,moved))/np.sqrt(len(clips))
                        if protocol=='rigid_rotation':assert abs(np.dot(moved,direction(target,10))-np.dot(u,direction(20,10)))<1e-12
                    x,_,gain=mix_at_snr(s,nf,0,speech_activity_mask(s[:,4]))
                if protocol=='legacy_counterrotation':
                    prior=oldmap[target];saved,sr=sf.read(prior['noisy_path'],dtype='float32',always_2d=True);assert sha(prior['noisy_path'])==prior['input_sha256'] and np.array_equal(pcm24_field(x*oldscale),saved)
                variants.append((protocol,target,x,s,gain,actual))
        scale=min(oldscale,.95/max(float(np.abs(a).max()) for _,_,x,s,_,_ in variants for a in (x,s)));rescaled+=scale!=oldscale
        for protocol,target,x,s,gain,actual in variants:
            sid=f'g14_{len(rows):04d}_{r["source_id"]}';xp=BASE/'noisy'/(sid+'.wav');sp=BASE/'reference'/(sid+'.wav')
            with xp.open('xb') as f:sf.write(f,x*scale,16000,format='WAV',subtype='PCM_24')
            with sp.open('xb') as f:sf.write(f,s[:,4]*scale,16000,format='WAV',subtype='PCM_24')
            xx,sr=sf.read(xp,dtype='float32',always_2d=True);ss,rr=sf.read(sp,dtype='float32');assert sr==rr==16000
            canonical[group].add((hashlib.sha256(xx[:,4].tobytes()).hexdigest(),hashlib.sha256(ss.tobytes()).hexdigest()))
            if target==20 or clean:
                key=(group,target)
                if key in negative:assert np.array_equal(negative[key],xx)
                else:negative[key]=xx
            oldrow=oldmap[target];unchanged=sha(xp)==oldrow['input_sha256'];old_exact+=protocol=='legacy_counterrotation' and unchanged
            actualrel=None if clean or len(actual)!=1 else wrap_degrees(azimuth_degrees(actual[0])-target)
            if not clean and len(actual)==1 and protocol=='rigid_rotation':assert abs(wrap_degrees(actualrel-r['relative_angle']))<1e-10
            error=None if clean else abs(snr_db(ss,xx[:,4]-ss,speech_activity_mask(ss)))
            if error is not None:assert error<.002
            rows.append({'sample_id':sid,'source_id':r['source_id'],'speaker_id':r['speaker_id'],'transcript':r['transcript'],'split':'geometry_correction_diagnostic',
                'condition':r['condition'],'noise_field':r['condition'],'noise_id':r['noise_id'],'snr_db':None if clean else 0,'relative_angle':r['relative_angle'],
                'relative_angle_notice':'Baseline label; use actual_relative_azimuth for legacy counterrotation','actual_relative_azimuth':actualrel,'target_azimuth':target,'geometry_protocol':protocol,
                'geometry_pair':group+'|'+str(target),'counterfactual_group':group,'noisy_path':str(xp),'reference_path':str(sp),'input_sha256':sha(xp),'reference_sha256':sha(sp),
                'common_scale':scale,'old_G10_scale':oldscale,'noise_gain':gain,'base_noise_directions':r.get('base_noise_directions',[]),'actual_noise_directions':actual,'noise_intervals':r.get('noise_intervals',[]),
                'old_G10_sample_id':oldrow['sample_id'],'old_G10_input_sha256':oldrow['input_sha256'],'matches_old_G10_input':unchanged,'frames':len(ss),'saved_active_SNR_error_db':error})
        if (ix+1)%8==0:status(stage='render',groups_done=ix+1,total_groups=56,cases=len(rows))
        if time.monotonic()-start>plan['max_data_seconds']-60:raise TimeoutError('G14data reserve')
    assert len(rows)==336 and len(canonical)==56 and all(len(v)==1 for v in canonical.values()) and sum(r['condition']=='clean' for r in rows)==48
    with (BASE/'manifest.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    done={'status':'ok','completed_utc':now(),'cases':336,'noisy':288,'clean':48,'all168_old_inputs_reproduced_before_joint_scaling':True,
        'legacy_after_joint_scaling_still_exact_G10_cases':old_exact,'joint_groups_rescaled':rescaled,'matched_reference_groups':56,'all_reference_mic_and_target_samples_equal':True,
        'target20_and_clean_negative_controls_exact':True,'manifest_sha256':sha(BASE/'manifest.jsonl'),'elapsed_seconds':time.monotonic()-start}
    dump(OUT/'PREPARATION_COMPLETE.json',done,True)
    report=ROOT/'docs/G14A_正确几何配对自由场协议与数据.md';report.write_text('# G14A 正确几何配对自由场协议与数据\n\nG13坐标审计及96项测试验收后冻结。复用预先选定的G10完整8句/4人、OOFFICE/NPARK的原噪声片段和八源方向；无新来源、训练、权重或依赖。旧模型rotated登记名保留，实际训练为legacy counter-rotation，不称正确旋转训练。\n\n旧反向旋转与正确刚性旋转两协议，各144带噪+24干净，共336条件/系统、15系统5040对。源与噪声共同正确转动时，所有目标/噪声方向点积保持，点源实际相对角−80/+25恒定；旧协议实际角另写字段，旧relative_angle只作基准标签。\n\n56组跨两协议/三目标角共享一个防削波尺度、噪声片段及完全一致的参考第5麦输入/干净语音。20°两协议全部六通道逐样本相同，所有干净输入两协议也相同，作为负控制。168个旧输入在共同重新缩放前全部逐样本复现；共同缩放后仍与G10完全相同的legacy输入'+str(old_exact)+'个，需新共同缩放组'+str(rescaled)+'个。所有输出将新跑，双ASR仅按同模型和样本SHA缓存，不能按成绩选缓存。\n\n自由场名义六麦，完整语句加既有128样本FIR保护，无混响。修正的是测试几何，训练策略未修正；此实验不能证明正确旋转重训收益，也不当新独立最终测试。旧数据/模型/评分及G12中止数据全部保留。数据报告归档后才运行直接模型，再MVDR、Whisper、CTC。\n\nUTC：'+now()+'\n',encoding='utf-8')
    files=[OUT/'frozen_plan.json',OUT/'PIPELINE_READY.json',OUT/'systems.json',OUT/'selected_sources.json',OUT/'analysis_test_observation.json',OUT/'PREPARATION_COMPLETE.json',BASE/'manifest.jsonl']+[ROOT/name for name in json.loads((OUT/'PIPELINE_READY.json').read_text())['files']]
    archive('preparation',report,files)


if __name__=='__main__':main()
