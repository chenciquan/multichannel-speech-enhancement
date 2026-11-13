#!/usr/bin/env python3
"""Regenerate only geometry, prove every old training/validation input first."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,shutil,time
import numpy as np
import soundfile as sf
from g15_common import ROOT,OUT,BASE,now,sha,dump,read,status,archive
from mcse.geometry import rotate_azimuth
from mcse.spatial_noise import direction,highpass40
from mcse.simulation import render_plane_wave,geometric_delays,mix_at_snr,speech_activity_mask
from mcse.components import pcm24_field
from g5_prepare import rotate as legacy_rotate


def main():
    plan=json.loads((OUT/'frozen_plan.json').read_text());start=time.monotonic();BASE.mkdir(exist_ok=False);(BASE/'validation').mkdir()
    old=ROOT/'data/g5_v1/rotated';catalog=json.loads((old/'train_catalog.json').read_text());assert len(catalog)==2048
    source={r['source_id']:r for r in read(ROOT/'data/g3_v1/sources/speech_manifest.jsonl')};sources={};noise={};registry=json.loads((ROOT/'data/g3_v1/sources/noise_manifest.json').read_text());pos=json.loads((ROOT/'configs/g1_geometry.json').read_text())['microphone_positions_m']
    def speech(source_id):
        if source_id not in sources:
            r=source[source_id];p=ROOT/'data/g3_v1/sources'/r['path'];assert sha(p)==r['sha256'];a,sr=sf.read(p,dtype='float64');assert sr==16000;sources[source_id]=a
        return sources[source_id]
    def background(name):
        if name not in noise:
            r=next(r for r in registry if r['noise_id']==name);assert sha(r['path'])==r['sha256'];a,sr=sf.read(r['path'],dtype='float64');assert sr==16000;noise[name]=highpass40(a)
        return noise[name]
    def render(mono,clips,ds,target,snr,correct):
        s=render_plane_wave(mono,geometric_delays(pos,direction(target,10)));nf=np.zeros_like(s);rotate=rotate_azimuth if correct else legacy_rotate
        for clip,u in zip(clips,ds):
            v=rotate(u,target-20)
            if correct:assert abs(v@direction(target,10)-np.asarray(u)@direction(20,10))<1e-12
            nf+=render_plane_wave(clip,geometric_delays(pos,v))/np.sqrt(len(clips))
        x,_,_=mix_at_snr(s,nf,snr,speech_activity_mask(s[:,4]));return x,s
    oldx=np.load(old/'train_x.npy',mmap_mode='r');oldy=np.load(old/'train_y.npy',mmap_mode='r');assert oldx.shape==(2048,6,16128)
    newx=np.lib.format.open_memmap(BASE/'train_x.npy',mode='w+',dtype='float32',shape=oldx.shape);newcat=[];maxpeak=0.;outside=0
    for i,r in enumerate(catalog):
        assert source[r['source_id']]['split']=='train';mono=speech(r['source_id'])[r['source_crop_start']:r['source_crop_start']+16000];assert len(mono)==16000
        clips=[background(r['noise_id'])[lo:hi] for lo,hi in r['noise_intervals']];ds=r['base_noise_directions'];assert all(len(c)==16000 for c in clips)
        legacy,s=render(mono,clips,ds,r['target_azimuth'],r['snr_db'],False);correct,_=render(mono,clips,ds,r['target_azimuth'],r['snr_db'],True);scale=r['common_scale']
        assert np.array_equal(np.asarray((legacy*scale).T,dtype='float32'),oldx[i]);a=np.asarray((correct*scale).T,dtype='float32')
        assert np.array_equal(a[4],oldx[i,4]) and np.array_equal(np.asarray(s[:,4]*scale,dtype='float32'),oldy[i]);newx[i]=a
        maxpeak=max(maxpeak,float(np.abs(a).max()));outside+=np.any(np.abs(a)>1)
        newcat.append({**r,'geometry_policy':'rigid_rotation','legacy_paired_index':i,'actual_noise_directions':[rotate_azimuth(u,r['target_azimuth']-20).tolist() for u in ds]})
        if (i+1)%256==0:status(stage='training_data',done=i+1,total=2048)
        if time.monotonic()-start>plan['max_data_seconds']-180:raise TimeoutError('G15preparation reserve')
    newx.flush();del newx
    for name in ('train_y.npy','clean_x.npy'):
        shutil.copy2(old/name,BASE/name);assert sha(old/name)==sha(BASE/name)
    dump(BASE/'train_catalog.json',newcat,True)
    # Replay the old validation RNG, checking saved intervals and all six channels.
    oldval=read(ROOT/'data/g5_v1/validation/manifest.jsonl');val_sources=list(dict.fromkeys(r['source_id'] for r in oldval));assert len(oldval)==54 and len(val_sources)==6
    assert not set(val_sources)&{r['source_id'] for r in catalog} and all(source[s]['split']=='validation' for s in val_sources)
    lookup={(r['source_id'],r['condition'],r['relative_angle'],r['target_azimuth']):r for r in oldval};rng=np.random.default_rng(2026111503);valrows=[];reproduced=0
    for sid in val_sources:
        mono=speech(sid)
        for field,rel,k in [('point',-80,1),('point',25,1),('multi8',None,8)]:
            clips=[];intervals=[]
            for j in range(k):
                lo=int((125+50*j/k)*16000);hi=int((125+50*(j+1)/k)*16000)-len(mono);offset=int(rng.integers(lo,hi+1));intervals.append([offset,offset+len(mono)]);clips.append(background('OOFFICE')[offset:offset+len(mono)])
            if k==1:ds=[direction(20+rel,-15)]
            else:
                ds=[rng.normal(size=3) for _ in clips];ds=[u/np.linalg.norm(u) for u in ds]
            variants=[]
            for ta in (-30,20,60):
                prior=lookup[(sid,field,rel,ta)];assert prior['noise_intervals']==intervals;x,s=render(mono,clips,ds,ta,0,False);saved,sr=sf.read(prior['noisy_path'],dtype='float32',always_2d=True)
                assert sr==16000 and sha(prior['noisy_path'])==prior['input_sha256'] and np.array_equal(pcm24_field(x*prior['common_scale']),saved);reproduced+=1
                correct,_=render(mono,clips,ds,ta,0,True);variants.append((prior,x,s,correct))
            scale=min(variants[0][0]['common_scale'],.95/max(float(np.abs(a).max()) for _,x,s,c in variants for a in (x,s,c)))
            for prior,x,s,c in variants:
                sample_id='g15_val_'+prior['sample_id'];xp=BASE/'validation'/(sample_id+'_6ch.wav');sp=BASE/'validation'/(sample_id+'_ref.wav')
                with xp.open('xb') as f:sf.write(f,c*scale,16000,format='WAV',subtype='PCM_24')
                with sp.open('xb') as f:sf.write(f,s[:,4]*scale,16000,format='WAV',subtype='PCM_24')
                valrows.append({**prior,'sample_id':sample_id,'geometry_policy':'rigid_rotation','noisy_path':str(xp),'reference_path':str(sp),'input_sha256':sha(xp),'reference_sha256':sha(sp),'common_scale':scale,
                    'base_noise_directions':[u.tolist() for u in ds],'actual_noise_directions':[rotate_azimuth(u,prior['target_azimuth']-20).tolist() for u in ds]})
    assert reproduced==54
    with (BASE/'validation/manifest.jsonl').open('x') as f:
        for r in valrows:f.write(json.dumps(r)+'\n')
    files={str(p.relative_to(ROOT)):sha(p) for p in BASE.rglob('*') if p.is_file()};dump(OUT/'data_file_hashes.json',files,True)
    done={'status':'ok','completed_utc':now(),'training_records':2048,'all_legacy_training_inputs_reconstructed_exact':True,'all_reference_mic_training_samples_exact':True,
        'all_training_targets_and_clean_arrays_byte_identical':True,'training_input_peak':maxpeak,'training_records_with_abs_over1':int(outside),'training_storage':'float32NPY,no clipping',
        'training_unique_sources':len({r['source_id'] for r in catalog}),'training_speakers':len({r['speaker_id'] for r in catalog}),'validation_sources':6,'validation_speakers':len({source[s]['speaker_id'] for s in val_sources}),
        'all54_old_validation_inputs_replayed_exact':True,'validation_new_corrected_cases':54,'train_validation_sources_disjoint':True,'elapsed_seconds':time.monotonic()-start};dump(OUT/'PREPARATION_COMPLETE.json',done,True)
    lines=['# G15A 正确旋转训练数据与验证重放','','G14完整验收和后续决策保存后冻结。2048条原G5训练记录逐一按旧公式还原六通道数组，全部逐样本相同；仅把噪声转向改为与目标同向，保持每源/片段/角序列/缩放及采样分类。新旧参考麦输入全部逐样本相同，监督train_y和clean_x数组字节相同，所有目标/噪声点积保持。',
      '',f'训练源{done["training_unique_sources"]}句/{done["training_speakers"]}人；2048条是重复裁剪混合，不是独立语句。1秒语音核心加128样本FIR保护，float32NPY，不经过PCM截幅；同一旧缩放不根据新结果重调。新输入峰值{maxpeak:.6f}，abs>1记录{outside}条，实际数值保留而不静默削波。',
      '', '验证按原G5 RNG、片段抽样顺序和八源方向重放，54个旧输入全部六通道逐样本还原；再生成54个正确几何验证样本。6句验证源与训练不交叉，验证仅诊断不选检查点。G10/G14音频不进入训练器。',
      '', '下一阶段三种子各3000步，固定末步注册。旧引擎的STOI门和selected字段保留作诊断但不用于注册；不得因新验证成绩改步数或漏种子。数据/代码/参数/哈希报告保存后才训练。',
      '', '这不是新模型结构或创新已成立。旧反向训练作为另一增强策略保留，未来使用相同正确测试集比较固定/反向/刚性训练及MVDR。','', 'UTC：'+now()]
    report=ROOT/'docs/G15A_正确旋转训练数据与验证重放.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');archive('preparation',report,[OUT/'frozen_plan.json',OUT/'PIPELINE_READY.json',OUT/'PREPARATION_COMPLETE.json',OUT/'data_file_hashes.json',BASE/'train_catalog.json',BASE/'validation/manifest.jsonl']+[ROOT/n for n in json.loads((OUT/'PIPELINE_READY.json').read_text())['files']])


if __name__=='__main__':main()
