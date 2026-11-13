#!/usr/bin/env python3
"""Paired fixed/rotated scenes, common validation, fresh balanced confirmation."""
import json,shutil,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.simulation import render_plane_wave,geometric_delays,speech_activity_mask,mix_at_snr
from mcse.spatial_noise import highpass40,direction
from g4_build import sha,dump,read
OUT=ROOT/'artifacts/g5_rotation_v1';BASE=ROOT/'data/g5_v1';PLAN=json.loads((ROOT/'configs/g5_rotation_plan.json').read_text())
def rotate(u,degrees):
    a=np.deg2rad(degrees);return np.array([np.cos(a)*u[0]-np.sin(a)*u[1],np.sin(a)*u[0]+np.cos(a)*u[1],u[2]])
def main():
    assert (ROOT/'artifacts/g4_counterfactual_v1/stages/completion/archive_manifest.json').exists();OUT.mkdir(parents=True,exist_ok=False)
    dump(OUT/'frozen_plan.json',PLAN);geom=json.loads((ROOT/'configs/g1_geometry.json').read_text());pos=geom['microphone_positions_m']
    source=read(ROOT/'data/g3_v1/sources/speech_manifest.jsonl');sm={r['source_id']:r for r in source};audio={}
    for r in source:
        if r['split']!='test':
            p=ROOT/'data/g3_v1/sources'/r['path'];assert sha(p)==r['sha256'];audio[r['source_id']]=sf.read(p,dtype='float64')[0]
    noise={}
    for r in json.loads((ROOT/'data/g3_v1/sources/noise_manifest.json').read_text()):
        assert sha(r['path'])==r['sha256'];noise[r['noise_id']]=highpass40(sf.read(r['path'],dtype='float64')[0])
    def scene(mono,clips,base_directions,ta,snr):
        s=render_plane_wave(mono,geometric_delays(pos,direction(ta,10)));nf=np.zeros_like(s)
        for clip,u in zip(clips,base_directions):nf+=render_plane_wave(clip,geometric_delays(pos,rotate(u,ta-20)))/np.sqrt(len(clips))
        x,_,gain=mix_at_snr(s,nf,snr,speech_activity_mask(s[:,4]));return x,s,gain
    catalogue=json.loads((ROOT/'data/g3_v1/dataset/train_catalog.json').read_text());maps={};metas={g:[] for g in ('fixed','rotated')}
    for g in maps.keys():pass
    for g in ('fixed','rotated'):
        folder=BASE/g;folder.mkdir();(folder/'validation').mkdir()
        maps[g]={name:np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype='float32',shape=shape) for name,shape in
                 [('train_x',(2048,6,16128)),('train_y',(2048,16128)),('clean_x',(128,6,16128))]}
    rng=np.random.default_rng(2026111501);rot_rng=np.random.default_rng(2026111502);equality=0
    for i,r in enumerate(catalogue):
        mono=audio[r['source_id']][r['source_crop_start']:r['source_crop_start']+16000]
        clips=[noise[r['noise_id']][lo:hi] for lo,hi in r['noise_intervals']]
        if r['field']=='point':dirs=[direction(r['angle'],-15)]
        else:
            dirs=[rng.normal(size=3) for _ in clips];dirs=[u/np.linalg.norm(u) for u in dirs]
        target=float(rot_rng.uniform(-45,75));xf,sf_,gf=scene(mono,clips,dirs,20,r['snr_db']);xr,sr_,gr=scene(mono,clips,dirs,target,r['snr_db'])
        assert np.array_equal(xf[:,4],xr[:,4]) and np.array_equal(sf_[:,4],sr_[:,4]);equality+=1
        scale=min(1.,.95/max(float(np.abs(a).max()) for a in (xf,xr,sf_,sr_)))
        for g,x,s,ta in [('fixed',xf,sf_,20),('rotated',xr,sr_,target)]:
            maps[g]['train_x'][i]=(x*scale).T;maps[g]['train_y'][i]=s[:,4]*scale
            if i<128:maps[g]['clean_x'][i]=(s*scale).T
            metas[g].append({**r,'target_azimuth':ta,'base_noise_directions':[u.tolist() for u in dirs],'common_scale':scale,'noise_gain':gf,'paired_index':i})
        if (i+1)%256==0:print(json.dumps({'stage':'paired_train_build','completed':i+1,'total':2048}),flush=True)
    for g in maps:
        for m in maps[g].values():m.flush()
        dump(BASE/g/'train_catalog.json',metas[g])
    del maps
    dump(OUT/'PAIRED_TRAIN_COMPLETE.json',{'examples':2048,'paired_reference_mic_and_target_equal':equality==2048,'source_crops_and_noise_intervals_shared':True})
    def directions(rng,k,relative=None):
        if k==1:return [direction(20+relative,-15)]
        ds=[rng.normal(size=3) for _ in range(k)];return [u/np.linalg.norm(u) for u in ds]
    def clips_for(rng,noise_id,lo,hi,n,k):
        out=[];intervals=[]
        for j in range(k):
            lower=int((lo+(hi-lo)*j/k)*16000);upper=int((lo+(hi-lo)*(j+1)/k)*16000)-n
            offset=int(rng.integers(lower,upper+1));out.append(noise[noise_id][offset:offset+n]);intervals.append([offset,offset+n])
        return out,intervals
    def make_split(records,split,targets,noisetypes,window,seed):
        folder=BASE/split;folder.mkdir();(folder/'noisy').mkdir();(folder/'reference').mkdir();manifest=[];rng=np.random.default_rng(seed)
        for r in records:
            mono=audio[r['source_id']]
            for noise_id in noisetypes:
                for field,rel,k in [('point',-80,1),('point',25,1),('multi8',None,8)]:
                    clips,intervals=clips_for(rng,noise_id,*window,len(mono),k);ds=directions(rng,k,rel);variants=[]
                    for ta in targets:
                        x,s,gain=scene(mono,clips,ds,ta,0);variants.append((ta,x,s))
                    scale=min(1.,.95/max(float(np.abs(a).max()) for _,x,s in variants for a in (x,s)))
                    for ta,x,s in variants:
                        sid=f'g5_{split}_{len(manifest):04d}_{r["source_id"]}';xp=folder/'noisy'/(sid+'.wav');sp=folder/'reference'/(sid+'.wav')
                        sf.write(xp,x*scale,16000,subtype='PCM_24');sf.write(sp,s[:,4]*scale,16000,subtype='PCM_24')
                        manifest.append({'sample_id':sid,'source_id':r['source_id'],'speaker_id':r['speaker_id'],'transcript':r['transcript'],'split':split,
                          'condition':field,'noise_field':field,'relative_angle':rel,'noise_id':noise_id,'snr_db':0,'target_azimuth':ta,
                          'noisy_path':str(xp),'reference_path':str(sp),'input_sha256':sha(xp),'reference_sha256':sha(sp),'common_scale':scale,'noise_intervals':intervals})
            if split=='confirmation':
                clean=[(ta,render_plane_wave(mono,geometric_delays(pos,direction(ta,10)))) for ta in targets];scale=min(1.,.95/max(float(np.abs(s).max()) for _,s in clean))
                for ta,s in clean:
                    sid=f'g5_clean_{len(manifest):04d}_{r["source_id"]}';xp=folder/'noisy'/(sid+'.wav');sp=folder/'reference'/(sid+'.wav');sf.write(xp,s*scale,16000,subtype='PCM_24');sf.write(sp,s[:,4]*scale,16000,subtype='PCM_24')
                    manifest.append({'sample_id':sid,'source_id':r['source_id'],'speaker_id':r['speaker_id'],'transcript':r['transcript'],'split':'clean_test',
                      'condition':'clean','noise_field':'clean','relative_angle':None,'noise_id':'none','snr_db':None,'target_azimuth':ta,'noisy_path':str(xp),'reference_path':str(sp),
                      'input_sha256':sha(xp),'reference_sha256':sha(sp),'common_scale':scale})
        with (folder/'manifest.jsonl').open('x') as f:
            for r in manifest:f.write(json.dumps(r)+'\n')
        return manifest
    val=[]
    for sp in sorted({r['speaker_id'] for r in source if r['split']=='validation'}):val += sorted([r for r in source if r['split']=='validation' and r['speaker_id']==sp],key=lambda r:r['source_id'])[:3]
    valrows=make_split(val,'validation',[-30,20,60],['OOFFICE'],[125,175],2026111503);assert len(valrows)==54
    for g in ('fixed','rotated'):shutil.copy2(BASE/'validation/manifest.jsonl',BASE/g/'validation/manifest.jsonl')
    deadline=datetime.fromisoformat(PLAN['data_cutoff_utc'].replace('Z','+00:00')).timestamp()
    while not (BASE/'sources/speech_manifest.jsonl').exists():
        stat=BASE/'sources/status.json'
        if stat.exists() and json.loads(stat.read_text()).get('status')=='failed':raise RuntimeError('Fresh confirmation fetch failed; no old-source fallback')
        if time.time()>deadline:raise TimeoutError('Fresh-source preparation deadline')
        time.sleep(5)
    fresh=read(BASE/'sources/speech_manifest.jsonl');assert len(fresh)==12
    for r in fresh:
        p=BASE/'sources'/r['path'];assert sha(p)==r['sha256'];audio[r['source_id']]=sf.read(p,dtype='float64')[0]
    assert not {r['speaker_id'] for r in fresh}&{r['speaker_id'] for r in source}
    testrows=make_split(fresh,'confirmation',[-40,20,70],['OOFFICE','PCAFETER'],[250,295],2026111504);assert len(testrows)==252
    report=ROOT/'docs/G5A_配对旋转数据与新语句协议.md'
    report.write_text('# G5A 配对旋转数据与新语句协议\n\nG4报告和审计保存后开展。使用同一2,048个源语音裁片、噪声时间段、源方向种子和SNR，重新生成固定场景与整场景旋转两套配对训练数据。全部声源共同旋转，目标方位均匀分布−45至75°；人声与噪声相对关系保持。每对共享防削波尺度，第5麦带噪样本和干净目标完全相同，已逐例断言。每套含128个对应干净训练片段。\n\n验证为原验证集两位说话人各3句，三种目标方向×两种点声源相对方向/八源叠加，OOFFICE 0dB，共54例，所有组共用。\n\n新确认集从官方有界流取3位此前完全未出现的说话人，各4句，1—3.8秒。全部G1/G2/G3说话人和源文件哈希排除。目标−40/20/70°，两种噪声，点声源相对−80/+25°或八源叠加，0dB：216个带噪条件及36个干净输入。仅语句/说话人是独立新增，噪声仍是原录音的后段，不能声称全新独立录音。\n\n比较固定/旋转×干净约束有/无，均衡采样、rank4、3584参数、FP32、3000步；按相同54例验证规则选检查点。至少完成一整组四单元种子，后续整组只由剩余时间决定；全部选权重和训练报告保存后才打开确认集。旋转增强是已有思路，不冒充新模块或创新。\n',encoding='utf-8')
    folder=OUT/'stages/preparation';folder.mkdir(parents=True);entries=[]
    for i,p in enumerate([ROOT/'configs/g5_rotation_plan.json',report,Path(__file__),BASE/'sources/status.json',BASE/'sources/speech_manifest.jsonl',BASE/'validation/manifest.jsonl']):
        dst=folder/f'{i:02d}_{p.name}';shutil.copy2(p,dst);entries.append({'source':str(p.relative_to(ROOT)),'snapshot':dst.name,'sha256':sha(p)})
    dump(folder/'archive_manifest.json',{'saved_at_utc':datetime.now(timezone.utc).isoformat(),'files':entries,'confirmation_manifest_sha256':sha(BASE/'confirmation/manifest.jsonl')})
    dump(OUT/'PREPARATION_COMPLETE.json',{'paired_train_examples':2048,'validation':54,'confirmation':252,'sources':12,'speakers':3,'fresh_sources_per_speaker':4,'saved_utc':datetime.now(timezone.utc).isoformat()});print('G5 preparation complete',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();sys.exit(1)
