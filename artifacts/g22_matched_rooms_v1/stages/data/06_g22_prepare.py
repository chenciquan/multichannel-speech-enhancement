#!/usr/bin/env python3
"""Matched reference/new-room data, with shared levels and saved source segments."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
import time
import traceback
from collections import defaultdict
from functools import lru_cache
from importlib.metadata import version
import numpy as np
import soundfile as sf
import rir_generator as rir
import g8_common as common
from mcse.components import pcm24_field
from mcse.reverberation import full_speech,continuous_noise,schroeder_t20
from mcse.simulation import speech_activity_mask,mix_at_snr,snr_db
from mcse.spatial_noise import highpass40

ROOT=common.ROOT;OUT=ROOT/'artifacts/g22_matched_rooms_v1';BASE=ROOT/'data/g22_v1'
G18=ROOT/'artifacts/g18_room_stress_v1';G21=ROOT/'artifacts/g21_truth_covariance_v1'
sha,now,dump,read=common.sha,common.now,common.dump,common.read


def archive(stage,report,files):
    previous=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous


def main():
    assert json.loads((G21/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    decision=ROOT/'docs/G21_识别结论与配对房间确认决策.md'
    with decision.open('x',encoding='utf-8') as f:
        f.write('# G21 识别结论与下一阶段决策\n\nG21完整阶段已验收。请求RT=.35时，真值MVDR/MWF的SI-SDR为4.37/9.70 dB，而Whisper WER为24.25%/32.72%，CTC为30.25%/40.30%。使用真值统计也没有消除声学与识别权衡；这些系统不可部署，也不是理论上界。\n\n下一阶段固定全部方法和参数，选择四位既有说话人各自源清单中的第一句，比较原房间与一个新房间。为排除电平影响，两房间在同源、噪声场型、全部RT与方向间共用防削波尺度。因此同时重渲染原房间，不能直接拿不同尺度的旧输出声称纯房间效应。先物理及数据归档，再运行模型；不新增训练。\n')
    previous=common.OUT
    try:common.OUT=G21;common.archive('next_matched_room_decision',decision,[G21/'FINAL_ACCEPTANCE_AUDIT.json'])
    finally:common.OUT=previous
    OUT.mkdir(exist_ok=False);BASE.mkdir(exist_ok=False)
    for name in ('noisy','reference','rir'):(BASE/name).mkdir()
    sources=[];speakers=set()
    for s in json.loads((G18/'selected_sources.json').read_text()):
        if s['speaker_id'] not in speakers:sources.append(s);speakers.add(s['speaker_id'])
    assert len(sources)==4
    source_ids={s['source_id'] for s in sources}
    old_cases=[c for c in read(ROOT/'data/g18_v1/manifest.jsonl') if c['source_id'] in source_ids]
    assert len(old_cases)==252
    pins=['scripts/g22_prepare.py','configs/g1_geometry.json','src/mcse/components.py','src/mcse/reverberation.py',
          'src/mcse/simulation.py','src/mcse/spatial_noise.py','data/g18_v1/manifest.jsonl',
          'artifacts/g18_room_stress_v1/selected_sources.json','artifacts/g18_room_stress_v1/RIR_manifest.json',
          'artifacts/g18_room_stress_v1/noise_sources.json','artifacts/g21_truth_covariance_v1/FINAL_ACCEPTANCE_AUDIT.json']
    plan={'frozen_utc':now(),'max_seconds':1800,'speakers':4,'utterances':4,
          'selection':'First listed source per G18 speaker, before new-room scores',
          'reference_room_dimensions_m':[6,5,3],'new_room_dimensions_m':[4,3.5,2.8],
          'new_reference_position_m':[2,1.75,1.4],'source_distance_m':1.5,
          'RT_requests':[0,.15,.35],'target_angles':[-40,20,70],
          'RIR_samples':7680,'SNR_db':0,'no_new_training':True,
          'cases_per_room':252,'cases_total':504,'noisy_cases_total':432,'clean_cases_total':72,
          'paired_level':'One anticlip scale per source/noise/field group over both rooms x all9 RT/angles',
          'RT0_control_max_abs':1/2**23,'SNR_tolerance_db':.002,
          'future_systems':'All31 blind registered direct/MVDR/MWF systems; truth-covariance two filters only on noisy432; expected16488 signals if full evaluation proceeds',
          'scope':'New room dimensions with matched source/noise segments; existing speakers and recordings, not real-array evidence',
          'hashes':{n:sha(ROOT/n) for n in pins}}
    dump(OUT/'frozen_plan.json',plan,True);dump(OUT/'selected_sources.json',sources,True)
    protocol=ROOT/'docs/G22A_匹配电平的新房间确认协议.md'
    with protocol.open('x',encoding='utf-8') as f:
        f.write('# G22A 匹配电平的新房间确认协议\n\n4位既有说话人各第一句。原房间6×5×3m、新房间4×3.5×2.8m，参考麦位于各自水平中心、高1.4m。麦阵形状、源距1.5m、相对方向、噪声片段、三个请求RT及目标方向不变。全部真实RIR、实测T20外推和末端能量保存，不把请求RT当实测T60。\n\n两房间共用跨18个变体的防削波尺度；重新渲染原房间，消除单独选尺度的电平差异。旧尺度下的原房间须逐样本还原旧输入和参考。新共同尺度下，RT0两房间输入/参考差须不超过一个PCM24 LSB，提前固定不根据成绩调整。\n\n共504条件（每房间216带噪+36干净）。先物理检查和数据归档，不运行模型/ASR。未来比较保留全部三种子、全部成熟滤波对照，真值统计始终单独标记。只有一个新增房间，不能声称广泛房间泛化。\n')
    archive('protocol',protocol,[OUT/'frozen_plan.json',OUT/'selected_sources.json',ROOT/'scripts/g22_prepare.py'])
    started=time.monotonic()
    asset=json.loads((ROOT/'artifacts/g12_room_stress_v2/ASSET_COMPLETE.json').read_text())
    assert version('rir-generator')=='0.3.0' and all(sha(p)==h for p,h in asset['installed_module_files_sha256'].items())
    old_rirs=json.loads((G18/'RIR_manifest.json').read_text())
    old_ref=np.array([3,2.5,1.4]);new_ref=np.array(plan['new_reference_position_m'])
    positions=np.array(json.loads((ROOT/'configs/g1_geometry.json').read_text())['microphone_positions_m'])
    receivers=new_ref+(positions-positions[4]);room=np.array(plan['new_room_dimensions_m'])
    assert ((receivers>0)&(receivers<room)).all()
    kernels={};rir_rows=[];verified={}
    def verify(path,h):
        path=str(path)
        if path not in verified:assert sha(path)==h;verified[path]=h
        else:assert verified[path]==h
    for old in old_rirs:
        assert time.monotonic()-started<plan['max_seconds']
        verify(old['path'],old['sha256']);prior=np.load(old['path'])
        source=new_ref+1.5*np.array(old['source_direction']);assert ((source>0)&(source<room)).all()
        rt=old['requested_rt60'];kw={'beta':np.zeros(6)} if rt==0 else {'reverberation_time':rt}
        h=rir.generate(c=343,fs=16000,r=receivers,s=source,L=room,nsample=7680,mtype=rir.mtype.omnidirectional,
                       order=-1,dim=3,hp_filter=False,**kw)*(4*np.pi*1.5)
        assert h.shape==(7680,6) and np.isfinite(h).all()
        arrival=np.linalg.norm(receivers-source,axis=1)/343*16000
        if rt==0:assert np.max(np.abs(np.argmax(np.abs(h),axis=0)-arrival))<=2
        path=BASE/'rir'/f'{len(rir_rows):03d}.npy';np.save(path,h)
        kernels[old['path']]={'reference':(prior,old['path']),'new':(h,str(path))}
        measured=schroeder_t20(h[:,4]) if rt else None
        if rt:assert measured is not None and measured>0
        rir_rows.append({'old_path':old['path'],'path':str(path),'sha256':sha(path),'requested_rt60':rt,
                         'source_direction':old['source_direction'],'source_position_m':source.tolist(),
                         'measured_T20_extrapolated_RT60':measured,
                         'truncated_last20ms_energy_fraction':float(np.sum(h[-320:,4]**2)/np.sum(h[:,4]**2)),
                         'RT0_old_new_RIR_max_abs':float(np.abs(prior-h).max()) if not rt else None})
    assert len(rir_rows)==99
    dump(OUT/'RIR_manifest.json',rir_rows,True)
    physics=ROOT/'docs/G22A_新房间物理核查.md'
    with physics.open('x',encoding='utf-8') as f:
        f.write('# G22A 新房间物理核查\n\n已核对现有rir-generator0.3.0模块哈希；99个新RIR全部有限，阵列/全部源位于新房间内，RT0直达峰在预测到达2样本内。请求RT和各位置实测T20外推分别保存。未安装或修改依赖、驱动、CUDA。先保存本记录及实际RIR清单，再渲染音频。\n')
    archive('physics',physics,[OUT/'RIR_manifest.json',OUT/'frozen_plan.json',ROOT/'artifacts/g12_room_stress_v2/ASSET_COMPLETE.json'])
    noises={}
    for n in json.loads((G18/'noise_sources.json').read_text()):
        verify(n['path'],n['sha256']);a,sr=sf.read(n['path'],dtype='float64');assert sr==16000;noises[n['noise_id']]=highpass40(a)
    monos={}
    for s in sources:
        p=ROOT/'data/g17_v1/sources'/s['path'];verify(p,s['sha256']);monos[s['source_id']],sr=sf.read(p,dtype='float64');assert sr==16000
    @lru_cache(maxsize=24)
    def speech(sid,rpath,room_id):return full_speech(monos[sid],kernels[rpath][room_id][0])
    groups=defaultdict(list)
    for c in old_cases:groups[c['paired_room_group']].append(c)
    assert len(groups)==28 and all(len(v)==9 for v in groups.values())
    rows=[];controls=[];snr_errors=[]
    with (BASE/'manifest.jsonl').open('x',encoding='utf-8') as log:
        for group,items in groups.items():
            variants=[]
            for c in items:
                for rid in ('reference','new'):
                    assert time.monotonic()-started<plan['max_seconds']
                    s=speech(c['source_id'],c['speech_RIR_path'],rid);n=np.zeros_like(s);k=len(c['noise_RIR_paths'])
                    nrirs=[]
                    for old_path,(lo,hi) in zip(c['noise_RIR_paths'],c['noise_intervals']):
                        h,path=kernels[old_path][rid];nrirs.append(path)
                        n+=continuous_noise(noises[c['noise_id']][lo:hi],h,16000,c['frames'])/np.sqrt(k)
                    mask=None if c['condition']=='clean' else speech_activity_mask(s[:,4])
                    if mask is None:x=s;gain=0.
                    else:x,_,gain=mix_at_snr(s,n,0,mask)
                    if rid=='reference':
                        verify(c['noisy_path'],c['input_sha256']);verify(c['reference_path'],c['reference_sha256'])
                        oldx,sr=sf.read(c['noisy_path'],dtype='float32');oldref,rr=sf.read(c['reference_path'],dtype='float32')
                        assert np.array_equal(pcm24_field(x*c['common_scale']),oldx)
                        assert np.array_equal(pcm24_field(s*c['common_scale'])[:,4],oldref)
                    variants.append((c,rid,s,x,gain,mask,nrirs))
            scale=min(1.,.95/max(float(np.abs(a).max()) for _,_,s,x,_,_,_ in variants for a in (s,x)))
            control={}
            for c,rid,s,x,gain,mask,nrirs in variants:
                sid=f'g22_{len(rows):04d}_{c["source_id"]}_{rid}'
                xp=BASE/'noisy'/(sid+'.wav');rp=BASE/'reference'/(sid+'.wav')
                with xp.open('xb') as f:sf.write(f,x*scale,16000,format='WAV',subtype='PCM_24')
                with rp.open('xb') as f:sf.write(f,s[:,4]*scale,16000,format='WAV',subtype='PCM_24')
                xx,sr=sf.read(xp,dtype='float64');rr,rs=sf.read(rp,dtype='float64');assert sr==rs==16000
                error=None if mask is None else abs(snr_db(rr,xx[:,4]-rr,mask))
                if error is not None:assert error<plan['SNR_tolerance_db'];snr_errors.append(error)
                spans=[] if mask is None else np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))).reshape(-1,2).tolist()
                row={**c,'sample_id':sid,'old_G18_sample_id':c['sample_id'],'old_input_sha256':c['input_sha256'],
                     'room_id':rid,'room_dimensions_m':[6,5,3] if rid=='reference' else plan['new_room_dimensions_m'],
                     'noisy_path':str(xp),'reference_path':str(rp),'input_sha256':sha(xp),'reference_sha256':sha(rp),
                     'common_scale':scale,'old_G18_common_scale':c['common_scale'],'noise_gain':gain,
                     'speech_RIR_path':kernels[c['speech_RIR_path']][rid][1],'noise_RIR_paths':nrirs,
                     'paired_room_group':group,'matched_room_pair':c['sample_id'],'split':'matched_room_confirmation_reused_voices',
                     'saved_active_SNR_error_db':error,'activity_spans_from_unquantized_reference':spans,
                     'recomputed_PCM_activity_difference_samples':int(np.count_nonzero(mask!=speech_activity_mask(rr))) if mask is not None else 0,
                     'saved_recomputed_PCM_activity_SNR_error_db':abs(snr_db(rr,xx[:,4]-rr,speech_activity_mask(rr))) if mask is not None else None}
                if c['requested_rt60']==0:control[c['sample_id'],rid]=(xx,rr)
                log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
            for c in items:
                if c['requested_rt60']!=0:continue
                a,ar=control[c['sample_id'],'reference'];b,br=control[c['sample_id'],'new']
                dx=float(np.abs(a-b).max());dr=float(np.abs(ar-br).max())
                assert max(dx,dr)<=plan['RT0_control_max_abs']
                controls.append({'old_sample_id':c['sample_id'],'input_max_abs':dx,'reference_max_abs':dr})
            log.flush();dump(OUT/'status.json',{'stage':'data','groups_done':len(rows)//18,'total_groups':28,'rows':len(rows),'utc':now()})
    assert len(rows)==504 and len(controls)==84 and sum(c['condition']=='clean' for c in rows)==72
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    dump(OUT/'RT0_controls.json',controls,True);dump(OUT/'verified_files.json',verified,True)
    done={'status':'ok','completed_utc':now(),'cases':504,'rooms':2,'utterances':4,'speakers':4,'new_RIRs':99,
          'old_replay_exact_cases':252,'RT0_pairs':84,'max_RT0_input_abs':max(c['input_max_abs'] for c in controls),
          'max_RT0_reference_abs':max(c['reference_max_abs'] for c in controls),'max_saved_SNR_error_db':max(snr_errors),
          'manifest_sha256':sha(BASE/'manifest.jsonl'),'elapsed_seconds':time.monotonic()-started}
    dump(OUT/'DATA_COMPLETE.json',done,True)
    report=ROOT/'docs/G22B_匹配房间数据与负控制报告.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G22B 匹配房间数据与负控制报告\n\n完成2房间×252条件，共504条（432带噪、72干净）。旧尺度下的252条原房间输入/参考全部逐样本还原；两房间共同尺度后，84对RT0负控制差均不超过预定一个PCM24 LSB。\n\n两房间共用原语句、噪声片段、源星座、源距与防削波尺度，仅房间边界及参考绝对位置改变。正RT反射参考可以不同，不能要求相等。所有输入、参考、实际RIR及来源哈希保存；请求RT和实测T20分别记录。\n\n本阶段没有模型或ASR成绩，不作方法优劣结论。旧输出电平可能不同，不直接复用为本轮配对方法结果。后续如完整评估，保留所有31盲系统、三训练种子及仅供诊断的真值统计。\n')
    archive('data',report,[OUT/'DATA_COMPLETE.json',OUT/'RT0_controls.json',OUT/'verified_files.json',BASE/'manifest.jsonl',OUT/'frozen_plan.json',ROOT/'scripts/g22_prepare.py'])
    dump(OUT/'status.json',{'stage':'data_archived_pending_readback','utc':now()})


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():dump(OUT/'failure.json',{'utc':now(),'error':repr(e)})
        traceback.print_exc();raise
