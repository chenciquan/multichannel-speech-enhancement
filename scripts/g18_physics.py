#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import gc,json
from importlib.metadata import version
import numpy as np
import soundfile as sf
import rir_generator as rir
import torch
from g18_common import ROOT,OUT,now,sha,dump,read,archive,status
from mcse.spatial_noise import direction
from mcse.reverberation import schroeder_t20
from mcse.runtime import load_system,enhance_array


def main():
    status(stage='physics');plan=json.loads((OUT/'frozen_plan.json').read_text());asset=json.loads((ROOT/'artifacts/g12_room_stress_v2/ASSET_COMPLETE.json').read_text());assert version('rir-generator')=='0.3.0'
    assert all(sha(p)==h for p,h in asset['installed_module_files_sha256'].items())
    pos=np.asarray(json.loads((ROOT/'configs/g1_geometry.json').read_text())['microphone_positions_m']);ref=np.array(plan['reference_mic4_position_m']);receivers=ref+(pos-pos[4]);room=np.array(plan['room_dimensions_m']);source=ref+1.5*direction(20,10)
    assert ((receivers>0)&(receivers<room)).all() and ((source>0)&(source<room)).all();probes=[];kernels={}
    for rt in plan['requested_RT60_seconds']:
        kw={'beta':np.zeros(6)} if rt==0 else {'reverberation_time':rt}
        args={'c':343,'fs':16000,'r':receivers,'s':source,'L':room,'nsample':7680,'mtype':rir.mtype.omnidirectional,'order':-1,'dim':3,'hp_filter':False,**kw}
        h=rir.generate(**args)*(4*np.pi*1.5);assert h.shape==(7680,6) and np.isfinite(h).all() and np.array_equal(h,rir.generate(**args)*(4*np.pi*1.5));kernels['rt_'+str(rt)]=h
        expected=np.linalg.norm(receivers-source,axis=1)/343*16000;peak=np.argmax(np.abs(h),axis=0);tail=float(np.sum(h[320:,4]**2)/np.sum(h[:,4]**2));measured=schroeder_t20(h[:,4]) if rt else None
        if rt==0:assert np.max(np.abs(peak-expected))<=2 and tail<1e-8
        else:assert measured is not None and tail>1e-8
        probes.append({'requested_RT60':rt,'measured_T20_extrapolated_RT60':measured,'tail_fraction_after20ms':tail,'expected_direct_samples':expected.tolist(),'largest_peak_samples':peak.tolist(),'deterministic_repeat_exact':True})
    assert probes[2]['measured_T20_extrapolated_RT60']>probes[1]['measured_T20_extrapolated_RT60']>0
    np.savez_compressed(OUT/'physics_smoke_RIRs.npz',**kernels);source_rows=read(ROOT/'data/g17_v1/sources/speech_manifest.jsonl');assert len(source_rows)==8 and all(r['duration_seconds']<=3.51 for r in source_rows);longest=max(r['samples'] for r in source_rows)+7679;assert longest<64000
    torch.set_num_threads(2);rng=np.random.default_rng(2026112301);synthetic=(rng.standard_normal((longest,6))*.03).astype('float32');case=read(ROOT/'data/g17_v1/manifest.jsonl')[0];x,sr=sf.read(case['noisy_path'],dtype='float32',always_2d=True);old=read(ROOT/'artifacts/g17_fresh_confirmation_v1/evaluation/combined_signals.jsonl');capacity=[]
    for model in ('mcnet','g15/rigid_without_clean_identity_s20261112'):
        runtime=load_system(model,'cuda');previous=next(r for r in old if r['system']==model.split('/')[-1] and r['sample_id']==case['sample_id']);assert sha(previous['output_path'])==previous['output_sha256'];expected=sf.read(previous['output_path'],dtype='float32')[0]
        torch.cuda.reset_peak_memory_stats();y,_=enhance_array(runtime,x);error=float(np.max(np.abs(y-expected)));assert error<=1e-5;z,seconds=enhance_array(runtime,synthetic);assert len(z)==longest and np.isfinite(z).all()
        capacity.append({'system':model,'old_G17_max_abs':error,'maximum_full_convolution_frames':longest,'synthetic_capacity_seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'metadata':runtime.metadata});del runtime;gc.collect();torch.cuda.empty_cache()
    done={'status':'ok','completed_utc':now(),'RIR_version':'0.3.0','all_existing_generator_module_hashes_match':True,'new_installations':False,'RIR_probes':probes,'capacity_probes':capacity};dump(OUT/'PHYSICS_COMPLETE.json',done,True)
    lines=['# G18A 固定房间协议、生成器与容量检查','','G17最终读回与决策保存后冻结。原有rir-generator0.3.0的全部已登记模块SHA仍匹配，没有安装/下载或改驱动。6×5×3m房间、原名义六麦偏移、源距1.5m、7680点全向响应，禁用内置高通。无反射用零反射系数，两正参数沿用官方接口；同参重复逐样本相同。',
       '', '无反射直接峰与传播距离误差≤2样本，20ms后能量近零；正参数有反射尾部，测得T20外推顺序正确。请求参数不等于实测衰减，所有源位置后续另存。','','|请求RT秒|此位置T20外推秒|20ms后能量比例|','|---:|---:|---:|']
    for r in probes:lines.append(f'|{r["requested_RT60"]}|{r["measured_T20_extrapolated_RT60"]}|{r["tail_fraction_after20ms"]:.6f}|')
    lines+=['','|模型|旧G17最大波形差|最长帧|Torch分配GiB|Torch保留GiB|','|---|---:|---:|---:|---:|']
    for r in capacity:lines.append(f'|{r["system"]}|{r["old_G17_max_abs"]}|{longest}|{r["peak_allocated_bytes"]/2**30:.3f}|{r["peak_reserved_bytes"]/2**30:.3f}|')
    lines+=['','容量输入为固定随机波形，非质量或方法选择；只作原模型/一个同结构刚性适配器容量，整批实际峰值后续记录。完整原语句加有限0.48秒滤波器尾部均小于4秒。',
       '', '源星座在固定墙体内转动，保留直接相对几何但反射场变化；RT0是近场无反射内部对照，不等同G17平面波。固定全局八源星座也不同G17逐组随机场，不能将跨阶段全部差值归因混响。参考是第5麦混响语音，仅衡量降噪。连续噪声用1秒预卷积覆盖语句与响应尾部。',
       '', '本阶段尚未生成房间评估混合或取得其成绩。数据→直接→MVDR→Whisper→CTC继续逐阶段归档，全部21系统不变。有限响应不保证自然无限尾部完整，末端能量另记；单房间及已查看4人不能代替真实/多房间验证。','', 'UTC：'+now()]
    report=ROOT/'docs/G18A_固定房间物理协议与容量验证.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');archive('physics',report,[OUT/'frozen_plan.json',OUT/'PHYSICS_COMPLETE.json',ROOT/'artifacts/g12_room_stress_v2/ASSET_COMPLETE.json',ROOT/'scripts/g18_physics.py'])


if __name__=='__main__':main()
