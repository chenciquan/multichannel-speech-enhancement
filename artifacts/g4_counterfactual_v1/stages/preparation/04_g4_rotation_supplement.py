#!/usr/bin/env python3
"""Pre-inference control: rotate target and interferer together at fixed offset."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.simulation import render_plane_wave,geometric_delays,speech_activity_mask,mix_at_snr
from mcse.spatial_noise import highpass40,direction
import g4_build as helper
OUT=ROOT/'artifacts/g4_counterfactual_v1';BASE=ROOT/'data/g4_v1'
def main():
    assert (OUT/'BUILD_COMPLETE.json').exists() and not (OUT/'evaluation').exists()
    old=helper.read(BASE/'manifest.jsonl');plan=json.loads((OUT/'frozen_protocol.json').read_text());new=[];controls={}
    source={r['source_id']:r for r in helper.read(ROOT/'data/g3_v1/sources/speech_manifest.jsonl')}
    geom=json.loads((ROOT/'configs/g1_geometry.json').read_text());pos=geom['microphone_positions_m'];noises={}
    for r in json.loads((ROOT/'data/g3_v1/sources/noise_manifest.json').read_text()):
        if r['noise_id'] in ('OOFFICE','PCAFETER'):noises[r['noise_id']]=highpass40(sf.read(r['path'],dtype='float64')[0])
    for sid in plan['selected_source_ids']:
        r=source[sid];mono,_=sf.read(ROOT/'data/g3_v1/sources'/r['path'],dtype='float64')
        for noise_id in ('OOFFICE','PCAFETER'):
            base=next(a for a in old if a['source_id']==sid and a['noise_id']==noise_id and a['snr_db']==0 and a['condition']=='normal')
            offset=base['noise_offset_samples'];clip=noises[noise_id][offset:offset+len(mono)];variants=[]
            for ta in (-30,20,60):
                speech=render_plane_wave(mono,geometric_delays(pos,direction(ta,10)));s=speech[:,4];activity=speech_activity_mask(s)
                for relative in (-80,25):
                    na=ta+relative;nf=render_plane_wave(clip,geometric_delays(pos,direction(na,-15)))
                    x,_,_=mix_at_snr(speech,nf,0,activity);variants.append((x,s,ta,na,relative))
            scale=min(1.,.95/max(max(float(np.abs(x).max()),float(np.abs(s).max())) for x,s,*_ in variants));hashes=set()
            for x,s,ta,na,relative in variants:
                sample=f'g4_rotation_{len(new):04d}_{sid}';xp=BASE/'noisy'/(sample+'.wav');sp=BASE/'reference'/(sample+'.wav')
                sf.write(xp,x*scale,16000,subtype='PCM_24');sf.write(sp,s*scale,16000,subtype='PCM_24')
                xx,_=sf.read(xp,dtype='float32',always_2d=True);ss,_=sf.read(sp,dtype='float32')
                xh=hashlib.sha256(xx[:,4].tobytes()).hexdigest();sh=hashlib.sha256(ss.tobytes()).hexdigest();hashes.add((xh,sh))
                new.append({'sample_id':sample,'source_id':sid,'speaker_id':r['speaker_id'],'transcript':r['transcript'],'split':'stress_test',
                  'condition':'rotation','noise_id':noise_id,'snr_db':0,'target_azimuth':ta,'noise_azimuth':na,'relative_angle':relative,
                  'counterfactual_group':f'{sid}|{noise_id}|0|rotation','noisy_path':str(xp),'reference_path':str(sp),
                  'input_sha256':helper.sha(xp),'reference_sha256':helper.sha(sp),'raw_mic5_sample_sha256':xh,'reference_sample_sha256':sh,
                  'common_scale':scale,'noise_offset_samples':offset,'sample_rate':16000,'ref_channel':4})
            assert len(hashes)==1;controls[f'{sid}|{noise_id}|rotation']=True
    assert len(new)==96
    with (BASE/'manifest_extended.jsonl').open('x') as f:
        for r in old+new:f.write(json.dumps(r)+'\n')
    amendment={'frozen_utc':datetime.now(timezone.utc).isoformat(),'registered_before_any_G4_inference':True,'original_manifest_sha256':helper.sha(BASE/'manifest.jsonl'),
      'extended_manifest_sha256':helper.sha(BASE/'manifest_extended.jsonl'),'extra_cases':96,'total_cases_per_system':472,
      'reason':'Changing target with fixed interferer also changes relative separation. Add rigid-scene rotations at fixed relative azimuth -80/+25deg, SNR0, both noises, same reference samples. Array anisotropy still prevents a pure causal proof of training-direction overfit.',
      'all_counterfactual_groups_equal':all(controls.values())}
    helper.dump(OUT/'rotation_supplement.json',amendment)
    report=ROOT/'docs/G4A_数据与压力测试协议.md'
    report.write_text('# G4A 数据与压力测试协议\n\nG3报告及审计已保存后启动本阶段。本阶段不训练，冻结原始McNet和G3首种子的四种适配器，另有带噪参考。\n\n选取旧保留测试的8句、4位说话人，按说话人/语句ID轮询；它是同源探索性压力测试，不是独立新测试。核心288例将目标方位−30/20/60°、噪声方位−60/45°、SNR−5/0/+5dB、两种噪声完全交叉；另64例固定通道增益/同步偏差、24例干净输入。\n\n为避免相对角度变化被误解为绝对方向过拟合，在模型推理前增补登记96例整体旋转对照：目标与噪声同转、相对方位固定−80/+25°，SNR0。阵列各向异性仍然存在，因此不能单凭它证明过拟合机制。合计472例/系统、6系统，共2,832信号对及对应ASR评估。\n\n同源/噪声/SNR组使用同一噪声片段及共同防削波尺度；改变其他麦克风空间关系时，第5麦带噪样本和干净参考的哈希必须逐组完全相同，已在构建中断言。增益与同步偏差不作用于参考第5麦；同步偏差为零填充固定采样延迟，不是时钟漂移。\n\n仅名义阵列自由场平面波，无混响或真实硬件泛化结论。所有方案固定为种子20261112，不挑选最佳测试种子。信号指标报告先保存，再运行固定tiny.en；本阶段报告及审计保存后才进入后续阶段。\n',encoding='utf-8')
    folder=OUT/'stages/preparation';folder.mkdir(parents=True)
    import shutil
    files=[OUT/'frozen_protocol.json',OUT/'rotation_supplement.json',report,ROOT/'scripts/g4_build.py',Path(__file__)]
    evidence=[]
    for i,p in enumerate(files):
        target=folder/f'{i:02d}_{p.name}';shutil.copy2(p,target);evidence.append({'source':str(p.relative_to(ROOT)),'snapshot':target.name,'sha256':helper.sha(p)})
    helper.dump(folder/'archive_manifest.json',{'saved_at_utc':datetime.now(timezone.utc).isoformat(),'files':evidence,'manifest_sha256':helper.sha(BASE/'manifest_extended.jsonl')})
    print(json.dumps(amendment))
if __name__=='__main__':main()
