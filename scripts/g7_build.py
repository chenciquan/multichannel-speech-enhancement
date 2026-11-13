#!/usr/bin/env python3
"""Bounded new noise members and matched, frozen-model stress cases."""
import io,json,runpy,shutil,sys,time,zipfile,hashlib
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
from collections import defaultdict
import numpy as np
import requests
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.simulation import render_plane_wave,geometric_delays,speech_activity_mask,mix_at_snr
from mcse.spatial_noise import highpass40,direction
from mcse.runtime import sha256,archived_json
from g5_prepare import rotate
OUT=ROOT/'artifacts/g7_unseen_noise_v1';BASE=ROOT/'data/g7_v1'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def main():
    assert json.loads((ROOT/'artifacts/g6_local_runtime_v1/FINAL_COMPLETION_AUDIT.json').read_text())['status']=='ok'
    plan=json.loads((OUT/'frozen_plan.json').read_text());BASE.mkdir(parents=True,exist_ok=False);(BASE/'noise').mkdir();(BASE/'noisy').mkdir();(BASE/'reference').mkdir();start=time.perf_counter()
    helper=runpy.run_path(str(ROOT/'scripts/fetch_diagnostic_sources.py'),run_name='g7_range_helper');noise={};noise_rows=[]
    with requests.Session() as session:
        for name in plan['noise_ids']:
            url=f'https://zenodo.org/records/1227121/files/{name}_16k.zip?download=1';remote=helper['RangeReader'](session,url,max_bytes=20*1024**2)
            with zipfile.ZipFile(remote) as archive:
                matches=[m for m in archive.namelist() if PurePosixPath(m).name.lower()=='ch01.wav'];assert len(matches)==1
                info=archive.getinfo(matches[0]);payload=archive.read(matches[0]);crc=info.CRC
            audio,sr=sf.read(io.BytesIO(payload),dtype='float64');assert sr==16000 and audio.ndim==1 and len(audio)>=295*16000
            p=BASE/'noise'/(name+'_ch01.wav');p.write_bytes(payload);noise[name]=highpass40(audio)
            noise_rows.append({'noise_id':name,'path':str(p),'source_url':url,'official_record':plan['official_record'],'license':'CC-BY-SA-3.0','sha256':sha256(p),
               'sample_rate':sr,'channels':1,'duration_seconds':len(audio)/sr,'range_bytes_received':remote.received,'remote_archive_bytes':remote.size,'archive_member':matches[0],'member_crc32':crc,'full_archive_hash_verified':False})
            dump(OUT/'noise_sources.json',noise_rows);print(json.dumps({'stage':'noise','noise_id':name,'range_bytes':remote.received}),flush=True)
    selected=archived_json(ROOT/'artifacts/g5_rotation_v1/selected_checkpoints.json',ROOT/'artifacts/g5_rotation_v1/stages/training/archive_manifest.json')
    systems=[{'system':'noisy','kind':'noisy'},{'system':'mcnet','kind':'mcnet'}]+selected['systems'];assert len(systems)==14;dump(OUT/'systems.json',systems)
    source=read(ROOT/'data/g5_v1/sources/speech_manifest.jsonl');assert len(source)==12
    geom=json.loads((ROOT/'configs/g1_geometry.json').read_text());pos=geom['microphone_positions_m'];rng=np.random.default_rng(plan['seed']);manifest=[];canonical=defaultdict(set)
    for r in source:
        p=ROOT/'data/g5_v1/sources'/r['path'];assert sha256(p)==r['sha256'];mono,sr=sf.read(p,dtype='float64');assert sr==16000
        for noise_id in plan['noise_ids']:
            for field,relative,k in [('point',-80,1),('point',25,1),('multi8',None,8)]:
                clips=[];intervals=[]
                for j in range(k):
                    lower=int(295*j/k*16000);upper=int(295*(j+1)/k*16000)-len(mono);offset=int(rng.integers(lower,upper+1));clips.append(noise[noise_id][offset:offset+len(mono)]);intervals.append([offset,offset+len(mono)])
                if k==1:directions=[direction(20+relative,-15)]
                else:
                    directions=[rng.normal(size=3) for _ in clips];directions=[u/np.linalg.norm(u) for u in directions]
                fields={};speeches={}
                for ta in plan['target_azimuth_degrees']:
                    clean=render_plane_wave(mono,geometric_delays(pos,direction(ta,10)));nf=np.zeros_like(clean)
                    for clip,u in zip(clips,directions):nf+=render_plane_wave(clip,geometric_delays(pos,rotate(u,ta-20)))/np.sqrt(k)
                    fields[ta]=nf;speeches[ta]=clean
                for snr in plan['snr_db']:
                    variants=[]
                    for ta in plan['target_azimuth_degrees']:
                        clean=speeches[ta];x,_,gain=mix_at_snr(clean,fields[ta],snr,speech_activity_mask(clean[:,4]));variants.append((ta,x,clean,gain))
                    scale=min(1.,.95/max(float(np.abs(a).max()) for _,x,s,_ in variants for a in (x,s)))
                    for ta,x,s,gain in variants:
                        sid=f'g7_{len(manifest):04d}_{r["source_id"]}';xp=BASE/'noisy'/(sid+'.wav');sp=BASE/'reference'/(sid+'.wav');sf.write(xp,x*scale,16000,subtype='PCM_24');sf.write(sp,s[:,4]*scale,16000,subtype='PCM_24')
                        xx,_=sf.read(xp,dtype='float32',always_2d=True);ss,_=sf.read(sp,dtype='float32');xh=hashlib.sha256(xx[:,4].tobytes()).hexdigest();sh=hashlib.sha256(ss.tobytes()).hexdigest();key=f'{r["source_id"]}|{noise_id}|{field}|{relative}|{snr}';canonical[key].add((xh,sh))
                        manifest.append({'sample_id':sid,'source_id':r['source_id'],'speaker_id':r['speaker_id'],'transcript':r['transcript'],'split':'unseen_noise_stress','condition':field,'noise_field':field,
                          'relative_angle':relative,'noise_id':noise_id,'snr_db':snr,'target_azimuth':ta,'noise_azimuth':ta+relative if relative is not None else None,
                          'counterfactual_group':key,'noisy_path':str(xp),'reference_path':str(sp),'input_sha256':sha256(xp),'reference_sha256':sha256(sp),'raw_mic5_sha256':xh,'reference_samples_sha256':sh,
                          'common_scale':scale,'noise_gain':gain,'noise_intervals':intervals,'base_noise_directions':[u.tolist() for u in directions]})
        print(json.dumps({'stage':'render','sources_completed':len({r['source_id'] for r in manifest}),'cases':len(manifest)}),flush=True)
    assert len(manifest)==648 and all(len(v)==1 for v in canonical.values())
    with (BASE/'manifest.jsonl').open('x') as f:
        for r in manifest:f.write(json.dumps(r)+'\n')
    report=ROOT/'docs/G7A_新噪声迁移协议与数据.md';report.write_text('# G7A 新噪声迁移协议与数据\n\n在G6报告与审计归档后登记，不训练或选择新权重。固定G5全部12个适配器（三种子四单元）、原McNet和带噪参考，共14系统。复用G5的12句、3位说话人，因此是同源新噪声压力测试，不是新的说话人确认。\n\n新噪声仅取NPARK与PSTATION的ch01。官方[DEMAND记录](https://zenodo.org/records/1227121)已核实文件及CC-BY-SA3.0；HTTP206有界范围提取、成员CRC及本地SHA256验证，没有下载全ZIP或核实整包MD5。两文件此前未用于本项目训练/评估，不能推断原始预训练模型从未见过类似环境。\n\n目标−40/20/70°、SNR−5/0/+5dB、两种点声源相对方位−80/+25°和八源叠加，全部交叉：648例/系统，预期9,072个信号和ASR对。噪声先固定40Hz高通，再按活动语音标定SNR；共同旋转场景，成组共享片段和防削波尺度。逐组断言参考第5麦和干净参考样本完全相同。\n\n主分析逐噪声/SNR/方向报告旋转减固定，不能只引用非训练方向平均数。点声源条件占2/3，八源占1/3，另报告分组和每说话人样本数。保持原tiny.en和解码选项，WER不截断；大于100%或插入过多只标记识别退化候选，不自动诊断真实吞字。原G5干净输入结果不重复当新证据。\n\n仅名义阵列自由场模拟，不是新的真实六通道语音数据。所有噪声来源、文件与清单哈希、模型路径、运行代码留档；信号报告保存后才进入ASR，完成报告和审计保存后再推进。\n',encoding='utf-8')
    folder=OUT/'stages/preparation';folder.mkdir(parents=True);entries=[]
    for i,p in enumerate([report,OUT/'frozen_plan.json',OUT/'noise_sources.json',OUT/'systems.json',ROOT/'configs/g7_unseen_noise_plan.json',Path(__file__)]):
        dst=folder/f'{i:02d}_{p.name}';shutil.copy2(p,dst);entries.append({'source':str(p.relative_to(ROOT)),'snapshot':dst.name,'sha256':sha256(p)})
    dump(folder/'archive_manifest.json',{'saved_utc':datetime.now(timezone.utc).isoformat(),'files':entries,'manifest_sha256':sha256(BASE/'manifest.jsonl')})
    dump(OUT/'PREPARATION_COMPLETE.json',{'cases':648,'systems':14,'expected_pairs':9072,'groups_with_equal_raw_reference':len(canonical),'all_equal':True,'elapsed_seconds':time.perf_counter()-start});print('G7 data archived',flush=True)
if __name__=='__main__':main()
