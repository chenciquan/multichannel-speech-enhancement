#!/usr/bin/env python3
"""Four complete old development utterances, no selection from new outcomes."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import hashlib
import json
import time
import traceback
from functools import lru_cache
import numpy as np
import soundfile as sf
from g32_common import ROOT,OUT,BASE,SOURCES,RIR,TRAIN,PREV,sha,read,dump,now,meta,archive,status
from mcse.reverberation import full_speech,continuous_noise
from mcse.spatial_noise import highpass40
from mcse.simulation import speech_activity_mask,mix_at_snr,snr_db
from mcse.asr_scoring import normalize_english


def main():
    assert meta(PREV/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    assert meta(PREV/'primary_decision.json')['extension_gate_passed'] is False
    assert (PREV/'stages/scientific_readback/archive_manifest.json').exists()
    sources=meta(PREV/'next_complete_development_sources.json');assert len(sources)==4
    original=read(SOURCES/'speech_manifest.jsonl');expected=[]
    for sp in sorted({r['speaker_id'] for r in original if r['split']=='validation'}):
        expected.extend([r for r in original if r['split']=='validation' and r['speaker_id']==sp and r['samples']<=56320][:2])
    assert sources==expected
    if OUT.exists():
        # Only resume the explicitly preserved, pre-freeze bootstrap failure with no generated audio.
        assert (OUT/'bootstrap_failure_v1/preserved_error.json').exists() and not (OUT/'data_plan.json').exists()
        assert {p.name for p in OUT.iterdir()}<={'bootstrap_failure_v1','data_failure.json','stages'}
        assert not any(p.is_file() for p in BASE.rglob('*'))
    else:OUT.mkdir(exist_ok=False)
    BASE.mkdir(exist_ok=True)
    for name in ('noisy','reference'):(BASE/name).mkdir(exist_ok=True)
    names=['scripts/g32_common.py','scripts/g32_prepare.py','scripts/g32_data_accept.py',
        'src/mcse/reverberation.py','src/mcse/spatial_noise.py','src/mcse/simulation.py','src/mcse/asr_scoring.py',
        'data/g3_v1/sources/speech_manifest.jsonl','data/g3_v1/sources/noise_manifest.json',
        'artifacts/g29_room_development_v1/frozen_plan.json','artifacts/g29_room_development_v1/RIR_manifest.json',
        'artifacts/g29_room_development_v1/FINAL_ACCEPTANCE_AUDIT.json','artifacts/g30_room_objectives_v1/registered_checkpoints.json',
        'artifacts/g31_development_signals_v1/FINAL_ACCEPTANCE_AUDIT.json','artifacts/g31_development_signals_v1/primary_decision.json',
        'artifacts/g31_development_signals_v1/next_complete_development_sources.json']
    plan={'utc':now(),'seed':2026111302,'max_seconds':600,'sources':sources,'source_count':4,'speakers':2,
        'unique_reference_word_tokens':sum(len(normalize_english(s['transcript'])) for s in sources),
        'requests':[0,.16,.28],'fields':['point','multi8','clean'],'cases':36,
        'source_crop':False,'RIR_samples':7680,'max_output_samples':64000,'source_max_samples':56320,
        'reference_target':'full reverberant speech channel4','room':'sameG29dev0','source_distance_m':1.2,
        'angle_policy':'source-index parity -30/+30, each olddevspeaker has bothangles',
        'noise_policy':'source-index parity OOFFICE/DKITCHEN;point rel-80/+25,SNR0;multi8SNR-5;cleanNone',
        'noise_block':[125,175],'noise_preprocessing':'highpass40 independently inside devblock,1sinner margin',
        'preroll_samples':16000,'SNR_tolerance_db':.001,'training':False,'new_model_or_data_download':False,
        'scope':'secondary complete-utterance development diagnostic; no G31 gateoverride, no new-speaker finaltest',
        'planned_outputs':468,'planned_ASR_rows_each_backend':936,'ASR_regimes':['raw','level_control'],
        'dry_ASR_anchors':'4original complete dry sources, bothregimes, supplemental only; transcripts never prompts',
        'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'data_plan.json',plan,True)
    report=ROOT/'docs/G32A_完整开发语句补充诊断冻结协议.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G32A 完整开发语句补充诊断协议\n\n来源在G31科学读回前已按原清单每个旧开发说话人前两条、长度≤56320样本选定，共4条、2人；合计16个原始转写词token，很小，不代表群体识别准确率。整句不裁剪，加7680样本全卷积尾仍≤4秒。\n\n复用G29开发房间及RIR，三请求RT(0/.16/.28)×点源/八源/干净，共36条件。源序号奇偶配−30/+30目标、OOFFICE/DKITCHEN，点源相对−80/+25和SNR0，八源SNR−5；每位说话人含两个方向。噪声只在[125,175]秒块内独立高通，保留完整1秒预滚/尾部区间和1秒块内边界；八源片段互不重叠。每例混合与全六通道目标共用防削波尺度，输出FLOAT WAV，活动SNR容差.001dB。\n\n冻结原McNet及G30三组登记1500步、直接/MVDR/MWF与原带噪输入，共468信号。数据与信号先验收，再现有两个固定识别器raw及固定G25/G26电平控制分别评分，各936系统记录；另列四条原干声的双条件参考识别，辅助判断固有识别/拼写错误。无需新训练、依赖或下载。参考转写只用于评分，不作识别提示。\n\n是旧开发角色的补充诊断，不是未见说话人最终验证，不覆盖或重判G31未通过的主门槛。新波形与原1秒裁剪的噪声区间、尺度及SNR标定可能不同，不把两轮差值单独归因于上下文长度。两个识别器的S/D/I不当作声学吞字。完整语句仍很短、含人名和非标准语法，需对照原干声及混响参考识别，不改原转写/拼写来降低WER。\n')
    archive('data_protocol',report,[OUT/'data_plan.json']+[ROOT/n for n in names[:7]])
    start=time.monotonic();rng=np.random.default_rng(plan['seed']);old=meta(RIR/'frozen_plan.json');rirs=meta(RIR/'RIR_manifest.json')
    noise_meta={r['noise_id']:r for r in meta(SOURCES/'noise_manifest.json')};blocks={};verified={}
    for nid in ('OOFFICE','DKITCHEN'):
        r=noise_meta[nid];assert sha(r['path'])==r['sha256'];a,sr=sf.read(r['path'],dtype='float64',start=125*16000,stop=175*16000)
        assert sr==16000 and len(a)==50*16000;blocks[nid]=highpass40(a)
    @lru_cache(maxsize=60)
    def kernel(template,kind):
        r=next(r for r in rirs if r['template_id']==template and r['kind']==kind)
        assert sha(r['path'])==r['sha256'];verified[r['path']]=r['sha256'];return np.load(r['path'])
    rows=[]
    with (BASE/'manifest.jsonl').open('x',encoding='utf-8') as log:
        for si,s in enumerate(sources):
            path=SOURCES/s['path'];assert sha(path)==s['sha256'];mono,sr=sf.read(path,dtype='float64')
            assert sr==16000 and len(mono)==s['samples']<=56320
            angle=(-30,30)[si%2];nid=('OOFFICE','DKITCHEN')[si%2];relative=(-80,25)[si%2]
            for rt in plan['requests']:
                template=next(t for t in old['templates'] if t['split']=='validation' and t['rt']==rt and t['angle']==angle)
                clean=full_speech(mono,kernel(template['id'],'speech'));frames=len(clean);assert frames==len(mono)+7679<=64000
                activity=speech_activity_mask(clean[:,4])
                for field in plan['fields']:
                    assert time.monotonic()-start<plan['max_seconds'];n=np.zeros_like(clean);intervals=[];gain=0.;snr=None
                    if field!='clean':
                        k=1 if field=='point' else 8;length=16000+frames
                        for i in range(k):
                            a=int((1+48*i/k)*16000);b=int((1+48*(i+1)/k)*16000)
                            offset=int(rng.integers(a,b-length+1));clip=blocks[nid][offset:offset+length]
                            kind=('point_minus80' if relative==-80 else 'point_plus25') if k==1 else f'multi{i}'
                            n+=continuous_noise(clip,kernel(template['id'],kind),16000,frames)/np.sqrt(k)
                            intervals.append([125*16000+offset,125*16000+offset+length])
                        snr=0 if field=='point' else -5;x,_,gain=mix_at_snr(clean,n,snr,activity)
                    else:x=clean.copy()
                    scale=min(1.,.95/max(float(np.abs(x).max()),float(np.abs(clean).max())))
                    xx=(x*scale).astype('float32');yy=(clean[:,4]*scale).astype('float32')
                    sid=f"g32_{len(rows):03d}_{s['source_id']}";xp=BASE/'noisy'/(sid+'.wav');yp=BASE/'reference'/(sid+'.wav')
                    for p,a in ((xp,xx),(yp,yy)):
                        with p.open('xb') as f:sf.write(f,a,16000,format='WAV',subtype='FLOAT')
                        z,rr=sf.read(p,dtype='float32');assert rr==16000 and np.array_equal(z,a)
                    error=None if snr is None else abs(snr_db(yy,xx[:,4]-yy,activity)-snr);assert error is None or error<=.001
                    row={'sample_id':sid,'source_id':s['source_id'],'speaker_id':s['speaker_id'],'transcript':s['transcript'],
                        'split':'complete_old_development_supplement','ASR_scoring_allowed':True,'full_source':True,'source_crop_start':0,
                        'source_path':str(path),'source_sha256':s['sha256'],'original_source_samples':len(mono),'frames':frames,
                        'condition':field,'noise_field':field,'noise_id':None if field=='clean' else nid,'snr_db':snr,
                        'relative_angle':relative if field=='point' else None,'target_azimuth':angle,'requested_rt60':rt,'room_id':'dev0',
                        'template_id':template['id'],'noisy_path':str(xp),'reference_path':str(yp),'input_sha256':sha(xp),
                        'reference_sha256':sha(yp),'input_samples_sha256':hashlib.sha256(xx.tobytes()).hexdigest(),
                        'reference_samples_sha256':hashlib.sha256(yy.tobytes()).hexdigest(),'common_scale':scale,'noise_gain':gain,
                        'noise_intervals':intervals,'preroll':16000,'saved_active_SNR_error_db':error}
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
            log.flush();status(stage='data',sources_done=si+1,total=4,cases=len(rows))
    assert len(rows)==36 and all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    dump(OUT/'used_RIR_hashes.json',verified,True)
    complete={'status':'ok','utc':now(),'cases':36,'noisy':24,'clean':12,'sources':4,'speakers':2,
        'unique_reference_word_tokens':plan['unique_reference_word_tokens'],'manifest_sha256':sha(BASE/'manifest.jsonl'),
        'used_RIRs':len(verified),'used_RIR_hashes_sha256':sha(OUT/'used_RIR_hashes.json'),'elapsed_seconds':time.monotonic()-start}
    dump(OUT/'DATA_COMPLETE.json',complete,True)
    report=ROOT/'docs/G32B_完整开发语句生成覆盖.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G32B 完整开发语句生成\n\n四条整句×三个RT×三种条件，共36条，24带噪/12干净；未裁剪源语音，保留完整RIR尾，长度均≤64000。所有FLOAT WAV回读样本相同，原来源及已使用RIR哈希核对。每条活动SNR、噪声全区间和共同尺度保存。尚未运行增强、ASR或训练。先完整重放验收再生成468个系统输出。\n')
    archive('data_complete',report,[OUT/'DATA_COMPLETE.json',BASE/'manifest.jsonl',OUT/'used_RIR_hashes.json'])
    status(stage='data_complete_pending_acceptance');print(complete,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():dump(OUT/f'data_failure_{time.time_ns()}.json',{'utc':now(),'error':repr(e)},True)
        traceback.print_exc();raise
