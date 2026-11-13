#!/usr/bin/env python3
"""Full input/target replay and complete-transcript provenance before inference."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from functools import lru_cache
import hashlib
import numpy as np
import soundfile as sf
from g32_common import ROOT,OUT,BASE,SOURCES,RIR,sha,read,dump,now,meta,archive,status,data_guard
from mcse.reverberation import full_speech,continuous_noise
from mcse.spatial_noise import highpass40
from mcse.simulation import speech_activity_mask,snr_db


def main():
    plan=data_guard();done=meta(OUT/'DATA_COMPLETE.json');assert done['status']=='ok' and sha(BASE/'manifest.jsonl')==done['manifest_sha256']
    assert sha(OUT/'used_RIR_hashes.json')==done['used_RIR_hashes_sha256']
    for p,h in meta(OUT/'used_RIR_hashes.json').items():assert sha(p)==h
    sources={s['source_id']:s for s in plan['sources']};rirs=meta(RIR/'RIR_manifest.json');noise={r['noise_id']:r for r in meta(SOURCES/'noise_manifest.json')}
    @lru_cache(maxsize=4)
    def source(sid):
        s=sources[sid];assert sha(SOURCES/s['path'])==s['sha256'];a,sr=sf.read(SOURCES/s['path'],dtype='float64');assert sr==16000 and len(a)==s['samples'];return a
    @lru_cache(maxsize=60)
    def kernel(template,kind):return np.load(next(r['path'] for r in rirs if r['template_id']==template and r['kind']==kind))
    @lru_cache(maxsize=2)
    def block(nid):
        n=noise[nid];assert sha(n['path'])==n['sha256'];a,sr=sf.read(n['path'],dtype='float64',start=125*16000,stop=175*16000);assert sr==16000;return highpass40(a)
    records=read(BASE/'manifest.jsonl');assert len(records)==len({r['sample_id'] for r in records})==36;errors=[]
    for r in records:
        s=sources[r['source_id']];assert r['full_source'] and r['source_crop_start']==0 and r['ASR_scoring_allowed']
        assert r['transcript']==s['transcript'] and r['source_sha256']==s['sha256'] and r['speaker_id']==s['speaker_id']
        clean=full_speech(source(s['source_id']),kernel(r['template_id'],'speech'));assert len(clean)==r['frames']==s['samples']+7679<=64000
        n=np.zeros_like(clean);intervals=r['noise_intervals'];k=0 if r['condition']=='clean' else 1 if r['condition']=='point' else 8;assert len(intervals)==k
        for i,(a,b) in enumerate(intervals):
            assert 126*16000<=a<b<=174*16000 and b-a==16000+r['frames']
            assert all(b<=c or d<=a for c,d in intervals[:i])
            kind=('point_minus80' if r['relative_angle']==-80 else 'point_plus25') if k==1 else f'multi{i}'
            n+=continuous_noise(block(r['noise_id'])[a-125*16000:b-125*16000],kernel(r['template_id'],kind),16000,r['frames'])/np.sqrt(k)
        x=((clean+r['noise_gain']*n)*r['common_scale']).astype('float32');y=(clean[:,4]*r['common_scale']).astype('float32')
        assert sha(r['noisy_path'])==r['input_sha256'] and sha(r['reference_path'])==r['reference_sha256']
        xx,sr=sf.read(r['noisy_path'],dtype='float32');yy,rr=sf.read(r['reference_path'],dtype='float32')
        assert sr==rr==16000 and np.array_equal(x,xx) and np.array_equal(y,yy)
        assert hashlib.sha256(xx.tobytes()).hexdigest()==r['input_samples_sha256'] and hashlib.sha256(yy.tobytes()).hexdigest()==r['reference_samples_sha256']
        if k:
            error=abs(snr_db(yy,xx[:,4]-yy,speech_activity_mask(yy))-r['snr_db']);assert error<=.001;errors.append(error)
        else:assert r['noise_gain']==0 and np.array_equal(xx[:,4],yy)
    data_guard();stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    audit={'status':'ok','utc':now(),'exact_input_replays':36,'exact_target_replays':36,'full_transcript_cases':36,
        'unique_original_word_tokens':plan['unique_reference_word_tokens'],'max_recomputed_SNR_error_db':max(errors),
        'used_RIRs_verified':done['used_RIRs'],'stages':stages,'manifest_sha256':sha(BASE/'manifest.jsonl'),
        'training_or_ASR_performed':False,'long_running_awake_helper_requested':False}
    dump(OUT/'DATA_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G32_完整开发语句数据验收.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G32 完整开发语句数据验收\n\n36条六通道输入及参考按原始整句、固定RIR、完整噪声区间、增益和尺度全部逐位重放；原整句转写、长度及不裁剪来源核对通过。噪声区间分区与八源不重叠、活动SNR、源/波形/RIR及阶段快照哈希通过。无模型、训练或ASR；短数据作业未申请长时间保持唤醒。\n\n四条原句总计16词token，条件重复不能扩充有效文本或说话人样本。下一步保持全部四模型与三种输出，不挑有利系统；该补充开发诊断不覆盖G31主门槛。\n')
    archive('data_acceptance',report,[OUT/'DATA_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g32_data_accept.py'])
    status(stage='data_accepted');print(audit,flush=True)


if __name__=='__main__':main()
