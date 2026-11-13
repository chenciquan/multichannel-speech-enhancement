#!/usr/bin/env python3
"""Replay every saved training/development input and target before model access."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import hashlib
import itertools
import json
import time
import traceback
from collections import Counter
from functools import lru_cache
import numpy as np
import soundfile as sf
from g29_common import ROOT,OUT,BASE,SOURCES,sha,read,dump,now,meta,archive,status
from mcse.reverberation import full_speech,continuous_noise
from mcse.simulation import speech_activity_mask,snr_db
from mcse.spatial_noise import highpass40


def audit():
    assert meta(OUT/'power_helper.json')['status']=='released'
    done=meta(OUT/'DATA_COMPLETE.json');plan=meta(OUT/'frozen_plan.json');assert done['status']=='ok'
    frozen={'utc':now(),'max_seconds':1200,'data_complete_sha256':sha(OUT/'DATA_COMPLETE.json'),
        'plan_sha256':sha(OUT/'frozen_plan.json'),'code_sha256':sha(ROOT/'scripts/g29_accept.py'),
        'scope':'All684 inputs and labels exact reconstruction from sources, RIRs, intervals,gain,scale; all split and source checks'}
    dump(OUT/'audit_frozen_plan.json',frozen,True)
    protocol=ROOT/'docs/G29D_完整输入与标签重放验收协议.md'
    with protocol.open('x',encoding='utf-8') as f:f.write('# G29D 完整数据重放验收协议\n\n核对所有数据文件及264RIR哈希、各阶段快照和源角色。按冻结裁剪位置、独立噪声时间块、完整预滚区间、RIR、噪声增益及共同尺度，重建全部576训练+108开发的六通道输入和目标；要求FLOAT32逐位相同。重新计算保存样本活动SNR、检查完整区间分区、八源片段互不重叠、训练条件因子平衡和开发裁剪禁用整句ASR。1200秒上限，无模型访问、训练、ASR或参数选择。\n')
    archive('audit_protocol',protocol,[OUT/'audit_frozen_plan.json',ROOT/'scripts/g29_accept.py'])
    started=time.monotonic()
    for p,h in done['hashes'].items():assert sha(ROOT/p)==h,p
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h,n
    rirs=meta(OUT/'RIR_manifest.json');assert len(rirs)==264
    for r in rirs:assert sha(r['path'])==r['sha256']
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
    source_rows=read(SOURCES/'speech_manifest.jsonl');by_id={r['source_id']:r for r in source_rows}
    roles={s:{r['speaker_id'] for r in source_rows if r['split']==s} for s in ('train','validation','test')}
    for a,b in itertools.combinations(roles,2):assert not roles[a]&roles[b]
    noise={r['noise_id']:r for r in meta(SOURCES/'noise_manifest.json')}
    @lru_cache(maxsize=32)
    def speech(sid):
        r=by_id[sid];path=SOURCES/r['path'];assert sha(path)==r['sha256']
        a,sr=sf.read(path,dtype='float64');assert sr==16000 and len(a)==r['samples'];return a
    @lru_cache(maxsize=264)
    def kernel(template,kind):return np.load(next(r['path'] for r in rirs if r['template_id']==template and r['kind']==kind))
    @lru_cache(maxsize=6)
    def block(nid,split):
        r=noise[nid];assert sha(r['path'])==r['sha256'];lo,hi=plan['noise_time_blocks_seconds'][split]
        a,sr=sf.read(r['path'],dtype='float64',start=lo*16000,stop=hi*16000);assert sr==16000 and len(a)==(hi-lo)*16000
        return highpass40(a)
    all_records=[];summary={}
    for split,count in (('train',576),('validation',108)):
        records=read(BASE/(split+'_manifest.jsonl'));assert len(records)==count and [r['index'] for r in records]==list(range(count))
        xmap=np.load(BASE/(split+'_x.npy'),mmap_mode='r');ymap=np.load(BASE/(split+'_y.npy'),mmap_mode='r')
        assert xmap.shape==(count,6,23679) and ymap.shape==(count,23679) and xmap.dtype==ymap.dtype==np.dtype('float32')
        errors=[]
        for r in records:
            assert time.monotonic()-started<frozen['max_seconds']
            sid=r['source_id'];origin=by_id[sid];assert origin['split']==split and origin['speaker_id']==r['speaker_id']
            assert origin['sha256']==r['source_sha256'] and r['source_sha256']==sha(r['source_path'])
            mono=speech(sid);start=r['source_crop_start'];piece=mono[start:start+16000]
            assert 0<=start<=len(mono)-16000 and len(piece)==16000 and np.sqrt(np.mean(piece**2))>1e-4
            if split=='validation':assert start==(len(mono)-16000)//2 and r['ASR_scoring_allowed'] is False
            s=full_speech(piece,kernel(r['template_id'],'speech'));n=np.zeros_like(s)
            assert s.shape==(23679,6)
            intervals=r['noise_intervals'];expected=0 if r['field']=='clean' else 1 if r['field']=='point' else 8
            assert len(intervals)==expected
            lo,hi=plan['noise_time_blocks_seconds'][split]
            for i,(a,b) in enumerate(intervals):
                assert (lo+1)*16000<=a<b<=(hi-1)*16000 and b-a==39679
                for c,d in intervals[:i]:assert b<=c or d<=a
                kind=('point_minus80' if r['relative_azimuth']==-80 else 'point_plus25') if expected==1 else f'multi{i}'
                clip=block(r['noise_id'],split)[a-lo*16000:b-lo*16000]
                n+=continuous_noise(clip,kernel(r['template_id'],kind),16000,23679)/np.sqrt(expected)
            x=(s+r['noise_gain']*n)*r['common_scale'];y=s[:,4]*r['common_scale']
            x=x.T.astype('float32');y=y.astype('float32');i=r['index']
            assert np.array_equal(xmap[i],x) and np.array_equal(ymap[i],y),'Reconstruction differs'
            assert hashlib.sha256(x.tobytes()).hexdigest()==r['input_samples_sha256']
            assert hashlib.sha256(y.tobytes()).hexdigest()==r['reference_samples_sha256']
            assert np.isfinite(x).all() and np.isfinite(y).all() and np.abs(x).max()<=.950001
            if expected:
                error=abs(snr_db(y,x[4]-y,speech_activity_mask(y))-r['snr_db']);assert error<=plan['SNR_tolerance_db'];errors.append(error)
            else:assert r['noise_gain']==0 and np.array_equal(x[4],y)
            if (i+1)%24==0:status(stage='full_replay_audit',split=split,done=i+1,total=count)
        summary[split]={'rows':count,'exact_input_replays':count,'exact_target_replays':count,
            'speaker_counts':dict(Counter(r['speaker_id'] for r in records)),'field_counts':dict(Counter(r['field'] for r in records)),
            'max_recomputed_activity_SNR_error_db':max(errors),'unique_sources':len({r['source_id'] for r in records})}
        all_records.extend(records)
    train=[r for r in all_records if r['split']=='train'];assert len({r['sample_id'] for r in all_records})==684
    # Both multi8 repetitions remain explicit, without pretending they are independent utterances.
    cell_counts=Counter((r['speaker_id'],r['template_id'],r['field'],r['relative_azimuth']) for r in train)
    assert len(cell_counts)==216 and set(cell_counts.values())=={2,4}
    for (sp,t,f,a),n in cell_counts.items():assert n==(4 if f=='multi8' else 2)
    assert set(Counter(r['speaker_id'] for r in train).values())=={144}
    assert set(Counter(r['noise_id'] for r in train).values())=={192}
    assert set(Counter(r['snr_db'] for r in train).values())=={192}
    all_sources={split:{r['source_id'] for r in all_records if r['split']==split} for split in ('train','validation')}
    assert not all_sources['train']&all_sources['validation']
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    result={'status':'ok','utc':now(),'summary':summary,'RIRs_verified':264,'full_replays':684,
        'frozen_audit_sha256':sha(OUT/'audit_frozen_plan.json'),'data_complete_sha256':sha(OUT/'DATA_COMPLETE.json'),
        'training_or_ASR_performed':False,'elapsed_seconds':time.monotonic()-started}
    dump(OUT/'NUMERICAL_ACCEPTANCE.json',result,True)


def accept():
    result=meta(OUT/'NUMERICAL_ACCEPTANCE.json');assert result['status']=='ok' and result['full_replays']==684
    assert meta(OUT/'power_audit_helper.json')['status']=='released'
    stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    audit={**result,'utc':now(),'status':'ok','stages':stages,'numerical_acceptance_sha256':sha(OUT/'NUMERICAL_ACCEPTANCE.json'),
        'data_and_audit_power_released':True,'code_sha256':sha(ROOT/'scripts/g29_accept.py')}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G29_共享混响训练开发数据完整验收.md'
    lines=['# G29 共享混响训练/开发数据完整验收','','全部576训练及108开发输入/目标按保存来源、RIR、裁剪、独立滤波时间块、预滚区间、增益和共同尺度逐位重放通过。264项RIR及数据文件、源角色和阶段快照核对通过。无模型训练或ASR，临时保持唤醒已释放。', '',
        '|角色|记录|实际语句|说话人数|最大重算活动SNR误差dB|','|---|---:|---:|---:|---:|']
    for s,r in result['summary'].items():lines.append(f"|{s}|{r['rows']}|{r['unique_sources']}|{len(r['speaker_counts'])}|{r['max_recomputed_activity_SNR_error_db']:.8f}|")
    lines+=['','训练每位说话人144条件，各噪声/各SNR192条件；点源两类各144、八源288。重复条件和随机裁剪不当作更多独立说话人。开发来自两个旧开发角色说话人、12条语句的中心裁剪，禁用整句WER，不是独立最终确认。', '',
        '本轮高通分别在训练/开发各自噪声时间块内计算，完整源区间及1秒内边界均核验，避免跨块滤波取样。这个约定不同于旧阶段对整条噪声先滤波的流程，所有新对照共用本轮数据；不替换或重写旧阶段成绩。', '',
        '后续在这些训练数据内进行梯度尺度校准，参数保持不变，再冻结相同预算的原cIRM、cIRM加简单掩码监督、cIRM加统计监督对照。系数不从G22/G27或本轮开发成绩选择；有限开发成功后才扩大训练种子与新确认。创新性与真实阵列迁移均未确立。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',OUT/'NUMERICAL_ACCEPTANCE.json',ROOT/'scripts/g29_accept.py',OUT/'power_audit_helper.json'])
    status(stage='accepted');print({'status':'accepted','replays':684,'summary':result['summary']},flush=True)


if __name__=='__main__':
    try:audit()
    except Exception as e:dump(OUT/'audit_failure.json',{'utc':now(),'error':repr(e)});traceback.print_exc();raise
    finally:(OUT/'audit_keep_awake_stop').write_text('G29 full replay audit ended\n')
    for _ in range(30):
        if (OUT/'power_audit_helper.json').exists() and meta(OUT/'power_audit_helper.json')['status']=='released':break
        time.sleep(2)
    accept()
