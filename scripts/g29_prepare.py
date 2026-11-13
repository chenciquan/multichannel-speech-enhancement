#!/usr/bin/env python3
"""Small balanced room data, separated source roles and independently filtered blocks."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import itertools
import json
import time
import traceback
from collections import Counter,defaultdict
from functools import lru_cache
from importlib.metadata import version
import numpy as np
import soundfile as sf
import rir_generator as rir
from g29_common import ROOT,OUT,BASE,SOURCES,sha,read,dump,now,meta,archive,status
from mcse.reverberation import full_speech,continuous_noise,schroeder_t20
from mcse.simulation import speech_activity_mask,mix_at_snr,snr_db
from mcse.spatial_noise import highpass40,direction,sphere_directions
from mcse.geometry import rotate_azimuth


def make_rir(job,plan):
    receiver=np.array(job['reference'])+(np.array(plan['positions'])-np.array(plan['positions'])[4])
    room=np.array(job['dimensions']);source=np.array(job['reference'])+plan['source_distance']*np.array(job['direction'])
    assert ((receiver>0)&(receiver<room)).all() and ((source>0)&(source<room)).all()
    request=job['rt'];kwargs={'beta':np.zeros(6)} if request==0 else {'reverberation_time':request}
    surface=2*(room[0]*room[1]+room[0]*room[2]+room[1]*room[2])
    absorption=None if request==0 else float(24*np.log(10)*np.prod(room)/(343*request*surface))
    assert absorption is None or 0<absorption<1,'Requested RT infeasible for this Sabine room'
    h=rir.generate(c=343,fs=16000,r=receiver,s=source,L=room,nsample=7680,mtype=rir.mtype.omnidirectional,
        order=-1,dim=3,hp_filter=False,**kwargs)*(4*np.pi*plan['source_distance'])
    assert h.shape==(7680,6) and np.isfinite(h).all()
    arrival=np.linalg.norm(receiver-source,axis=1)*16000/343
    error=float(np.abs(np.argmax(np.abs(h),axis=0)-arrival).max()) if request==0 else None
    if request==0:assert error<=2
    t20=schroeder_t20(h[:,4]) if request else None
    if request:assert t20 is not None and .01<t20<2
    tail=float(np.sum(h[-320:,4]**2)/np.sum(h[:,4]**2));assert tail<plan['tail_energy_limit']
    path=BASE/'rir'/f"{job['id']:03d}.npy"
    with path.open('xb') as f:np.save(f,h)
    saved=np.load(path);assert np.array_equal(saved,h)
    return {**job,'path':str(path),'sha256':sha(path),'source_position':source.tolist(),
        'receivers':receiver.tolist(),'RT0_arrival_error_max_samples':error,
        'measured_T20_extrapolated_RT60':t20,'last20ms_energy_fraction':tail,'Sabine_absorption_for_requested_RT':absorption}


def generate_records(plan,sources):
    rng=np.random.default_rng(plan['seed']);roles={s:[r for r in sources if r['split']==s] for s in ('train','validation')}
    speakers=sorted({r['speaker_id'] for r in roles['train']});assert len(speakers)==4 and len(roles['validation'])==12
    by_speaker={s:[r for r in roles['train'] if r['speaker_id']==s] for s in speakers}
    t_train=[t for t in plan['templates'] if t['split']=='train'];t_dev=[t for t in plan['templates'] if t['split']=='validation']
    cells=list(itertools.product(speakers,[t['id'] for t in t_train],('point_minus80','point_plus25','multi8_a','multi8_b')))
    assert len(cells)==288
    records=[]
    for repeat in range(2):
        for cell in rng.permutation(len(cells)):
            speaker,template,field=cells[int(cell)];source=by_speaker[speaker][int(rng.integers(len(by_speaker[speaker])))]
            records.append({'index':len(records),'split':'train','repeat':repeat,'template_id':template,
                'source_id':source['source_id'],'speaker_id':speaker,'source_path':str(SOURCES/source['path']),
                'source_sha256':source['sha256'],'source_samples':source['samples'],
                'crop_policy':'seeded random non-silent16000 samples, up to30attempts',
                'field':'multi8' if field.startswith('multi8') else 'point','relative_azimuth':None if field.startswith('multi8') else -80 if field=='point_minus80' else 25,
                'noise_id':plan['noise_ids'][len(records)%3],'snr_db':plan['snrs'][(len(records)//3)%3]})
    assert len(records)==576
    for si,source in enumerate(roles['validation']):
        angle=(-30,30)[si%2]
        for rt in plan['RT_requests']:
            template=next(t['id'] for t in t_dev if t['angle']==angle and t['rt']==rt)
            for field in ('point','multi8','clean'):
                records.append({'index':sum(r['split']=='validation' for r in records),'split':'validation','template_id':template,
                    'source_id':source['source_id'],'speaker_id':source['speaker_id'],'source_path':str(SOURCES/source['path']),
                    'source_sha256':source['sha256'],'source_samples':source['samples'],'crop_policy':'center16000 samples',
                    'field':field,'relative_azimuth':(-80,25)[si%2] if field=='point' else None,
                    'noise_id':None if field=='clean' else ('OOFFICE','DKITCHEN')[si%2],
                    'snr_db':None if field=='clean' else 0 if field=='point' else -5,
                    'transcript_role':'not_an_ASR_target_partial_crop','ASR_scoring_allowed':False})
    assert sum(r['split']=='validation' for r in records)==108
    return records


def main():
    assert meta(ROOT/'artifacts/g28_gradient_feasibility_v2/FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    assert (ROOT/'artifacts/g28_gradient_feasibility_v2/stages/next_room_development_decision/archive_manifest.json').exists()
    OUT.mkdir(exist_ok=False);BASE.mkdir(exist_ok=False);(BASE/'rir').mkdir()
    sources=read(SOURCES/'speech_manifest.jsonl');noise_meta=meta(SOURCES/'noise_manifest.json')
    positions=meta(ROOT/'configs/g1_geometry.json')['microphone_positions_m']
    rooms=[{'id':name,'dimensions':dims,'reference':[dims[0]/2,dims[1]/2,1.4],'split':split}
        for name,dims,split in (('train0',[5,4,3],'train'),('train1',[7,5.5,3.2],'train'),
                                ('train2',[4.5,3.8,2.8],'train'),('dev0',[5.5,4.6,3.1],'validation'))]
    templates=[]
    for room in rooms:
        for rt in (0,.16,.28):
            for angle in ((-50,50) if room['split']=='train' else (-30,30)):
                templates.append({**room,'room_id':room['id'],'id':len(templates),'rt':rt,'angle':angle})
    names=['scripts/g29_prepare.py','scripts/g29_common.py','configs/g1_geometry.json','src/mcse/geometry.py',
           'src/mcse/reverberation.py','src/mcse/spatial_noise.py','src/mcse/simulation.py',
           'data/g3_v1/sources/speech_manifest.jsonl','data/g3_v1/sources/noise_manifest.json',
           'artifacts/g12_room_stress_v2/ASSET_COMPLETE.json','artifacts/g28_gradient_feasibility_v2/FINAL_ACCEPTANCE_AUDIT.json']
    plan={'utc':now(),'seed':2026111301,'max_seconds':2400,'positions':positions,'source_distance':1.2,
        'RT_requests':[0,.16,.28],'RIR_samples':7680,'dry_frames':16000,'frames':23679,'preroll':16000,
        'tail_energy_limit':1e-4,'SNR_tolerance_db':.001,'RT0_room_rir_tolerance':1e-9,
        'train_examples':576,'development_examples':108,'rooms':rooms,'templates':templates,
        'noise_ids':['DWASHING','DKITCHEN','OOFFICE'],'snrs':[-5,0,5],
        'noise_time_blocks_seconds':{'train':[0,120],'validation':[125,175]},
        'noise_preprocessing':'Each split block independently highpass40, no shared filter state,1s inner margin; full preroll+frames intervals recorded',
        'target':'full reverberant reference channel4 float32; no dry target, quantization, resampling, alignment fitting',
        'sampling':'2 repeats of4speakers x18roomRTangle templates x(point-80,point+25,multi8,multi8), shuffled separately',
        'development':'12existing validation sources centered1sec,3RTx2noisefields+clean, angle alternates bysource; noWER',
        'new_training':False,'new_ASR':False,'hashes':{n:sha(ROOT/n) for n in names}}
    jobs=[]
    for template in templates:
        vectors=[('speech',direction(template['angle'],0)),('point_minus80',direction(template['angle']-80,-15)),
                 ('point_plus25',direction(template['angle']+25,-15))]
        vectors += [(f'multi{i}',rotate_azimuth(u,template['angle'])) for i,u in enumerate(sphere_directions(8))]
        for name,u in vectors:jobs.append({**template,'id':len(jobs),'template_id':template['id'],'kind':name,'direction':u.tolist()})
    assert len(jobs)==264
    dump(OUT/'frozen_plan.json',plan,True);dump(OUT/'RIR_jobs.json',jobs,True)
    protocol=ROOT/'docs/G29A_小规模混响训练开发数据冻结协议.md'
    with protocol.open('x',encoding='utf-8') as f:f.write('# G29A 小规模混响训练/开发数据冻结协议\n\n训练576条：四位旧训练说话人×三个新训练房间×三个请求RT(0/.16/.28)×两目标角度(−50/50)×点−80/点+25/八源/八源×两次，所有方法共用。开发108条：12条旧开发角色语句，各中心裁剪1秒，三个RT×(点源/八源/干净)，独立开发房间与±30°。开发裁剪不得按整句转写评WER。不是新说话人最终确认。\n\n目标为参考麦全混响语音，保留0.48秒RIR卷积尾，输入23679样本。保存FLOAT32训练/开发数组，不修改旧数据。用现有rir-generator0.3.0，先12项物理预检归档，再完整264项RIR，再渲染数据。无反射跨训练房间RIR差须≤1e−9，尾20ms能量占比<1e−4；请求RT与实测T20分别保存。最短正RT取.16而非.12，是执行前按最大房间Sabine吸收率可行性选择，未查看模型分数。\n\n噪声按训练[0,120]秒与开发[125,175]秒的独立块分别高通，防止零相位滤波跨分区。片段在块内留1秒边界；每个源完整记录预滚1秒和输出尾长度所用区间。八源使用同一噪声录音的八个不重叠时间子块，不称理想扩散场或真实阵列录音。防削波按每例混合及全六通道目标共同尺度，活动SNR误差≤.001dB。\n\n先核对语句/说话人/hash角色隔离及源文件，再生成。训练源随机裁剪最多30次仅排除静音，开发固定中心裁剪。数据阶段不读任何模型分数、不训练、不解码ASR、不调损失系数；2400秒上限，失败保留停止。\n')
    archive('protocol',protocol,[OUT/'frozen_plan.json',OUT/'RIR_jobs.json',ROOT/'scripts/g29_prepare.py',ROOT/'scripts/g29_common.py'])
    started=time.monotonic();asset=meta(ROOT/'artifacts/g12_room_stress_v2/ASSET_COMPLETE.json')
    assert version('rir-generator')=='0.3.0' and all(sha(p)==h for p,h in asset['installed_module_files_sha256'].items())
    roles={split:[r for r in sources if r['split']==split] for split in ('train','validation','test')}
    assert len(roles['train'])==212 and len(roles['validation'])==12
    rolecheck={split:{'sources':len(rs),'speakers':sorted({r['speaker_id'] for r in rs}),'sha256':sorted(r['sha256'] for r in rs)} for split,rs in roles.items()}
    for a,b in itertools.combinations(roles,2):
        assert not set(rolecheck[a]['speakers'])&set(rolecheck[b]['speakers'])
        assert not set(rolecheck[a]['sha256'])&set(rolecheck[b]['sha256'])
    used={r['source_id']:r for r in roles['train']+roles['validation']}
    for r in used.values():assert sha(SOURCES/r['path'])==r['sha256']
    dump(OUT/'source_role_audit.json',rolecheck,True)
    rir_rows={}
    probe=[j for j in jobs if j['room_id'] in ('train0','dev0') and j['angle']==(-50 if j['split']=='train' else -30) and j['kind'] in ('speech','point_minus80')]
    assert len(probe)==12
    for j in probe:rir_rows[j['id']]=make_rir(j,plan)
    dump(OUT/'RIR_preflight.json',list(rir_rows.values()),True)
    report=ROOT/'docs/G29A_房间物理预检与角色隔离.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G29A 房间预检与角色隔离\n\n12项固定RIR通过坐标、有限值、无反射到达时间、反射衰减及尾能量检查。既有训练212句/四人、开发12句/两人，与旧test角色的说话人及原文件哈希互不重叠；训练/开发源文件哈希均核对。不读取旧test语音，不根据模型输出选源。角色与RIR明细已保存。\n')
    archive('physical_preflight',report,[OUT/'source_role_audit.json',OUT/'RIR_preflight.json'])
    for j in jobs:
        assert time.monotonic()-started<plan['max_seconds']
        if j['id'] not in rir_rows:rir_rows[j['id']]=make_rir(j,plan)
        if len(rir_rows)%24==0:status(stage='RIR',done=len(rir_rows),total=264)
    rirs=[rir_rows[i] for i in range(264)];dump(OUT/'RIR_manifest.json',rirs,True)
    zero=defaultdict(list)
    for r in rirs:
        if r['split']=='train' and r['rt']==0:zero[r['angle'],r['kind']].append(r)
    errors=[]
    for key,rs in zero.items():
        assert len(rs)==3;h=np.load(rs[0]['path'])
        error=max(float(np.abs(np.load(r['path'])-h).max()) for r in rs)
        assert error<=plan['RT0_room_rir_tolerance'];errors.append({'angle':key[0],'kind':key[1],'max_abs':error})
    dump(OUT/'RT0_RIR_room_controls.json',errors,True)
    report=ROOT/'docs/G29B_完整RIR与无反射控制.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G29B 完整RIR与无反射控制\n\n264个RIR均已保存并回读核对，22组相同方向/声源在三个训练房间的无反射RIR差符合冻结1e−9界限。请求RT与实测T20外推、尾能量和源/阵列坐标逐项保留。仅验证模拟器及当前坐标约定，不证明真实房间匹配。\n')
    archive('physical_complete',report,[OUT/'RIR_manifest.json',OUT/'RT0_RIR_room_controls.json'])
    records=generate_records(plan,sources);dump(OUT/'planned_records.json',records,True)
    noise_blocks={};noise_sources={r['noise_id']:r for r in noise_meta}
    for nid in plan['noise_ids']:
        record=noise_sources[nid];assert sha(record['path'])==record['sha256']
        raw,sr=sf.read(record['path'],dtype='float64');assert sr==16000
        for split,(lo,hi) in plan['noise_time_blocks_seconds'].items():noise_blocks[nid,split]=highpass40(raw[lo*16000:hi*16000])
    @lru_cache(maxsize=32)
    def source(sid):
        r=used[sid];a,sr=sf.read(SOURCES/r['path'],dtype='float64');assert sr==16000 and len(a)==r['samples'];return a
    @lru_cache(maxsize=32)
    def kernel(template,kind):
        r=next(r for r in rirs if r['template_id']==template and r['kind']==kind);assert sha(r['path'])==r['sha256'];return np.load(r['path'])
    rng=np.random.default_rng(plan['seed']+1)
    def render(record):
        mono=source(record['source_id']);attempts=0
        if record['split']=='train':
            for attempts in range(1,31):
                crop=int(rng.integers(0,len(mono)-16000+1));piece=mono[crop:crop+16000]
                if np.sqrt(np.mean(piece**2))>1e-4:break
            else:raise ValueError('30 silent source crops')
        else:crop=(len(mono)-16000)//2;piece=mono[crop:crop+16000]
        assert len(piece)==16000 and np.sqrt(np.mean(piece**2))>1e-4
        s=full_speech(piece,kernel(record['template_id'],'speech'));activity=speech_activity_mask(s[:,4]);intervals=[]
        n=np.zeros_like(s);gain=0.;lo,hi=plan['noise_time_blocks_seconds'][record['split']]
        if record['field']!='clean':
            count=1 if record['field']=='point' else 8;block=noise_blocks[record['noise_id'],record['split']]
            for i in range(count):
                a=int((1+(hi-lo-2)*i/count)*16000);b=int((1+(hi-lo-2)*(i+1)/count)*16000)
                length=plan['preroll']+plan['frames'];offset=int(rng.integers(a,b-length+1))
                clip=block[offset:offset+length]
                kind=('point_minus80' if record['relative_azimuth']==-80 else 'point_plus25') if count==1 else f'multi{i}'
                n+=continuous_noise(clip,kernel(record['template_id'],kind),plan['preroll'],plan['frames'])/np.sqrt(count)
                intervals.append([lo*16000+offset,lo*16000+offset+length])
            x,_,gain=mix_at_snr(s,n,record['snr_db'],activity)
        else:x=s.copy()
        scale=min(1.,.95/max(float(np.abs(x).max()),float(np.abs(s).max())))
        xx=(x*scale).astype('float32');ss=(s*scale).astype('float32');yy=ss[:,4]
        error=None if record['field']=='clean' else abs(snr_db(yy,xx[:,4]-yy,activity)-record['snr_db'])
        assert error is None or error<=plan['SNR_tolerance_db']
        assert xx.shape==ss.shape==(23679,6) and np.isfinite(xx).all() and np.isfinite(ss).all()
        row={**record,'sample_id':f"g29_{record['split']}_{record['index']:04d}",'frames':23679,
            'source_crop_start':crop,'crop_attempts':attempts,'noise_intervals':intervals,'noise_gain':gain,'common_scale':scale,
            'saved_active_SNR_error_db':error,'input_samples_sha256':__import__('hashlib').sha256(xx.T.copy().tobytes()).hexdigest(),
            'reference_samples_sha256':__import__('hashlib').sha256(yy.copy().tobytes()).hexdigest()}
        return row,xx.T,yy
    summaries={}
    for split,count in (('train',576),('validation',108)):
        xmap=np.lib.format.open_memmap(BASE/(split+'_x.npy'),mode='w+',dtype='float32',shape=(count,6,23679))
        ymap=np.lib.format.open_memmap(BASE/(split+'_y.npy'),mode='w+',dtype='float32',shape=(count,23679));done=[]
        with (BASE/(split+'_manifest.jsonl')).open('x',encoding='utf-8') as log:
            for record in (r for r in records if r['split']==split):
                assert time.monotonic()-started<plan['max_seconds']
                row,x,y=render(record);i=row['index'];xmap[i]=x;ymap[i]=y
                assert np.array_equal(xmap[i],x) and np.array_equal(ymap[i],y)
                log.write(json.dumps(row,allow_nan=False)+'\n');done.append(row)
                if len(done)%24==0:log.flush();status(stage='render',split=split,done=len(done),total=count)
        xmap.flush();ymap.flush();del xmap,ymap
        summaries[split]={'rows':len(done),'unique_sources':len({r['source_id'] for r in done}),
            'speaker_counts':dict(Counter(r['speaker_id'] for r in done)),'field_counts':dict(Counter(r['field'] for r in done)),
            'max_SNR_error_db':max(r['saved_active_SNR_error_db'] or 0 for r in done)}
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    files=[BASE/(s+suffix) for s in ('train','validation') for suffix in ('_x.npy','_y.npy','_manifest.jsonl')]
    complete={'status':'ok','utc':now(),'training_or_ASR_performed':False,'summaries':summaries,'RIRs':264,
        'hashes':{str(p.relative_to(ROOT)):sha(p) for p in files},'elapsed_seconds':time.monotonic()-started}
    dump(OUT/'DATA_COMPLETE.json',complete,True)
    report=ROOT/'docs/G29C_共享混响训练开发数据完成报告.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G29C 共享混响训练/开发数据完成\n\n训练576条、开发108条，23679采样点(约1.48秒)，264项RIR。目标/混合共同尺度、活动SNR、完整噪声区间和源裁剪均逐例保存。训练/开发分别独立滤波，原始来源角色及说话人隔离已核对。数据不等于独立语句数，实际源覆盖及各条件计数见DATA_COMPLETE。\n\n开发是旧验证角色的1秒中心裁剪，不许按整句转写评WER；用于有限开发，不作新说话人最终确认。全部方法未来共用这份固定数据和相同随机抽样，不修改旧训练池。此阶段无模型训练、ASR或分数选择。下一步先独立验收再登记训练与目标系数。\n')
    archive('data_complete',report,[OUT/'DATA_COMPLETE.json',OUT/'planned_records.json']+files[-1:])
    status(stage='data_complete_pending_acceptance');print(complete['summaries'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():dump(OUT/'failure.json',{'utc':now(),'error':repr(e)})
        traceback.print_exc();raise
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('G29 data preparation ended\n')
