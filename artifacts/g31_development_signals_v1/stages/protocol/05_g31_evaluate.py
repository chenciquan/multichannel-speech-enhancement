#!/usr/bin/env python3
"""Pre-registered development outputs; partial crops explicitly exclude ASR."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import gc
import hashlib
import json
import runpy
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import soundfile as sf
import torch
from g31_common import ROOT,OUT,TRAIN,DATA,sha,read,dump,now,meta,archive,guard,status
from g31_worker import initialize,save_and_score,filtered
from mcse.adaptation import attach_lora,load_adapter


def main():
    assert meta(TRAIN/'FINAL_TRAINING_ACCEPTANCE.json')['status']=='ok'
    protocol=meta(ROOT/'configs/g31_development_protocol.json');assert protocol['expected_signal_records']==1404 and protocol['ASR_allowed'] is False
    # The criterion file was pinned before calibration/training, not selected from development outputs.
    assert sha(ROOT/'configs/g31_development_protocol.json')==meta(TRAIN/'frozen_plan.json')['hashes']['configs/g31_development_protocol.json']
    OUT.mkdir(exist_ok=False);(OUT/'reference').mkdir()
    names=['scripts/g31_common.py','scripts/g31_worker.py','scripts/g31_evaluate.py','scripts/g31_finish.py','scripts/g30_common.py',
        'scripts/g29_common.py','scripts/g8_common.py','scripts/mcnet_infer.py','src/mcse/adaptation.py','src/mcse/beamforming.py',
        'src/mcse/wiener.py','src/mcse/evaluation.py','configs/g31_development_protocol.json','data/g29_v1/validation_x.npy',
        'data/g29_v1/validation_y.npy','data/g29_v1/validation_manifest.jsonl','artifacts/g30_room_objectives_v1/FINAL_TRAINING_ACCEPTANCE.json']
    plan={'utc':now(),'max_seconds':3600,'protocol':protocol,'registered':meta(TRAIN/'registered_checkpoints.json'),
        'FP32':True,'batch':1,'metric_workers':2,'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G31A_开发信号与干净语音预登记评价协议.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G31A 开发信号与干净语音评价\n\n评价规则已在G30任何训练前写入配置并冻结哈希。本阶段使用G29全部108开发条件，无ASR；原噪声直接输入108、四个神经系统各直接/MVDR/MWF共1296，合计1404记录。三个适配组统一G30登记1500步，不按开发成绩改选。\n\n主比较是72个带噪条件的MVDR：统计监督组对原重建组和简单掩码组均需平均SI-SDR提高至少.25dB、PESQ/STOI不降；36个干净MVDR条件的平均NMSE增量均不超过.02；每位开发说话人相对原重建组平均SI-SDR不降。此门槛仅决定是否扩展实验，不代表显著性或创新。直接/MWF及全部RT/场型/说话人结果保留，不能替换失败的主比较。\n\n12条旧验证语句、两位说话人的中心裁剪及模拟开发房间，不能按108个独立语音样本计算显著性，不做整句WER或新说话人最终确认。保存原电平FLOAT音频、相同信号指标、干净RMS/NMSE，无拟合对齐、自动幅度恢复或重采样。3600秒上限，不训练、不下载。\n')
    archive('protocol',report,[OUT/'frozen_plan.json',ROOT/'configs/g31_development_protocol.json']+[ROOT/n for n in names[:12]])
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261113)
    xmap=np.load(DATA/'validation_x.npy',mmap_mode='r');ymap=np.load(DATA/'validation_y.npy',mmap_mode='r')
    cases=[]
    for r in read(DATA/'validation_manifest.jsonl'):
        assert r['ASR_scoring_allowed'] is False and r['split']=='validation'
        x=np.array(xmap[r['index']],copy=True);y=np.array(ymap[r['index']],copy=True)
        assert hashlib.sha256(x.tobytes()).hexdigest()==r['input_samples_sha256']
        assert hashlib.sha256(y.tobytes()).hexdigest()==r['reference_samples_sha256']
        path=OUT/'reference'/(r['sample_id']+'.wav')
        with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
        z,sr=sf.read(path,dtype='float32');assert sr==16000 and np.array_equal(z,y)
        template=meta(ROOT/'artifacts/g29_room_development_v1/frozen_plan.json')['templates'][r['template_id']]
        cases.append({**r,'condition':'clean' if r['field']=='clean' else r['field'],'room_id':template['room_id'],
            'requested_rt60':template['rt'],'target_azimuth':template['angle'],'reference_path':str(path),
            'reference_sha256':sha(path),'oracle_clean_information':False,'deployable':True})
    dump(OUT/'cases.json',cases,True);guard()
    direct=[];filters=[];resources=[]
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g31_runtime');repo=ROOT/'third_party/McNet';checkpoint=repo/rt['OFFLINE_CHECKPOINT']
    rt['verify_upstream'](repo,checkpoint)
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        with (OUT/'direct_signals.jsonl').open('x',encoding='utf-8') as log:
            pending=[]
            def collect():
                for future in pending:
                    record=future.result();direct.append(record);log.write(json.dumps(record,allow_nan=False)+'\n')
                pending.clear();log.flush()
            for c in cases:
                r={**c,'system':'noisy','source_system':'noisy','method':'direct','inference_seconds':0.}
                pending.append(pool.submit(save_and_score,(r,np.array(xmap[c['index'],4],copy=True))))
                if len(pending)>=8:collect()
            collect()
            for system in ('mcnet','cirm','cirm_mask','cirm_shape'):
                assert time.monotonic()-started<plan['max_seconds']
                model,icirm,metadata=rt['load_model'](repo,checkpoint,torch.device('cuda'))
                if system!='mcnet':
                    registered=plan['registered'][system];assert sha(ROOT/registered['path'])==registered['sha256']
                    state=torch.load(ROOT/registered['path'],map_location='cpu',weights_only=False)
                    assert state['step']==1500 and state['variant']==system;assert attach_lora(model)==3584;load_adapter(model,state['adapter'])
                model.eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats();model_start=time.monotonic()
                for i,c in enumerate(cases,1):
                    assert time.monotonic()-started<plan['max_seconds']
                    x=torch.from_numpy(np.array(xmap[c['index']],copy=True))[None].cuda();before=time.perf_counter()
                    with torch.inference_mode():
                        prediction,data=model(x);y=rt['reconstruct_waveform'](model,icirm,x,prediction.to(torch.complex64),x.shape[-1])[0].cpu().numpy()
                    seconds=time.perf_counter()-before
                    r={**c,'system':system,'source_system':system,'method':'direct','inference_seconds':seconds}
                    pending.append(pool.submit(save_and_score,(r,y)))
                    if len(pending)>=8:collect()
                    if i%36==0:status(stage='direct',system=system,done=i,total=108)
                collect()
                resource={'system':system,'cases':108,'elapsed_seconds':time.monotonic()-model_start,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'model_metadata':metadata,'registered_checkpoint':plan['registered'].get(system)}
                resources.append(resource);dump(OUT/(system+'_resource.json'),resource,True)
                report=ROOT/'docs'/f'G31_{system}_直接输出覆盖.md'
                with report.open('x',encoding='utf-8') as f:f.write(f'# G31 {system} 直接输出覆盖\n\n108个固定开发条件全部生成并从FLOAT WAV回读评分；没有选择有利条件或按分数改checkpoint。后续固定滤波与完整比较尚待执行。本阶段峰值与逐例结果另存，暂不判定改进。\n')
                archive('direct_'+system,report,[OUT/(system+'_resource.json')])
                del model,icirm,prediction,data,x,y;gc.collect();torch.cuda.empty_cache()
        assert len(direct)==540
        report=ROOT/'docs/G31B_完整开发直接输出覆盖.md'
        with report.open('x',encoding='utf-8') as f:f.write('# G31B 完整开发直接输出\n\n原带噪参考及四个神经系统各108条件，共540行，直接输出全部完成。冻结参数与输入不变，未跑ASR。后续用各神经输出生成固定掩码MVDR/MWF，原质量成绩保留。\n')
        archive('direct_complete',report,[OUT/'direct_signals.jsonl',OUT/'cases.json'])
        jobs=[(r,m) for r in direct if r['system']!='noisy' for m in ('mvdr','mwf')]
        with (OUT/'filter_signals.jsonl').open('x',encoding='utf-8') as log:
            for offset in range(0,len(jobs),24):
                assert time.monotonic()-started<plan['max_seconds']
                for r in pool.map(filtered,jobs[offset:offset+24]):filters.append(r);log.write(json.dumps(r,allow_nan=False)+'\n')
                log.flush();status(stage='filters',done=len(filters),total=864)
    guard();assert len(filters)==864;dump(OUT/'runtime_resources.json',resources,True)
    with (OUT/'signals.jsonl').open('x',encoding='utf-8') as f:
        for r in direct+filters:f.write(json.dumps(r,allow_nan=False)+'\n')
    assert len({(r['system'],r['sample_id']) for r in direct+filters})==1404
    dump(OUT/'SIGNAL_COMPLETE.json',{'status':'ok','utc':now(),'rows':1404,'direct':540,'filters':864,
        'signals_sha256':sha(OUT/'signals.jsonl'),'elapsed_seconds':time.monotonic()-started,'ASR_performed':False},True)
    from g31_finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():dump(OUT/'failure.json',{'utc':now(),'error':repr(e)})
        traceback.print_exc();raise
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('G31 development evaluation ended\n')
    for _ in range(30):
        if (OUT/'power_helper.json').exists() and meta(OUT/'power_helper.json')['status']=='released':break
        time.sleep(2)
    from g31_finish import accept
    accept();print('G31 complete development signal stage accepted',flush=True)
