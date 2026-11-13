#!/usr/bin/env python3
"""All fixed neural checkpoints and filters, complete development waveforms only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import gc
import json
import runpy
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import soundfile as sf
import torch
from g32_common import ROOT,OUT,BASE,TRAIN,sha,read,dump,now,meta,archive,status,data_guard
from g32_worker import initialize,save_and_score,filtered
from mcse.adaptation import attach_lora,load_adapter


def guard():
    p=meta(OUT/'signal_plan.json');data_guard()
    for n,h in p['hashes'].items():assert sha(ROOT/n)==h,n
    for r in p['registered'].values():assert sha(ROOT/r['path'])==r['sha256']
    return p


def main():
    assert meta(OUT/'DATA_ACCEPTANCE_AUDIT.json')['status']=='ok' and meta(TRAIN/'FINAL_TRAINING_ACCEPTANCE.json')['status']=='ok'
    names=['scripts/g32_signal.py','scripts/g32_worker.py','scripts/g32_signal_finish.py','scripts/g32_common.py',
        'scripts/mcnet_infer.py','src/mcse/adaptation.py','src/mcse/beamforming.py','src/mcse/wiener.py','src/mcse/evaluation.py',
        'data/g32_v1/manifest.jsonl','artifacts/g32_complete_dev_v1/DATA_ACCEPTANCE_AUDIT.json',
        'artifacts/g30_room_objectives_v1/FINAL_TRAINING_ACCEPTANCE.json']
    p={'utc':now(),'max_seconds':1800,'cases':36,'outputs':468,'FP32':True,'batch':1,'workers':2,
       'registered':meta(TRAIN/'registered_checkpoints.json'),'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'signal_plan.json',p,True)
    report=ROOT/'docs/G32C_完整开发语句信号评价协议.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G32C 完整开发语句信号协议\n\n固定36条件，原带噪参考36行，官方McNet及G30三个登记1500步的神经组各直接/MVDR/MWF共432行，合计468。所有模型、成熟滤波器与参数不改，FP32 batch1 GPU推理，两个CPU评分/滤波进程，1800秒上限。\n\n全部输入完整短句加RIR尾，不做额外裁剪、自动电平恢复、对齐拟合或重采样。保持原电平音频及SI-SDR/PESQ/STOI，干净输入另算RMS/NMSE。此阶段不训练、不跑ASR，不用这些旧开发结果重选G30参数或改判G31门槛。\n')
    archive('signal_protocol',report,[OUT/'signal_plan.json']+[ROOT/n for n in names[:9]])
    start=time.monotonic();cases=read(BASE/'manifest.jsonl');assert len(cases)==36
    for c in cases:assert c['full_source'] and c['ASR_scoring_allowed'] and c['frames']<=64000
    torch.set_num_threads(2);torch.manual_seed(20261113)
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g32_runtime');repo=ROOT/'third_party/McNet';cp=repo/rt['OFFLINE_CHECKPOINT'];rt['verify_upstream'](repo,cp)
    direct=[];filters=[];resources=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        with (OUT/'direct_signals.jsonl').open('x',encoding='utf-8') as log:
            pending=[]
            def collect():
                for future in pending:
                    r=future.result();direct.append(r);log.write(json.dumps(r,allow_nan=False)+'\n')
                pending.clear();log.flush()
            for c in cases:
                x,sr=sf.read(c['noisy_path'],dtype='float32');assert sr==16000 and sha(c['noisy_path'])==c['input_sha256']
                r={**c,'system':'noisy','source_system':'noisy','method':'direct','oracle_clean_information':False,'deployable':True,'inference_seconds':0.}
                pending.append(pool.submit(save_and_score,(r,x[:,4].copy())))
                if len(pending)>=8:collect()
            collect()
            for system in ('mcnet','cirm','cirm_mask','cirm_shape'):
                guard();model,icirm,metadata=rt['load_model'](repo,cp,torch.device('cuda'))
                if system!='mcnet':
                    registered=p['registered'][system];state=torch.load(ROOT/registered['path'],map_location='cpu',weights_only=False)
                    assert state['variant']==system and state['step']==1500;assert attach_lora(model)==3584;load_adapter(model,state['adapter'])
                model.eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
                for i,c in enumerate(cases,1):
                    assert time.monotonic()-start<p['max_seconds']
                    a,sr=sf.read(c['noisy_path'],dtype='float32');assert sr==16000 and a.shape==(c['frames'],6)
                    x=torch.from_numpy(np.ascontiguousarray(a.T))[None].cuda();before=time.perf_counter()
                    with torch.inference_mode():
                        prediction,data=model(x);y=rt['reconstruct_waveform'](model,icirm,x,prediction.to(torch.complex64),c['frames'])[0].cpu().numpy()
                    seconds=time.perf_counter()-before
                    r={**c,'system':system,'source_system':system,'method':'direct','oracle_clean_information':False,'deployable':True,'inference_seconds':seconds}
                    pending.append(pool.submit(save_and_score,(r,y)))
                    if len(pending)>=8:collect()
                    if i%12==0:status(stage='signal_direct',system=system,done=i,total=36)
                collect();resource={'system':system,'cases':36,'elapsed_seconds':time.monotonic()-started,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'model_metadata':metadata,'registered':p['registered'].get(system)}
                resources.append(resource);dump(OUT/(system+'_resource.json'),resource,True)
                report=ROOT/'docs'/f'G32_{system}_整句直接输出覆盖.md'
                with report.open('x',encoding='utf-8') as f:f.write(f'# G32 {system} 直接输出覆盖\n\n36个冻结完整开发条件全部推理、保存FLOAT并回读评分。固定模型与参数不变，未作有利子集/系统选择。资源与参数来源保存；固定滤波与识别尚待后续。\n')
                archive('direct_'+system,report,[OUT/(system+'_resource.json')])
                del model,icirm,prediction,data,x,y;gc.collect();torch.cuda.empty_cache()
        assert len(direct)==180
        report=ROOT/'docs/G32D_整句直接输出完整覆盖.md'
        with report.open('x',encoding='utf-8') as f:f.write('# G32D 整句直接输出完整覆盖\n\n原带噪与四个冻结神经系统各36条，180直接输出完成。保留全部原波形和信号指标，尚无ASR分数或方法总体结论。\n')
        archive('direct_complete',report,[OUT/'direct_signals.jsonl'])
        jobs=[(r,m) for r in direct if r['system']!='noisy' for m in ('mvdr','mwf')]
        with (OUT/'filter_signals.jsonl').open('x',encoding='utf-8') as log:
            for offset in range(0,len(jobs),24):
                assert time.monotonic()-start<p['max_seconds']
                for r in pool.map(filtered,jobs[offset:offset+24]):filters.append(r);log.write(json.dumps(r,allow_nan=False)+'\n')
                log.flush();status(stage='signal_filters',done=len(filters),total=288)
    guard();assert len(filters)==288
    with (OUT/'signals.jsonl').open('x',encoding='utf-8') as f:
        for r in direct+filters:f.write(json.dumps(r,allow_nan=False)+'\n')
    dump(OUT/'runtime_resources.json',resources,True)
    dump(OUT/'SIGNAL_COMPLETE.json',{'status':'ok','utc':now(),'rows':468,'signals_sha256':sha(OUT/'signals.jsonl'),
        'elapsed_seconds':time.monotonic()-start,'ASR_performed':False,'training_performed':False},True)
    from g32_signal_finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception as e:dump(OUT/'signal_failure.json',{'utc':now(),'error':repr(e)});traceback.print_exc();raise
    finally:(OUT/'signal_keep_awake_stop').write_text('G32 signal evaluation ended\n')
    for _ in range(30):
        if (OUT/'power_signal_helper.json').exists() and meta(OUT/'power_signal_helper.json')['status']=='released':break
        time.sleep(2)
    from g32_signal_finish import accept
    accept();print('G32 full complete-utterance signal stage accepted',flush=True)
