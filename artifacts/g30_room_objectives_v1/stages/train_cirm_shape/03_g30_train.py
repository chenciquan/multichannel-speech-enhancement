#!/usr/bin/env python3
"""One fixed-objective run with no validation/test or checkpoint selection."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import json
import runpy
import time
import traceback
import numpy as np
import torch
from g30_common import ROOT,OUT,DATA,sha,dump,now,meta,guard
from g28_gradient_v2 import digest,stft,ratio,statistic_loss
from mcse.adaptation import attach_lora,adapter_state,mask_loss


def main(variant):
    plan=guard();assert variant in plan['variants']
    calibration=meta(OUT/'CALIBRATION_COMPLETE.json');assert calibration['status']=='ok'
    assert (OUT/'stages/calibration/archive_manifest.json').exists()
    assert sha(OUT/'sample_indices.npy')==calibration['schedule_sha256']
    folder=OUT/'train'/variant;folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();assert torch.__version__=='2.5.1+cu121' and np.__version__=='1.26.4'
    torch.set_num_threads(2);torch.manual_seed(plan['seed']);np.random.seed(plan['seed']);torch.backends.cudnn.benchmark=False
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g30_train_runtime')
    repo=ROOT/'third_party/McNet';checkpoint=repo/rt['OFFLINE_CHECKPOINT'];rt['verify_upstream'](repo,checkpoint)
    model,icirm,_=rt['load_model'](repo,checkpoint,torch.device('cuda'));assert attach_lora(model)==3584;model.train()
    initial={'base':digest(model),'adapter':digest(model,True)};assert initial==calibration['initial_parameter_digests']
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=plan['learning_rate'])
    xmap=np.load(DATA/'train_x.npy',mmap_mode='r');ymap=np.load(DATA/'train_y.npy',mmap_mode='r')
    indices=np.load(OUT/'sample_indices.npy');assert indices.shape==(1500,) and indices.min()>=0 and indices.max()<576
    config={'variant':variant,'seed':plan['seed'],'steps':1500,'batch':1,'learning_rate':plan['learning_rate'],
        'initial_digests':initial,'calibration_sha256':sha(OUT/'CALIBRATION_COMPLETE.json'),'plan_sha256':sha(OUT/'frozen_plan.json'),
        'schedule_sha256':sha(OUT/'sample_indices.npy'),'coefficients':calibration['coefficients'],'development_or_test_access':False,
        'registration_step':1500,'selection':False,'started_utc':now()}
    dump(folder/'config.json',config,True);torch.cuda.reset_peak_memory_stats();history=[];window=[];used=[]
    with (folder/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step,index in enumerate(indices,1):
            assert time.monotonic()-started<plan['max_run_seconds'],'Training deadline exceeded'
            x=torch.from_numpy(np.array(xmap[index],copy=True))[None].cuda();target=torch.from_numpy(np.array(ymap[index],copy=True))[None].cuda()
            optimizer.zero_grad(set_to_none=True);prediction,data=model(x)
            reconstruction=mask_loss(model,x,target,prediction,data);aux=torch.zeros((),device='cuda')
            if variant!='cirm':
                X=stft(x[0]);truth=ratio(stft(target[0]),X[4])
                y=rt['reconstruct_waveform'](model,icirm,x,prediction.to(torch.complex64),x.shape[-1])[0];m=ratio(stft(y),X[4])
                aux=(m-truth).square().mean() if variant=='cirm_mask' else statistic_loss(X,m,truth)
            coefficient=0. if variant=='cirm' else calibration['coefficients']['mask' if variant=='cirm_mask' else 'shape']
            loss=reconstruction+coefficient*aux
            assert torch.isfinite(loss) and torch.isfinite(reconstruction) and torch.isfinite(aux)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,plan['gradient_clip'],error_if_nonfinite=True)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            optimizer.step();assert all(torch.isfinite(p).all() for p in parameters)
            used.append(int(index));window.append([float(loss.detach()),float(reconstruction.detach()),float(aux.detach())])
            if step%50==0:
                means=np.mean(window,axis=0);window=[]
                record={'step':step,'loss_mean50':float(means[0]),'cirm_mean50':float(means[1]),'aux_mean50':float(means[2]),
                    'gradient_norm_before_clip':float(norm),'seconds':time.monotonic()-started,'last_index':int(index)}
                log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();dump(folder/'status.json',{'utc':now(),'status':'training',**record})
            if step in plan['checkpoint_steps']:
                path=folder/f'adapter_{step:05d}.pt';assert not path.exists()
                torch.save({'adapter':adapter_state(model),'optimizer':optimizer.state_dict(),'step':step,'variant':variant,
                    'seed':plan['seed'],'base_checkpoint_sha256':sha(checkpoint),'coefficient':coefficient,
                    'schedule_sha256':config['schedule_sha256'],'plan_sha256':config['plan_sha256'],
                    'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state()},path)
                history.append({'step':step,'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
                print({'variant':variant,'step':step,'elapsed_seconds':time.monotonic()-started},flush=True)
    assert used==indices.tolist() and len(used)==1500
    assert digest(model)==initial['base'] and all(p.grad is None for p in model.parameters() if not p.requires_grad)
    final_adapter=digest(model,True);assert final_adapter!=initial['adapter']
    guard()
    result={'status':'ok','utc':now(),'variant':variant,'steps':1500,'optimizer_updates':1500,
        'initial_digests':initial,'final_base_digest':digest(model),'final_adapter_digest':final_adapter,
        'schedule_sha256':sha(OUT/'sample_indices.npy'),'actual_indices_sha256':hashlib.sha256(np.array(used,dtype=np.int64).tobytes()).hexdigest(),
        'coefficients':calibration['coefficients'],'checkpoints':history,'registered':history[-1],
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'elapsed_seconds':time.monotonic()-started,'development_or_test_access':False,'training_log_sha256':sha(folder/'steps.jsonl')}
    dump(folder/'result.json',result,True);dump(folder/'status.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('variant');a=p.parse_args()
    try:main(a.variant)
    except Exception as e:
        folder=OUT/'train'/a.variant
        if folder.exists():dump(folder/'failure.json',{'utc':now(),'error':repr(e)})
        traceback.print_exc();raise
