#!/usr/bin/env python3
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import runpy
import time
import traceback
import numpy as np
import torch
from g33_common import ROOT,OUT,DATA,sha,dump,now,meta,guard,status
from g28_gradient_v2 import digest
from mcse.adaptation import attach_lora,mask_loss


def all_digest(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(mode):
    plan=guard();assert mode in plan['modes'];folder=OUT/mode;folder.mkdir(exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(plan['seed']);torch.backends.cudnn.benchmark=False
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g33_runtime');repo=ROOT/'third_party/McNet';cp=repo/rt['OFFLINE_CHECKPOINT'];rt['verify_upstream'](repo,cp)
    model,_,metadata=rt['load_model'](repo,cp,torch.device('cuda'))
    if mode=='lora':assert attach_lora(model)==3584
    else:model.requires_grad_(True)
    model.train();parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    count=sum(p.numel() for _,p in parameters);base_digest=digest(model);initial=all_digest(model)
    assert (mode=='lora' and count==3584) or (mode=='full' and count==3328706)
    xmap=np.load(DATA/'train_x.npy',mmap_mode='r');ymap=np.load(DATA/'train_y.npy',mmap_mode='r')
    records=[];predictions={}
    for index in plan['indices']:
        assert time.monotonic()-started<plan['max_mode_seconds']
        model.zero_grad(set_to_none=True);torch.manual_seed(plan['seed']+index)
        x=torch.from_numpy(np.array(xmap[index],copy=True))[None].cuda();s=torch.from_numpy(np.array(ymap[index],copy=True))[None].cuda()
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();free,total=torch.cuda.mem_get_info();t=time.perf_counter()
        pred,data=model(x);loss=mask_loss(model,x,s,pred,data);assert torch.isfinite(loss)
        gradients=torch.autograd.grad(loss,[p for _,p in parameters],allow_unused=True);torch.cuda.synchronize()
        seconds=time.perf_counter()-t;allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        unused=[n for (n,p),g in zip(parameters,gradients) if g is None]
        if unused:dump(folder/'unconnected_parameters.json',{'index':index,'names':unused},True)
        assert not unused,'Requested trainable parameters unconnected; preserve and inspect'
        norms=[]
        for (name,p),g in zip(parameters,gradients):
            a=g.detach().cpu().double().numpy();assert np.isfinite(a).all();norms.append({'name':name,'parameters':p.numel(),'norm':float(np.linalg.norm(a))})
        norm=float(np.sqrt(sum(r['norm']**2 for r in norms)));assert np.isfinite(norm) and norm>0
        prediction=pred.detach().cpu().contiguous().numpy();predictions[f'record_{index}']=prediction
        records.append({'index':index,'mode':mode,'frames':int(x.shape[-1]),'loss':float(loss.detach()),
            'gradient_norm':norm,'connected_parameters':count,'parameter_grad_norms':norms,
            'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,'free_GPU_bytes_before':free,
            'GPU_total_bytes':total,'synchronized_compute_seconds':seconds,'forward_sha256':hashlib.sha256(prediction.tobytes()).hexdigest()})
        status(stage='gradient_probe',mode=mode,records=len(records),total=6)
        del pred,data,loss,gradients,x,s
    assert initial==all_digest(model) and base_digest==digest(model) and all(p.grad is None for p in model.parameters())
    guard();dump(folder/'records.json',records,True)
    with (folder/'initial_forward_outputs.npz').open('xb') as f:np.savez(f,**predictions)
    result={'status':'ok','utc':now(),'mode':mode,'records':6,'trainable_parameters':count,
        'total_parameters':sum(p.numel() for p in model.parameters()),'optimizer_updates':0,'optimizer_created':False,
        'original_parameters_unchanged':True,'all_state_digest':initial,'base_digest':base_digest,
        'two_FP32_Adam_moments_estimated_bytes':count*8,'metadata':metadata,
        'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in records),'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in records),
        'cold_first_compute_seconds':records[0]['synchronized_compute_seconds'],
        'warm5_compute_median_seconds':float(np.median([r['synchronized_compute_seconds'] for r in records[1:]])),
        'records_sha256':sha(folder/'records.json'),'forwards_sha256':sha(folder/'initial_forward_outputs.npz'),
        'elapsed_seconds':time.monotonic()-started,'development_or_test_audio_access':False}
    dump(folder/'COMPLETE.json',result,True);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('lora','full'));args=p.parse_args()
    try:main(args.mode)
    except Exception as e:
        folder=OUT/args.mode
        if folder.exists():dump(folder/'failure.json',{'utc':now(),'error':repr(e)})
        traceback.print_exc();raise
