"""Independent fixed-1500-update baseline; no validation/test loading or selection."""
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
from g34_common import ROOT,OUT,DATA,sha,dump,now,meta,guard,configure,state_digest
from mcse.adaptation import attach_lora,adapter_state,mask_loss


def main(run_id):
    plan=guard();run=next(r for r in plan['runs'] if r['id']==run_id)
    checked=meta(OUT/'INITIAL_COMPLETE.json');assert checked['status']=='ok' and checked['wave_passes']==6
    folder=OUT/'train'/run_id;folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();backend=configure(torch);assert backend==checked['backend'] and np.__version__=='1.26.4'
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g34_train_runtime');repo=ROOT/'third_party/McNet';cp=repo/rt['OFFLINE_CHECKPOINT']
    torch.manual_seed(plan['seed']);model,_,_=rt['load_model'](repo,cp,torch.device('cuda'))
    if run['mode']=='lora':assert attach_lora(model)==3584
    else:model.requires_grad_(True)
    model.train();initial={'base':state_digest(model.state_dict(),'base'),'all':state_digest(model.state_dict()),
        'adapter':state_digest(model.state_dict(),'adapter') if run['mode']=='lora' else None}
    assert initial==checked['initial'][run['mode']]
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];parameters=[p for _,p in named]
    assert sum(p.numel() for p in parameters)==run['parameters']
    optimizer=torch.optim.Adam(parameters,lr=run['lr'],betas=tuple(plan['betas']),eps=plan['eps'],weight_decay=plan['weight_decay'])
    xmap=np.load(DATA/'train_x.npy',mmap_mode='r');ymap=np.load(DATA/'train_y.npy',mmap_mode='r');indices=np.load(OUT/'sample_indices.npy')
    assert xmap.shape==(576,6,23679) and ymap.shape==(576,23679) and indices.shape==(1500,) and indices.min()>=0 and indices.max()<576
    torch.manual_seed(plan['seed']);np.random.seed(plan['seed'])
    config={'utc':now(),'run':run,'initial':initial,'backend':backend,'plan_sha256':sha(OUT/'frozen_plan.json'),
        'schedule_sha256':sha(OUT/'sample_indices.npy'),'initial_check_sha256':sha(OUT/'INITIAL_COMPLETE.json'),
        'trainable_names':[n for n,p in named],'no_development_or_test_audio':True}
    dump(folder/'config.json',config,True);torch.cuda.reset_peak_memory_stats();history=[];used=[];times=[]
    with (folder/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step,index in enumerate(indices,1):
            assert time.monotonic()-started<plan['max_run_seconds'],'Run deadline exceeded'
            x=torch.from_numpy(np.array(xmap[index],copy=True))[None].cuda();target=torch.from_numpy(np.array(ymap[index],copy=True))[None].cuda()
            torch.cuda.synchronize();t=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            prediction,data=model(x);loss=mask_loss(model,x,target,prediction,data);assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,plan['gradient_clip'],error_if_nonfinite=True)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            optimizer.step();assert all(torch.isfinite(p).all() for p in parameters)
            torch.cuda.synchronize();seconds=time.perf_counter()-t;times.append(seconds);used.append(int(index))
            record={'step':step,'index':int(index),'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm),
                'synchronized_step_seconds':seconds,'all_trainable_gradients_connected_finite':True,'all_updated_parameters_finite':True}
            log.write(json.dumps(record,allow_nan=False)+'\n')
            if step%50==0:
                log.flush();dump(folder/'status.json',{'utc':now(),'status':'training','step':step,'elapsed_seconds':time.monotonic()-started,
                    'loss':record['loss'],'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            if step in plan['checkpoint_steps']:
                path=folder/f'checkpoint_{step:05d}.pt';assert not path.exists()
                state=adapter_state(model) if run['mode']=='lora' else {n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
                torch.save({'weights':state,'optimizer':optimizer.state_dict(),'step':step,'run':run,'seed':plan['seed'],
                    'base_checkpoint_sha256':sha(cp),'schedule_sha256':config['schedule_sha256'],'plan_sha256':config['plan_sha256'],
                    'backend':backend,'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state()},path)
                history.append({'step':step,'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
                print({'run':run_id,'step':step,'elapsed_seconds':time.monotonic()-started},flush=True)
            del x,target,prediction,data,loss
    assert used==indices.tolist() and len(used)==1500
    final_base=state_digest(model.state_dict(),'base');final_all=state_digest(model.state_dict());assert final_all!=initial['all']
    if run['mode']=='lora':assert final_base==initial['base'] and all(p.grad is None for p in model.parameters() if not p.requires_grad)
    else:assert final_base!=initial['base']
    assert len(optimizer.state)==len(parameters) and all(int(s['step'])==1500 for s in optimizer.state.values())
    guard();result={'utc':now(),'status':'ok','run':run,'steps':1500,'optimizer_updates':1500,'initial':initial,
        'final_base_digest':final_base,'final_all_digest':final_all,'final_saved_weight_digest':state_digest(adapter_state(model) if run['mode']=='lora' else model.state_dict()),
        'backend':backend,'actual_indices_sha256':hashlib.sha256(np.array(used,dtype=np.int64).tobytes()).hexdigest(),
        'schedule_sha256':config['schedule_sha256'],'checkpoints':history,'registered':history[-1],
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'GPU_total_bytes':torch.cuda.get_device_properties(0).total_memory,'warm_step_median_seconds':float(np.median(times[1:])),
        'synchronized_step_seconds_sum':float(sum(times)),'elapsed_seconds':time.monotonic()-started,
        'no_development_or_test_audio':True,'training_log_sha256':sha(folder/'steps.jsonl')}
    dump(folder/'result.json',result,True);dump(folder/'status.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_id');args=parser.parse_args()
    try:main(args.run_id)
    except Exception as e:
        folder=OUT/'train'/args.run_id
        if folder.exists():dump(folder/'failure.json',{'utc':now(),'error':repr(e)})
        traceback.print_exc();raise
