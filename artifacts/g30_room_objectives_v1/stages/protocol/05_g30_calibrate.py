#!/usr/bin/env python3
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import hashlib
import runpy
import time
import numpy as np
import torch
from g30_common import ROOT,OUT,DATA,sha,read,dump,now,meta,archive,guard,status
from g28_gradient_v2 import digest,stft,ratio,statistic_loss
from mcse.adaptation import attach_lora,mask_loss


def main():
    plan=guard();assert meta(OUT/'READY_FOR_CALIBRATION.json')['status']=='ok'
    started=time.monotonic();assert torch.__version__=='2.5.1+cu121' and np.__version__=='1.26.4'
    torch.set_num_threads(2);torch.manual_seed(plan['seed']);np.random.seed(plan['seed'])
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g30_calibration_runtime')
    repo=ROOT/'third_party/McNet';rt['verify_upstream'](repo,repo/rt['OFFLINE_CHECKPOINT'])
    model,icirm,_=rt['load_model'](repo,repo/rt['OFFLINE_CHECKPOINT'],torch.device('cuda'));assert attach_lora(model)==3584;model.train()
    params=[p for p in model.parameters() if p.requires_grad];before={'base':digest(model),'adapter':digest(model,True)}
    xmap=np.load(DATA/'train_x.npy',mmap_mode='r');ymap=np.load(DATA/'train_y.npy',mmap_mode='r')
    records=[];vectors={}
    for index in plan['calibration_indices']:
        assert time.monotonic()-started<plan['max_calibration_seconds']
        x=torch.from_numpy(np.array(xmap[index],copy=True))[None].cuda();s=torch.from_numpy(np.array(ymap[index],copy=True))[None].cuda()
        X=stft(x[0]);target=ratio(stft(s[0]),X[4]);rng_cpu=torch.get_rng_state();rng_gpu=torch.cuda.get_rng_state();forward=None;entry={'index':index,'objectives':{}}
        for name in ('cirm','mask','shape'):
            torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_gpu);torch.cuda.reset_peak_memory_stats()
            pred,data=model(x);key=hashlib.sha256(pred.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            if forward is None:forward=key
            else:assert forward==key
            if name=='cirm':loss=mask_loss(model,x,s,pred,data)
            else:
                y=rt['reconstruct_waveform'](model,icirm,x,pred.to(torch.complex64),x.shape[-1])[0];m=ratio(stft(y),X[4])
                loss=(m-target).square().mean() if name=='mask' else statistic_loss(X,m,target)
            assert torch.isfinite(loss)
            gradient=torch.autograd.grad(loss,params,allow_unused=False);assert all(torch.isfinite(g).all() for g in gradient)
            vector=torch.cat([g.detach().cpu().double().flatten() for g in gradient]).numpy();norm=float(np.linalg.norm(vector));assert norm>0
            entry['objectives'][name]={'loss':float(loss.detach()),'gradient_norm':norm,'forward_sha256':key,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            vectors[f'record_{index}_{name}']=vector;del loss,gradient,pred,data
        records.append(entry);status(stage='calibration',records=len(records),total=6)
    coefficients={aux:plan['calibration_aux_strength']*float(np.median([r['objectives']['cirm']['gradient_norm']/r['objectives'][aux]['gradient_norm'] for r in records])) for aux in ('mask','shape')}
    assert all(np.isfinite(v) and 0<v<1000 for v in coefficients.values())
    assert before=={'base':digest(model),'adapter':digest(model,True)} and all(p.grad is None for p in model.parameters())
    guard();dump(OUT/'calibration_records.json',records,True)
    with (OUT/'calibration_gradients.npz').open('xb') as f:np.savez(f,**vectors)
    complete={'status':'ok','utc':now(),'coefficients':coefficients,'records':6,'optimizer_updates':0,
        'initial_parameter_digests':before,'records_sha256':sha(OUT/'calibration_records.json'),
        'vectors_sha256':sha(OUT/'calibration_gradients.npz'),'plan_sha256':sha(OUT/'frozen_plan.json'),
        'schedule_sha256':sha(OUT/'sample_indices.npy'),'elapsed_seconds':time.monotonic()-started}
    dump(OUT/'CALIBRATION_COMPLETE.json',complete,True)
    report=ROOT/'docs/G30B_训练输入辅助目标尺度校准.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G30B 训练输入辅助目标尺度校准\n\n六条训练输入、18个目标梯度均有限非零，参数未更新；从冻结公式计算lambda_mask='+str(coefficients['mask'])+'，lambda_shape='+str(coefficients['shape'])+'。系数在任何优化器更新和开发评分前保存，未使用验证或旧测试成绩。\n\n主/辅助梯度的中位数尺度只控制初始化附近的典型强度，不保证每条样本、每一步或方向冲突受控，不声称最优值。相同系数用于整组1500步，全部梯度向量及源哈希归档。\n')
    archive('calibration',report,[OUT/'CALIBRATION_COMPLETE.json',OUT/'calibration_records.json',OUT/'calibration_gradients.npz',ROOT/'scripts/g30_calibrate.py'])
    status(stage='calibration_archived_training_not_started');print(complete,flush=True)


if __name__=='__main__':main()
