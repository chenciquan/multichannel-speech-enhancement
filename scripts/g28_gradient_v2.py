#!/usr/bin/env python3
"""Training-only differentiability and memory probe; no optimizer or parameter updates."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import hashlib
import json
import runpy
import time
import traceback
from collections import defaultdict
import numpy as np
import torch
import g8_common as common
from mcse.adaptation import attach_lora,adapter_state,mask_loss

ROOT=common.ROOT;OUT=ROOT/'artifacts/g28_gradient_feasibility_v2';BASE=ROOT/'data/g15_v1'
sha,read,dump,now=common.sha,common.read,common.dump,common.now


def meta(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def archive(stage,report,files):
    old=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=old


def stft(x):
    return torch.stft(x,n_fft=512,hop_length=128,win_length=512,
        window=torch.hann_window(512,dtype=x.dtype,device=x.device),center=True,pad_mode='reflect',return_complex=True)


def ratio(E,Xref):return E.abs().square()/(E.abs().square()+(Xref-E).abs().square()+1e-10)


def covariance(X,m):return torch.einsum('mft,nft,ft->fmn',X,X.conj(),m.to(dtype=X.dtype))/X.shape[-1]


def shape_loss(estimated,target):
    a=estimated.diagonal(dim1=-2,dim2=-1).real.sum(-1).clamp_min(1e-30)
    b=target.diagonal(dim1=-2,dim2=-1).real.sum(-1).clamp_min(1e-30)
    distance=(estimated/a[:,None,None]-target/b[:,None,None]).abs().square().sum((-2,-1))
    return (distance*b).sum()/b.sum().clamp_min(1e-30)


def statistic_loss(X,m,target):
    return .5*(shape_loss(covariance(X,m),covariance(X,target))+
               shape_loss(covariance(X,1-m),covariance(X,1-target)))


def digest(model,only_lora=False):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():
        if only_lora!=('lora_' in n):continue
        h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def finite_difference(X,target):
    X=X[:,:8,:48].cpu().to(torch.complex128);target=target[:8,:48].cpu().to(torch.float64)
    mask=torch.full_like(target,.5,requires_grad=True)
    loss=statistic_loss(X,mask,target);grad=torch.autograd.grad(loss,mask)[0]
    flat=int(grad.abs().argmax());index=np.unravel_index(flat,grad.shape);h=1e-5
    plus=mask.detach().clone();minus=plus.clone();plus[index]+=h;minus[index]-=h
    fd=float((statistic_loss(X,plus,target)-statistic_loss(X,minus,target))/(2*h));auto=float(grad[index])
    assert abs(auto)>1e-12 and abs(fd-auto)<1e-9+2e-4*abs(auto)
    zero=float(statistic_loss(X,target,target));assert abs(zero)<1e-20
    return {'index':[int(i) for i in index],'analytic_gradient':auto,'finite_difference_gradient':fd,
        'step':h,'absolute_error':abs(fd-auto),'identical_target_loss':zero,'dtype':'CPUcomplex128'}


def main():
    assert meta(ROOT/'artifacts/g27_oracle_mask_v1/FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    assert (ROOT/'artifacts/g27_oracle_mask_v1/stages/scientific_readback/archive_manifest.json').exists()
    OUT.mkdir(exist_ok=False)
    catalog=meta(BASE/'train_catalog.json');selected=[]
    for field in ('point','multi8'):
        for angle in (-60,0,60):selected.append(next(r for r in catalog if r['field']==field and r['angle']==angle))
    assert len({r['index'] for r in selected})==6
    manifest=read(ROOT/'data/g3_v1/sources/speech_manifest.jsonl');roles={r['source_id']:r['split'] for r in manifest}
    assert all(roles[r['source_id']]=='train' for r in selected)
    names=['scripts/g28_gradient_v2.py','scripts/mcnet_infer.py','src/mcse/adaptation.py','data/g15_v1/train_catalog.json',
           'data/g15_v1/train_x.npy','data/g15_v1/train_y.npy','data/g3_v1/sources/speech_manifest.jsonl',
           'artifacts/g27_oracle_mask_v1/FINAL_ACCEPTANCE_AUDIT.json']
    plan={'utc':now(),'max_seconds':600,'seed':20261113,'records':selected,'objectives':['original_cIRM','power_mask_MSE','masked_covariance_shape'],
        'optimizer_updates':0,'input':'six fixed existing G15 TRAIN records; no G22/G27 waveforms or scores opened',
        'target':'ratio of clean reference power, not full6ch true source covariance',
        'mask_shape':'energy-weighted trace-normalized Frobenius, S/N average, commonT',
        'dtype':'modelFP32,mask/statscomplex64; finite-differenceCPUcomplex128',
        'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G28A_v2_训练输入可微路径与显存预检协议.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G28A 可微路径与显存预检协议\n\n固定读取G15训练池中point/multi8分别在−60/0/60类别的首条记录，共6条；核对均属旧训练说话人和语句。无新数据，不读取G22/G27测试音频，不构造优化器，不更新参数。\n\n对相同初始LoRA、相同前向随机状态，分别求原压缩cIRM损失、真值功率比掩码MSE、真值掩码所生成的混合信号协方差形状损失的梯度。第三者标签不是实际语音六通道真值协方差；不称MISD或新发明。记录梯度有限性/范数/夹角与显存、耗时，不据此选择测试最优系数。\n\n先在固定训练输入的前8频点/48帧上，以CPUcomplex128做协方差形状损失的中心有限差分核对，再运行模型梯度。检查所有原参数和LoRA参数摘要逐位不变，无原主干梯度。600秒上限；失败保留，不自动改系数或漏掉记录。预检通过仅证明当前短段的可微性与资源可行，不代表能训练成功、4秒显存或识别收益。\n')
    archive('protocol',report,[OUT/'frozen_plan.json',ROOT/'scripts/g28_gradient_v2.py'])
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(plan['seed']);np.random.seed(plan['seed'])
    xmap=np.load(BASE/'train_x.npy',mmap_mode='r');ymap=np.load(BASE/'train_y.npy',mmap_mode='r')
    x0=torch.from_numpy(np.array(xmap[selected[0]['index']],copy=True)).double()
    s0=torch.from_numpy(np.array(ymap[selected[0]['index']],copy=True)).double()
    X0=stft(x0);S0=stft(s0);check=finite_difference(X0,ratio(S0,X0[4]))
    dump(OUT/'finite_difference.json',check,True)
    math_report=ROOT/'docs/G28A_v2_协方差形状损失梯度数值核对.md'
    with math_report.open('x',encoding='utf-8') as f:f.write('# G28A 梯度数值核对\n\n固定训练样本截取的有限差分与解析梯度通过预登记容差，目标与自身的损失为0。该检查不涉及任何参数更新，也不证明目标有益。详细数值见finite_difference.json。\n')
    archive('finite_difference',math_report,[OUT/'finite_difference.json'])
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g28_runtime')
    repo=ROOT/'third_party/McNet';checkpoint=repo/rt['OFFLINE_CHECKPOINT'];rt['verify_upstream'](repo,checkpoint)
    model,icirm,metadata=rt['load_model'](repo,checkpoint,torch.device('cuda'));assert attach_lora(model)==3584
    model.train();params=[p for p in model.parameters() if p.requires_grad]
    before={'base':digest(model),'adapter':digest(model,True)};results=[];gradients={}
    for r in selected:
        assert time.monotonic()-started<plan['max_seconds']
        x=torch.from_numpy(np.array(xmap[r['index']],copy=True))[None].cuda()
        target=torch.from_numpy(np.array(ymap[r['index']],copy=True))[None].cuda()
        X=stft(x[0]);S=stft(target[0]);oracle=ratio(S,X[4]);vectors={}
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state();prediction_hash=None
        for objective in plan['objectives']:
            model.zero_grad(set_to_none=True);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
            pred,data=model(x);p_hash=hashlib.sha256(pred.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            if prediction_hash is None:prediction_hash=p_hash
            else:assert prediction_hash==p_hash,'Forward must be identical across objective probes'
            if objective=='original_cIRM':loss=mask_loss(model,x,target,pred,data)
            else:
                y=rt['reconstruct_waveform'](model,icirm,x,pred.to(torch.complex64),x.shape[-1])[0]
                m=ratio(stft(y),X[4])
                loss=(m-oracle).square().mean() if objective=='power_mask_MSE' else statistic_loss(X,m,oracle)
            assert torch.isfinite(loss)
            grads=torch.autograd.grad(loss,params,allow_unused=False)
            assert all(torch.isfinite(g).all() for g in grads)
            vector=torch.cat([g.detach().cpu().double().flatten() for g in grads]);norm=float(torch.linalg.vector_norm(vector));assert norm>0
            torch.cuda.synchronize();vectors[objective]=vector.numpy()
            results.append({'index':r['index'],'source_id':r['source_id'],'field':r['field'],'angle':r['angle'],
                'objective':objective,'loss':float(loss.detach()),'gradient_norm':norm,'gradients_finite':True,
                'forward_sha256':p_hash,'seconds':time.perf_counter()-t,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'frames':int(x.shape[-1])})
            del loss,grads,pred,data
        gradients.update({f"record_{r['index']}_{k}":v for k,v in vectors.items()})
        for row in results[-3:]:
            a=vectors[row['objective']];row['cosine_to_original_cIRM']=float(a@vectors['original_cIRM']/(np.linalg.norm(a)*np.linalg.norm(vectors['original_cIRM'])))
        dump(OUT/'status.json',{'utc':now(),'stage':'gradient_probe','records':len(results)//3,'total':6})
    assert before=={'base':digest(model),'adapter':digest(model,True)}
    assert all(p.grad is None for p in model.parameters())
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    dump(OUT/'results.json',results,True);np.savez(OUT/'gradients.npz',**gradients)
    done={'status':'ok','utc':now(),'records':6,'objective_probes':18,'optimizer_updates':0,'parameter_digests_unchanged':before,
        'results_sha256':sha(OUT/'results.json'),'gradients_sha256':sha(OUT/'gradients.npz'),'elapsed_seconds':time.monotonic()-started}
    dump(OUT/'COMPLETE.json',done,True)
    report=ROOT/'docs/G28_v2_可微路径与训练显存预检结果.md'
    lines=['# G28 可微路径与训练显存预检结果','','6个既有训练记录×3目标的18次梯度计算完成；优化器更新为0，主干与LoRA参数摘要均保持不变。数学有限差分检查通过。', '',
           '|记录|场型/角度|目标|损失|梯度范数|与原目标梯度余弦|Torch峰值已分配GiB|','|---:|---|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['index']}|{r['field']}/{r['angle']}|{r['objective']}|{r['loss']:.6g}|{r['gradient_norm']:.6g}|{r['cosine_to_original_cIRM']:.4f}|{r['peak_allocated_bytes']/2**30:.3f}|")
    lines+=['','显存是本次16128采样点短段的Torch已分配峰值，不含驱动/其他进程，不外推4秒或大批量。不同目标损失与梯度尺度不一致，不能据损失数值更小或范数更大选赢家。余弦只描述这6个初始点，不能证明收敛、训练收益或识别表现。协方差目标指向真值掩码产生的混合统计，非真实六通道源统计。', '',
        '后续训练应先登记开发条件与同预算对照。混响数据、损失权重、采样和训练步数不能根据已查看的G22/G27测试分数选择；保留简单掩码监督和原重建目标。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('gradient_results',report,[OUT/'COMPLETE.json',OUT/'results.json',OUT/'gradients.npz',ROOT/'scripts/g28_gradient_v2.py'])
    print({'status':'gradient_probe_complete','peak_allocated_GiB':max(r['peak_allocated_bytes'] for r in results)/2**30},flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():dump(OUT/'failure.json',{'utc':now(),'error':repr(e)})
        traceback.print_exc();raise
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('G28 gradient probe ended\n')
