#!/usr/bin/env python3
"""Locate forward differences without changing the failed G33 tolerance or any weights."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import hashlib
import json
import time
import traceback
from pathlib import Path
import g8_common as common
ROOT=common.ROOT;OUT=ROOT/'artifacts/g33_precision_audit_v1';OLD=ROOT/'artifacts/g33_full_gradient_probe_v1'
sha,read,dump,now=common.sha,common.read,common.dump,common.now


def meta(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})
def archive(stage,report,files):
    previous=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous


def main():
    assert meta(OLD/'PROBE_COMPLETE.json')['status']=='ok' and not (OLD/'FINAL_ACCEPTANCE_AUDIT.json').exists()
    assert (OLD/'stages/forward_acceptance_failure/archive_manifest.json').exists()
    old=meta(OLD/'frozen_plan.json');OUT.mkdir(exist_ok=False)
    profiles=[{'id':'cudnn_tf32_nondet','enabled':True,'tf32':True,'deterministic':False},
        {'id':'cudnn_fp32_nondet','enabled':True,'tf32':False,'deterministic':False},
        {'id':'cudnn_tf32_det','enabled':True,'tf32':True,'deterministic':True},
        {'id':'cudnn_fp32_det','enabled':True,'tf32':False,'deterministic':True},
        {'id':'native_fp32','enabled':False,'tf32':False,'deterministic':True}]
    names=['scripts/g33b_precision_audit.py','scripts/mcnet_infer.py','scripts/g28_gradient_v2.py','src/mcse/adaptation.py',
        'artifacts/g33_full_gradient_probe_v1/frozen_plan.json','artifacts/g33_full_gradient_probe_v1/PROBE_COMPLETE.json',
        'artifacts/g33_full_gradient_probe_v1/lora/initial_forward_outputs.npz','artifacts/g33_full_gradient_probe_v1/full/initial_forward_outputs.npz',
        'artifacts/g33_full_gradient_probe_v1/acceptance_failure_v1/forward_differences.json']
    hashes={n:sha(ROOT/n) for n in names}
    hashes.update({n:old['hashes'][n] for n in ('data/g29_v1/train_x.npy','data/g29_v1/train_manifest.jsonl')})
    plan={'utc':now(),'indices':old['indices'],'seed':old['seed'],'max_seconds_after_verification':900,
        'profiles':profiles,'model_modes':['full','lora'],'grad_enabled':[True,False],'repeats':2,
        'forwards':240,'rtol':old['forward_equivalence_rtol'],'atol':old['forward_equivalence_atol'],
        'training_mode':True,'matmul_allow_tf32':False,'cudnn_benchmark':False,
        'optimizer_updates':0,'no_development_or_test_audio':True,'hashes':hashes}
    dump(OUT/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G33B_自动求导与后端数值定位协议.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G33B 自动求导与后端数值定位协议\n\nG33原前向等价验收失败原样保留。本次固定六条训练输入、原CP与零残差LoRA，训练模式保持True；分别开启/关闭自动求导，每路径重复两次并重置种子。cuDNN启用时交叉TF32许可与deterministic开关共四种，再加无cuDNN/不许可TF32控制，总240次前向。matmulTF32始终False、benchmark始终False，不启用全局确定性算法，不改CUBLAS环境或任何系统配置。\n\n保存压缩cIRM与重建波形，比较同路径重复、两模型间、自动求导开关以及默认相似设置对旧G33输出的复现。原rtol1e−5/atol1e−6不变；本次完整收集并不代表任一预期差异必通过或原G33改判。记录实际后端初始/各配置及参数摘要，零参数更新、无优化器、无开发/测试评分。输入/源核验完成后900秒上限；失败与部分数组保留。\n')
    archive('protocol',report,[OUT/'frozen_plan.json',ROOT/'scripts/g33b_precision_audit.py'])
    status(stage='verifying_sources')
    for n,h in hashes.items():assert sha(ROOT/n)==h,n
    status(stage='loading_local_libraries')
    import numpy as np
    import torch
    import runpy
    from mcse.adaptation import attach_lora
    from g28_gradient_v2 import digest
    actual_defaults={'cudnn_enabled':torch.backends.cudnn.enabled,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'global_deterministic':torch.are_deterministic_algorithms_enabled(),
        'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'NVIDIA_TF32_OVERRIDE':os.environ.get('NVIDIA_TF32_OVERRIDE'),
        'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version()}
    assert actual_defaults['global_deterministic'] is False
    dump(OUT/'actual_initial_backend_settings.json',actual_defaults,True)
    torch.set_num_threads(2);torch.manual_seed(plan['seed']);started=time.monotonic()
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g33b_runtime');repo=ROOT/'third_party/McNet';cp=repo/rt['OFFLINE_CHECKPOINT'];rt['verify_upstream'](repo,cp)
    def state_hash(model):
        h=hashlib.sha256()
        for name,tensor in model.state_dict().items():h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    models={};decoders={};initial={};initial_all={}
    for mode in plan['model_modes']:
        torch.manual_seed(plan['seed']);model,icirm,_=rt['load_model'](repo,cp,torch.device('cuda'))
        if mode=='lora':assert attach_lora(model)==3584
        else:model.requires_grad_(True)
        model.train();models[mode]=model;decoders[mode]=icirm;initial[mode]=digest(model);initial_all[mode]=state_hash(model)
    assert initial['full']==initial['lora']
    assert all(float(p.detach().abs().max())==0 for n,p in models['lora'].named_parameters() if n.endswith('lora_B'))
    inputs=np.load(ROOT/'data/g29_v1/train_x.npy',mmap_mode='r');arrays={};waves={};records=[];actual_profiles=[]
    try:
        for profile in profiles:
            torch.backends.cudnn.enabled=profile['enabled'];torch.backends.cudnn.allow_tf32=profile['tf32']
            torch.backends.cudnn.deterministic=profile['deterministic'];torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False
            actual_profiles.append({'id':profile['id'],'cudnn_enabled':torch.backends.cudnn.enabled,
                'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
                'global_deterministic':torch.are_deterministic_algorithms_enabled()})
            for index in plan['indices']:
                x=torch.from_numpy(np.array(inputs[index],copy=True))[None].cuda()
                for mode in plan['model_modes']:
                    for enabled in plan['grad_enabled']:
                        for repeat in range(2):
                            assert time.monotonic()-started<plan['max_seconds_after_verification']
                            torch.manual_seed(plan['seed']+index);model=models[mode]
                            key=f"{profile['id']}__{index}__{mode}__grad{int(enabled)}__repeat{repeat}"
                            with torch.set_grad_enabled(enabled):pred,data=model(x)
                            with torch.no_grad():wave=rt['reconstruct_waveform'](model,decoders[mode],x,pred.to(torch.complex64),x.shape[-1])[0]
                            a=pred.detach().cpu().contiguous().numpy();w=wave.detach().cpu().numpy();assert np.isfinite(a).all() and np.isfinite(w).all()
                            arrays[key]=a;waves[key]=w;records.append({'key':key,'profile':profile['id'],'index':index,'mode':mode,'grad_enabled':enabled,'repeat':repeat,
                                'prediction_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'waveform_sha256':hashlib.sha256(w.tobytes()).hexdigest()})
                            del pred,data,wave
                del x;status(stage='forwards',profile=profile['id'],records=len(records),total=240)
        assert all(digest(models[m])==initial[m] for m in models)
        assert all(state_hash(models[m])==initial_all[m] for m in models)
        assert all(p.grad is None for m in models.values() for p in m.parameters())
    finally:
        dump(OUT/'forward_records.json',records,True)
        dump(OUT/'actual_profile_settings.json',actual_profiles,True)
        with (OUT/'predictions.npz').open('xb') as f:np.savez(f,**arrays)
        with (OUT/'waveforms.npz').open('xb') as f:np.savez(f,**waves)
    assert len(records)==240
    def diff(a,b):
        delta=np.abs(a-b);den=float(np.linalg.norm(b.astype(np.complex128).ravel()))
        return {'exact':bool(np.array_equal(a,b)),'within_original_tolerance':bool(np.allclose(a,b,rtol=plan['rtol'],atol=plan['atol'])),
            'max_abs':float(delta.max()),'relative_RMS':float(np.linalg.norm((a-b).astype(np.complex128).ravel())/max(den,1e-30)),
            'failing_elements':int((delta>plan['atol']+plan['rtol']*np.abs(b)).sum()),'elements':int(delta.size)}
    comparisons=[]
    for profile in profiles:
        for index in plan['indices']:
            def key(m,g,r=0):return f"{profile['id']}__{index}__{m}__grad{int(g)}__repeat{r}"
            pairs=[('cross_model_grad',key('lora',True),key('full',True)),('cross_model_no_grad',key('lora',False),key('full',False)),
                   ('full_grad_flag',key('full',True),key('full',False)),('lora_grad_flag',key('lora',True),key('lora',False))]
            pairs += [(f'repeat_{m}_grad{int(g)}',key(m,g,0),key(m,g,1)) for m in ('full','lora') for g in (True,False)]
            for name,a,b in pairs:comparisons.append({'profile':profile['id'],'index':index,'comparison':name,'prediction':diff(arrays[a],arrays[b]),'waveform':diff(waves[a],waves[b])})
    replay=[]
    for mode in ('full','lora'):
        original=np.load(OLD/mode/'initial_forward_outputs.npz')
        for index in plan['indices']:
            key=f'cudnn_tf32_nondet__{index}__{mode}__grad1__repeat0'
            replay.append({'mode':mode,'index':index,**diff(arrays[key],original[f'record_{index}'])})
    dump(OUT/'comparisons.json',comparisons,True);dump(OUT/'old_G33_replay.json',replay,True)
    for n,h in hashes.items():assert sha(ROOT/n)==h,n
    summary=[]
    for profile in profiles:
        for name in sorted({r['comparison'] for r in comparisons}):
            rs=[r for r in comparisons if r['profile']==profile['id'] and r['comparison']==name]
            summary.append({'profile':profile['id'],'comparison':name,'records':6,
                'prediction_tolerance_pass':sum(r['prediction']['within_original_tolerance'] for r in rs),
                'waveform_tolerance_pass':sum(r['waveform']['within_original_tolerance'] for r in rs),
                'prediction_exact':sum(r['prediction']['exact'] for r in rs),'prediction_max_abs':max(r['prediction']['max_abs'] for r in rs),
                'waveform_max_abs':max(r['waveform']['max_abs'] for r in rs)})
    dump(OUT/'summary.json',summary,True)
    marker={'status':'ok','utc':now(),'forwards':240,'comparisons':len(comparisons),'optimizer_updates':0,'parameters_unchanged':True,
        'elapsed_seconds_after_verification':time.monotonic()-started,'G33_original_acceptance_still_failed':True,
        'all_initial_parameter_digests_unchanged':initial_all,
        'hashes':{p.name:sha(p) for p in [OUT/'forward_records.json',OUT/'predictions.npz',OUT/'waveforms.npz',OUT/'comparisons.json',OUT/'old_G33_replay.json',OUT/'summary.json',OUT/'actual_profile_settings.json']}}
    dump(OUT/'NUMERICAL_COMPLETE.json',marker,True)
    report=ROOT/'docs/G33B_自动求导与后端数值定位结果.md'
    lines=['# G33B 自动求导与后端数值定位','','240次固定训练输入前向及波形已完整保存，零参数更新、无开发/测试访问。各行6条记录，沿用G33原容差；通过计数只是该路径对照的结果，不表示旧G33改判。', '',
        '|后端配置|对照|预测容差通过/6|波形容差通过/6|预测逐位相同/6|预测最大差|波形最大差|','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['profile']}|{r['comparison']}|{r['prediction_tolerance_pass']}|{r['waveform_tolerance_pass']}|{r['prediction_exact']}|{r['prediction_max_abs']:.6g}|{r['waveform_max_abs']:.6g}|")
    lines+=['','配置名称tf32表示许可，不证明每个RNN算子实际使用TF32；det表示cuDNN开关，不代表全局确定性。无cuDNN控制只说明另一执行路径，不是精度真值。旧默认相似配置的逐例复现另存old_G33_replay.json。没有放宽原容差，也没有用语音/ASR指标挑配置。后端选项只作用于本进程，未改系统、驱动或CUDA环境变量。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('numerical_results',report,[OUT/'NUMERICAL_COMPLETE.json',OUT/'summary.json',OUT/'comparisons.json',OUT/'old_G33_replay.json',OUT/'actual_initial_backend_settings.json',ROOT/'scripts/g33b_precision_audit.py'])
    status(stage='complete_pending_readback');print([r for r in summary if r['comparison']=='cross_model_grad'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():dump(OUT/'failure.json',{'utc':now(),'error':repr(e)},True)
        traceback.print_exc();raise
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('G33B precision audit ended\n')
