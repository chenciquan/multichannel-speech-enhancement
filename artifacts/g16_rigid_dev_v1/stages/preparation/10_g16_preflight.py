#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import ast,difflib,gc,json,subprocess,sys
import numpy as np
import soundfile as sf
import torch
from pystoi import stoi
from g16_common import ROOT,OUT,now,sha,dump,read,archive,status
from mcse.evaluation import si_sdr
from mcse.runtime import registered_systems,load_system,enhance_array


def main():
    status(stage='runtime_preflight');torch.set_num_threads(2);prior=ROOT/'artifacts/g16_runtime_revision_v1/runtime_before.py';g14=ROOT/'artifacts/g14_corrected_geometry_v1'
    assert sha(prior)==json.loads((g14/'PIPELINE_READY.json').read_text())['files']['src/mcse/runtime.py']
    old=prior.read_text();new=(ROOT/'src/mcse/runtime.py').read_text();assert new!=old
    before,after=ast.parse(old),ast.parse(new);function=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='registered_systems')
    entries=next(n.value for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='entries' for t in n.targets))
    assert len(entries.elts)==3 and entries.elts[-1].elts[0].value=='g15';entries.elts.pop();assert ast.dump(before)==ast.dump(after)
    (OUT/'runtime_registry_only.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='G14frozen/runtime.py',tofile='G16/runtime.py')),encoding='utf-8')
    systems=registered_systems();reg=json.loads((ROOT/'artifacts/g15_rigid_training_v1/registered_checkpoints.json').read_text());assert all('g15/'+r['system'] in systems for r in reg['systems'])
    cases=read(ROOT/'data/g15_v1/validation/manifest.jsonl');case=next(r for r in cases if 512<=sf.info(r['noisy_path']).frames<=64000);x,sr=sf.read(case['noisy_path'],dtype='float32',always_2d=True);s,rr=sf.read(case['reference_path'],dtype='float32');assert sr==rr==16000
    probes=[]
    for row in reg['systems']:
        runtime=load_system('g15/'+row['system'],'cuda');y,_=enhance_array(runtime,x);saved=json.loads((ROOT/'artifacts/g15_rigid_training_v1/train'/f'without_clean_identity_s{row["seed"]}'/'validation_03000.json').read_text());expected=next(r for r in saved['cases'] if r['sample_id']==case['sample_id'])
        si=si_sdr(s,y);st=float(stoi(s,y,16000,extended=False));assert abs(si-expected['si_sdr_db'])<=1e-5 and abs(st-expected['stoi'])<=1e-7
        probes.append({'system':row['system'],'case':case['sample_id'],'SI_difference':si-expected['si_sdr_db'],'STOI_difference':st-expected['stoi'],'metadata':runtime.metadata});del runtime;gc.collect();torch.cuda.empty_cache()
    case=next(r for r in read(ROOT/'data/g14_v1/manifest.jsonl') if r['geometry_protocol']=='rigid_rotation');x,sr=sf.read(case['noisy_path'],dtype='float32',always_2d=True);oldrows=read(g14/'evaluation/combined_signals.jsonl');oldprobes=[]
    for system in ('mcnet','g5/rotated_without_clean_identity_s20261112'):
        runtime=load_system(system,'cuda');r=next(r for r in oldrows if r['system']==system.split('/')[-1] and r['sample_id']==case['sample_id']);assert sha(r['output_path'])==r['output_sha256'];y,_=enhance_array(runtime,x);expected,yr=sf.read(r['output_path'],dtype='float32');error=float(np.max(np.abs(y-expected)));assert error<=1e-5
        oldprobes.append({'system':system,'max_abs':error});del runtime;gc.collect();torch.cuda.empty_cache()
    log=OUT/'full98_runtime_tests.log'
    with log.open('x') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},stdout=f,stderr=subprocess.STDOUT,timeout=300,check=True)
    assert 'Ran 98 tests' in log.read_text() and '\nOK' in log.read_text()
    dump(OUT/'RUNTIME_QA.json',{'status':'ok','completed_utc':now(),'previous_runtime_sha256':sha(prior),'current_runtime_sha256':sha(ROOT/'src/mcse/runtime.py'),'new_model_probes':probes,'old_model_probes':oldprobes,'full_tests':98},True)
    report=ROOT/'docs/G16A_正确训练权重注册与推理兼容验证.md';report.write_text('# G16A 正确训练权重注册与推理兼容验证\n\nG15验收后，只给共享推理接口增加归档G15注册表入口。修改前runtime完整保存，SHA与G14冻结源码一致，原模型及解码代码不变。新三种子仅允许从训练归档核验的末步3000检查点加载。\n\n按预定“首条完整且≤4秒验证音频”探测三新模型，重现已保存G15末步逐例SI-SDR/STOI，容差1e-5dB/1e-7；两旧模型G14波形重现差≤1e-5。实际差见RUNTIME_QA.json。98项完整测试通过。该探测验证加载/接口，不选模型或证明性能提升。\n\n下一阶段复用G14已查看正确几何168条件和15基线，补充三新直接/对应MVDR系统。只新增1008输出，总3528记录/后端；不下载、训练、改权重。开发诊断完成后再决定和冻结新语音确认协议。\n\nUTC：'+now()+'\n',encoding='utf-8')
    archive('runtime',report,[OUT/'RUNTIME_QA.json',OUT/'runtime_registry_only.diff',log,prior,ROOT/'src/mcse/runtime.py',ROOT/'scripts/g16_preflight.py',OUT/'frozen_plan.json'])


if __name__=='__main__':main()
