"""G35-P1 stage A only: geometry controls and <=14 existing-input forwards."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/g35_admission_v1'
REPORT=ROOT/'docs/G35A_几何与模型准入报告.md'
START=time.monotonic()
R={'stage':'A_only','status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
   'checks':[],'models':[],'forwards':[],'forward_attempts':0,'maximum_forwards':14,
   'training':False,'scan_inputs_generated':0,'ASR_decodes':0}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def check(name,ok,**details):
    row={'name':name,'passed':bool(ok),**details};R['checks'].append(row)
    save('progress.json',R)
    if not ok:raise RuntimeError('Admission failed: '+name)

def guard():
    if time.monotonic()-START>780:raise TimeoutError('900s stage budget;120s reserved for closure')

def main():
    import numpy as np
    import soundfile as sf
    import torch
    import importlib.metadata as md
    from mcse.geometry import rotate_azimuth,azimuth_degrees,wrap_degrees
    from mcse.spatial_noise import direction
    from mcse.simulation import geometric_delays,render_plane_wave,speech_activity_mask,snr_db
    from mcse.runtime import registered_systems,load_system,read_input,enhance_array
    R['versions']={p:md.version(p) for p in ['torch','torchaudio','numpy','scipy','soundfile','pesq','pystoi']}
    check('frozen_versions',all(R['versions'][k]==v for k,v in {
        'torch':'2.5.1+cu121','torchaudio':'2.5.1+cu121','numpy':'1.26.4',
        'scipy':'1.11.4','soundfile':'0.12.1','pesq':'0.0.4','pystoi':'0.4.1'}.items()))
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled=True;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    check('global_deterministic_disabled',not torch.are_deterministic_algorithms_enabled())
    check('CUDA_available_no_fallback',torch.cuda.is_available())
    R['backend']={'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'cudnn_version':torch.backends.cudnn.version(),'threads':torch.get_num_threads(),
        'cudnn_enabled':torch.backends.cudnn.enabled,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'benchmark':torch.backends.cudnn.benchmark,
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'global_deterministic':torch.are_deterministic_algorithms_enabled(),
        'GPU':torch.cuda.get_device_name(),'GPU_total_bytes':torch.cuda.get_device_properties(0).total_memory}
    R['gpu_before']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    protocol=ROOT/'docs/G35_前方180度方位扫描冻结协议_待审核.md'
    check('protocol_identity',sha(protocol)=='be1ef50a2018850029092b626d30bbd234dea151ae28ef0469dadf68cb5480d6')
    protected=[protocol,ROOT/'AGENTS.md',ROOT/'docs/毕设最终研究主线与精简实施方案.md']
    protected+=list((ROOT/'src/mcse').glob('*.py'))
    protected+=[ROOT/'scripts/mcnet_infer.py',ROOT/'configs/g1_geometry.json',ROOT/'data/g17_v1/manifest.jsonl']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected};save('protected_before.json',before)
    geometry=json.loads((ROOT/'configs/g1_geometry.json').read_text());pos=np.array(geometry['microphone_positions_m'])
    check('geometry_hash',sha(ROOT/'configs/g1_geometry.json')=='867d89f4f928f01a22de7db116539a09f0ad9bdb8b07e0439cb4053cb4b993dd')
    rows=[json.loads(s) for s in (ROOT/'data/g17_v1/manifest.jsonl').read_text().splitlines() if s.strip()]
    check('G17_manifest_identity',sha(ROOT/'data/g17_v1/manifest.jsonl')=='5f4924318aa2c45619610bc0177f591c051735dc9071b3d3bfc53be215173ce0')
    angles=list(range(-90,91,10));base=[r for r in rows if r['target_azimuth']==20]
    check('coordinate_positive90',np.max(np.abs(rotate_azimuth([0,1,0],90)-[1,0,0]))<=1e-12)
    maxnorm=maxdot=maxaz=maxdelay=maxanalytic=maxinverse=maxref=0.;required_guard=0;direction_checks=0
    # Synthetic unit impulse is a geometry-only FIR fixture, never a model input.
    impulse=np.zeros(512);impulse[256]=1.;reference_fixture=None
    for theta in angles:
        analytic_target=np.array([np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(10)),np.cos(np.deg2rad(theta))*np.cos(np.deg2rad(10)),np.sin(np.deg2rad(10))])
        fixture=render_plane_wave(impulse,geometric_delays(pos,analytic_target))
        if reference_fixture is None:reference_fixture=fixture[:,4].copy()
        maxref=max(maxref,float(np.max(np.abs(fixture[:,4]-reference_fixture))))
        for r in base:
            old=[direction(20,10)]+[np.array(u) for u in r['base_noise_directions']]
            new=[rotate_azimuth(u,theta-20) for u in old]
            maxanalytic=max(maxanalytic,float(np.max(np.abs(new[0]-analytic_target))))
            if r['condition']=='point':
                rel=r['relative_angle'];a=np.deg2rad(theta+rel);e=np.deg2rad(-15)
                expected=np.array([np.sin(a)*np.cos(e),np.cos(a)*np.cos(e),np.sin(e)])
                maxanalytic=max(maxanalytic,float(np.max(np.abs(new[1]-expected))))
                maxaz=max(maxaz,abs(wrap_degrees(azimuth_degrees(new[1])-theta-rel)))
            maxdot=max(maxdot,float(np.max(np.abs(np.array(old)@np.array(old).T-np.array(new)@np.array(new).T))))
            for u,v in zip(old,new):
                maxnorm=max(maxnorm,abs(float(np.linalg.norm(v))-1.))
                maxinverse=max(maxinverse,float(np.max(np.abs(rotate_azimuth(v,20-theta)-u))))
                actual=geometric_delays(pos,v)
                independent=np.array([-sum((float(pos[m,j])-float(pos[4,j]))*float(v[j]) for j in range(3))*16000/343 for m in range(6)])
                maxdelay=max(maxdelay,float(np.max(np.abs(actual-independent))))
                check_ref=actual[4];assert check_ref==0
                required_guard=max(required_guard,32+int(np.ceil(np.abs(actual).max()))+1)
                direction_checks+=1
    right=geometric_delays(pos,[1,0,0])
    G={'angles':angles,'groups':len(base),'source_direction_checks':direction_checks,'norm_max_error':maxnorm,
       'dot_max_error':maxdot,'relative_azimuth_max_error_deg':maxaz,'analytic_direction_max_error':maxanalytic,
       'inverse_max_error':maxinverse,'TDOA_max_error_samples':maxdelay,'required_guard_samples':required_guard,
       'reference_impulse_cross_angle_max_error':maxref,'synthetic_fixture_only':True,'scan_audio_generated':False}
    save('geometry.json',G)
    check('19_angles_56_groups',len(angles)==19 and len(base)==56)
    check('unit_dot_analytic_inverse_1e12',max(maxnorm,maxdot,maxanalytic,maxinverse)<=1e-12,**G)
    check('relative_azimuth_1e10_deg',maxaz<=1e-10)
    check('independent_TDOA_1e10_samples',maxdelay<=1e-10)
    check('right_source_arrives_earlier_at_right_mic',right[2]<right[0] and right[5]<right[3])
    check('FIR_guard_64',required_guard<=64)
    check('natural_reference_FIR_consistency',maxref==0.)
    inputs=[]
    for sid in ['g17_0001_6930-75918-0000','g17_0019_6930-75918-0000']:
        r=next(r for r in rows if r['sample_id']==sid)
        check('input_and_reference_hash_'+sid,sha(r['noisy_path'])==r['input_sha256'] and sha(r['reference_path'])==r['reference_sha256'])
        x=read_input(r['noisy_path']);s,sr=sf.read(r['reference_path'],dtype='float32')
        check('input_reference_shape_finite_'+sid,x.shape==(56208,6) and s.shape==(56208,) and sr==16000 and np.isfinite(s).all())
        check('input_peak_'+sid,float(np.abs(x).max())<=.95+2**-23)
        if r['condition']=='clean':check('clean_reference_exact',np.array_equal(x[:,4],s))
        else:check('saved_SNR_0p002dB',abs(snr_db(s,x[:,4]-s,speech_activity_mask(s)))<=.002)
        inputs.append((r,x,s))
    check('two_inputs_same_target_reference',np.array_equal(inputs[0][2],inputs[1][2]))
    R['inputs']=[{k:r[k] for k in ['sample_id','noisy_path','reference_path','input_sha256','reference_sha256','common_scale','noise_gain']} for r,_,_ in inputs]
    systems=['mcnet']+[f'g5/fixed_without_clean_identity_s{s}' for s in (20261112,20261113,20261114)]+[f'g15/rigid_without_clean_identity_s{s}' for s in (20261112,20261113,20261114)]
    registry=registered_systems();save('selected_registry.json',{k:registry[k] for k in systems})
    check('upstream_commit_clean',subprocess.check_output(['git','-C',str(ROOT/'third_party/McNet'),'rev-parse','HEAD'],text=True).strip()=='9c194032659a3047a4b4c244a9f541f4d34b1682' and not subprocess.check_output(['git','-C',str(ROOT/'third_party/McNet'),'status','--porcelain'],text=True).strip())
    def digest(model,base_only=False):
        h=hashlib.sha256()
        for name,p in sorted(model.named_parameters()):
            if base_only and 'lora_' in name:continue
            h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    official_digest=None
    for system in systems:
        guard();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
        runtime=load_system(system,'cuda');model=runtime.model
        shapes={n:{'shape':list(p.shape),'dtype':str(p.dtype),'finite':bool(torch.isfinite(p).all()),'requires_grad':p.requires_grad} for n,p in model.named_parameters()}
        check('FP32_finite_frozen_'+system,all(v['dtype']=='torch.float32' and v['finite'] and not v['requires_grad'] for v in shapes.values()))
        base_digest=digest(model,True);initial=digest(model)
        if official_digest is None:official_digest=base_digest
        check('original_base_parameters_identical_'+system,base_digest==official_digest)
        lora={n:v for n,v in shapes.items() if 'lora_' in n}
        expected={'freq.fc_output_layer.lora_A':[4,256],'freq.fc_output_layer.lora_B':[64,4],
                  'narr.fc_output_layer.lora_A':[4,512],'narr.fc_output_layer.lora_B':[64,4]}
        check('parameter_shapes_'+system,({n:v['shape'] for n,v in lora.items()}==expected if system!='mcnet' else not lora))
        save('shapes_'+system.replace('/','_')+'.json',shapes)
        counts=[0]
        def count_forward(module,args):
            counts[0]+=1
        hook=model.register_forward_pre_hook(count_forward)
        for r,x,s in inputs:
            guard();assert R['forward_attempts']<14
            check('same_model_input_hash_'+system+'_'+r['sample_id'],sha(r['noisy_path'])==r['input_sha256'])
            R['forward_attempts']+=1;save('progress.json',R)
            torch.cuda.synchronize();t=time.perf_counter();y,_=enhance_array(runtime,x);torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            check('output_interface_'+system+'_'+r['sample_id'],y.dtype==np.float32 and y.shape==(len(x),) and np.isfinite(y).all())
            path=OUT/f"{system.replace('/','_')}__{r['sample_id']}.wav"
            with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
            reread,rate=sf.read(path,dtype='float32');check('FLOAT_roundtrip_'+path.stem,rate==16000 and np.array_equal(y,reread))
            R['forwards'].append({'system':system,'sample_id':r['sample_id'],'seconds':elapsed,'samples':len(y),
                'output_path':str(path.relative_to(ROOT)),'output_sha256':sha(path),'output_abs_peak':float(np.abs(y).max()),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            save('progress.json',R)
        hook.remove();check('exactly_two_root_forwards_'+system,counts[0]==2)
        check('in_memory_parameters_unchanged_'+system,digest(model)==initial)
        R['models'].append({'system':system,'metadata':runtime.metadata,'root_forward_calls':counts[0],
            'base_parameter_digest':base_digest,'initial_and_final_parameter_digest':initial,'parameter_count':sum(p.numel() for p in model.parameters()),
            'adapter_parameter_count':sum(p.numel() for n,p in model.named_parameters() if 'lora_' in n),
            'seconds_including_load_checks':time.monotonic()-began,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        save('progress.json',R);del runtime,model;gc.collect();torch.cuda.empty_cache()
    check('14_forwards_all_7_models',len(R['forwards'])==14 and len(R['models'])==7 and R['forward_attempts']==14)
    after={str(p.relative_to(ROOT)):sha(p) for p in protected}
    check('historical_protected_files_unchanged',before==after)
    for system in systems[1:]:
        row=registry[system];check('final_weight_hash_'+system,sha(ROOT/row['path'])==row['checkpoint_sha256'])
    R['status']='passed'

if __name__=='__main__':
    if OUT.exists() or REPORT.exists():raise SystemExit('Existing admission artifacts/report; refusing rerun or overwrite')
    OUT.mkdir()
    save('run_identity.json',{'pid':os.getpid(),'script_sha256':sha(__file__),'hard_limit_seconds':900,'maximum_forwards':14,'input_cases':2})
    try:main()
    except BaseException as e:
        R['status']='failed';R['error']=repr(e);R['traceback']=traceback.format_exc()
    finally:
        R['elapsed_seconds']=time.monotonic()-START;R['finished_utc']=datetime.now(timezone.utc).isoformat()
        R['not_executed']=['stage_B_1064_inputs','stage_C_7448_outputs','19_angle_performance_curves','ASR','training','G34_resume','git_commit','version_snapshot']
        save('result.json',R)
        lines=['# G35A 几何与模型准入报告','',f"状态：**{R['status']}**；总墙时 {R['elapsed_seconds']:.3f} 秒。",
            f"本次实际尝试 {R['forward_attempts']} 次、完成 {len(R['forwards'])} 次神经前向；最多2条已有G17输入，无训练、无19角批量扫描。",
            '', '## 实际检查','', '|项目|结果|','|---|---|']
        lines += [f"|{c['name']}|{'通过' if c['passed'] else '失败'}|" for c in R['checks']]
        lines += ['', '## 执行环境与来源','', '```json',json.dumps(R.get('backend',{}),ensure_ascii=False,indent=2),'```',
            '', '逐模型形状、原始参数摘要、注册来源/权重SHA、两输入SHA、每次耗时和显存详见同目录JSON。',
            '[完整机器结果](../artifacts/g35_admission_v1/result.json)、[几何原始检查](../artifacts/g35_admission_v1/geometry.json)、[冻结协议](G35_前方180度方位扫描冻结协议_待审核.md)。',
            '', '## 未执行项与曲线优先级','',
            '本报告仅A准入。19角SI-SDR/PESQ/STOI完整曲线需要B构建1064输入与C运行7448增强输出，超过本次最多14前向的明确上限；不能以两条接口检查代替性能扫描。后续应优先批准B、C及三项曲线汇总，ASR可按后续批准阶段处理。',
            '', '## 安全保存版本建议（未执行）','',
            '主项目尚无首次Git提交。建议先在另一个日期命名目录作只追加备份，明确复制范围/空间，保留根Git、官方子仓库来源、源码/配置/文档及数据/权重/日志；分别记录SHA清单。再经用户批准，以明确文件清单建立首次源码提交；被忽略产物另存对应快照。不运行git add/commit/push，不覆盖或移动原件。',
            '', '检查后停止，不能按旧队列状态继续G34或自动执行后续阶段。']
        if 'error' in R:lines+=['','## 失败原文','','```',R['traceback'],'```','失败后未更改容差或自动重跑。']
        with REPORT.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
        print(json.dumps({'status':R['status'],'forwards':len(R['forwards']),'attempts':R['forward_attempts'],'seconds':R['elapsed_seconds']},ensure_ascii=False),flush=True)
    raise SystemExit(0 if R['status']=='passed' else 1)
