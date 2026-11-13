"""Re-read initial arrays, all6000 log rows and all12 checkpoint optimizer states."""
import hashlib
import json
import numpy as np
import torch
from g34_common import ROOT,OUT,sha,meta,dump,now,guard,archive,status,state_digest


def main():
    plan=guard();done=meta(OUT/'TRAINING_COMPLETE.json');initial=meta(OUT/'INITIAL_COMPLETE.json')
    assert done['status']=='ok' and done['runs']==4 and done['updates']==6000 and done['no_development_or_test_audio']
    assert meta(OUT/'power_helper.json')['status']=='released'
    assert not (ROOT/'artifacts/g33_full_gradient_probe_v1/FINAL_ACCEPTANCE_AUDIT.json').exists()
    for n,h in initial['hashes'].items():assert sha(OUT/n)==h,n
    a=np.load(OUT/'initial_predictions.npz');w=np.load(OUT/'initial_waveforms.npz');records=meta(OUT/'initial_records.json')
    assert [r['index'] for r in records]==plan['initial_indices']
    for r in records:
        for arrays,key in ((a,'prediction'),(w,'waveform')):
            left,right=arrays[f"lora_{r['index']}"],arrays[f"full_{r['index']}"]
            delta=np.abs(left-right);passed=bool(np.allclose(left,right,rtol=plan['initial_wave_rtol'],atol=plan['initial_wave_atol']))
            assert passed==r[key]['within_tolerance'] and float(delta.max())==r[key]['max_abs']
            assert int((delta>plan['initial_wave_atol']+plan['initial_wave_rtol']*np.abs(right)).sum())==r[key]['failing_elements']
        assert r['waveform']['within_tolerance']
    assert sum(r['prediction']['within_tolerance'] for r in records)==initial['cirm_passes']
    indices=np.load(OUT/'sample_indices.npy');expected_indices=hashlib.sha256(indices.tobytes()).hexdigest()
    registry={};rows=[];reference_rng={}
    for run in plan['runs']:
        run_id=run['id'];folder=OUT/'train'/run_id;r=meta(folder/'result.json');cfg=meta(folder/'config.json')
        assert sha(folder/'result.json')==done['result_hashes'][run_id]
        assert r['status']=='ok' and r['steps']==r['optimizer_updates']==1500 and r['no_development_or_test_audio']
        assert r['run']==cfg['run']==run and r['initial']==cfg['initial']==initial['initial'][run['mode']]
        assert cfg['backend']==r['backend']==initial['backend']
        assert r['schedule_sha256']==cfg['schedule_sha256']==plan['schedule_sha256'] and r['actual_indices_sha256']==expected_indices
        assert sha(folder/'steps.jsonl')==r['training_log_sha256']
        logs=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        assert len(logs)==1500 and [q['step'] for q in logs]==list(range(1,1501)) and [q['index'] for q in logs]==indices.tolist()
        assert all(q['all_trainable_gradients_connected_finite'] and q['all_updated_parameters_finite'] and
            np.isfinite([q['loss'],q['gradient_norm_before_clip'],q['synchronized_step_seconds']]).all() and q['synchronized_step_seconds']>0 for q in logs)
        assert abs(sum(q['synchronized_step_seconds'] for q in logs)-r['synchronized_step_seconds_sum'])<1e-9
        assert float(np.median([q['synchronized_step_seconds'] for q in logs[1:]]))==r['warm_step_median_seconds']
        assert [c['step'] for c in r['checkpoints']]==[500,1000,1500] and r['registered']==r['checkpoints'][-1]
        if run['mode']=='lora':assert r['final_base_digest']==r['initial']['base']
        else:assert r['final_base_digest']!=r['initial']['base']
        for c in r['checkpoints']:
            path=ROOT/c['path'];assert sha(path)==c['sha256'];cp=torch.load(path,map_location='cpu',weights_only=False)
            assert cp['run']==run and cp['step']==c['step'] and cp['seed']==plan['seed'] and cp['backend']==initial['backend']
            assert cp['base_checkpoint_sha256']==initial['checkpoint_sha256']=='24b8c556cc52990e4d7db83da4909b7e8a1dc4a0c14880b98381bb62818beecf'
            assert cp['schedule_sha256']==plan['schedule_sha256'] and cp['plan_sha256']==sha(OUT/'frozen_plan.json')
            state=cp['weights'];expected=initial['state_schema'][run['mode']]
            expected={n:s for n,s in expected.items() if run['mode']=='full' or 'lora_' in n}
            assert set(state)==set(expected) and all(list(t.shape)==expected[n] and t.dtype==torch.float32 and torch.isfinite(t).all() for n,t in state.items())
            assert sum(t.numel() for t in state.values())==run['parameters']
            optimizer=cp['optimizer'];assert len(optimizer['param_groups'])==1
            pg=optimizer['param_groups'][0];assert pg['lr']==run['lr'] and tuple(pg['betas'])==tuple(plan['betas']) and pg['eps']==plan['eps'] and pg['weight_decay']==0
            assert len(optimizer['state'])==len(cfg['trainable_names'])==len(pg['params'])
            for name,pid in zip(cfg['trainable_names'],pg['params']):
                s=optimizer['state'][pid];assert int(s['step'])==c['step']
                assert list(s['exp_avg'].shape)==expected[name] and list(s['exp_avg_sq'].shape)==expected[name]
                assert all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v))
                assert (s['exp_avg_sq']>=0).all()
            rng=(cp['torch_rng_state'],cp['cuda_rng_state'])
            if c['step'] not in reference_rng:reference_rng[c['step']]=rng
            else:assert all(torch.equal(v,z) for v,z in zip(reference_rng[c['step']],rng))
            if c['step']==1500:assert state_digest(state)==r['final_saved_weight_digest']
        registry[run_id]={'run':run,**r['registered']};rows.append(r)
    stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    guard();dump(OUT/'registered_checkpoints.json',registry,True)
    audit={'utc':now(),'status':'ok','runs':4,'optimizer_updates':6000,'independently_read_log_rows':6000,
        'checkpoint_optimizer_states_verified':12,'six_initial_waveforms_verified':True,'cirm_residual_preserved':True,
        'old_G33_acceptance_still_failed':True,'same_initial_base':True,'same_schedule':True,'same_checkpoint_rng_states':True,
        'no_development_or_test_audio':True,'stages':stages,'registry_sha256':sha(OUT/'registered_checkpoints.json'),
        'training_complete_sha256':sha(OUT/'TRAINING_COMPLETE.json')}
    dump(OUT/'FINAL_TRAINING_ACCEPTANCE.json',audit,True)
    report=ROOT/'docs/G34_统一精度全参数与低秩训练验收.md'
    lines=['# G34 统一精度全参数与低秩训练验收','','四组均完成固定1500次原cIRM更新，共6000次。逐条读取6000条日志及12份checkpoint：索引、有限性记录、权重形状/数值、Adam步数/矩状态、基础CP、统一精度开关及登记权重哈希一致。相同CP和抽样序列，构造后重置种子，各checkpoint的CPU/CUDA随机状态一致。LoRA主干未变；全参数按预期改变。未加载开发/测试音频，不按成绩选择。', '',
        '|组|开放参数|Torch已分配峰值GiB|Torch保留峰值GiB|同步逐步中位秒|脚本墙时秒|',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['run']['id']}|{r['run']['parameters']}|{r['peak_allocated_bytes']/2**30:.3f}|{r['peak_reserved_bytes']/2**30:.3f}|{r['warm_step_median_seconds']:.6f}|{r['elapsed_seconds']:.1f}|")
    lines+=['','显存为此1.48秒/batch1且实际分配Adam状态的当前实现观测，不推广到长语句/大batch；同步逐步时间包含有限性检查与Python/CUDA调用，非纯GPU内核时间。墙时含加载、校验、保存及WSL挂载盘I/O。','',
        f"初始六条训练波形通过固定容差6/6，压缩cIRM通过{initial['cirm_passes']}/6。旧G33未通过状态保留，不认定残差已完全解释或无害。",'',
        '这里只验收训练完整性，不证明增强效果、内容保真或创新。三个全参数学习率必须全部进入后续开发表，不代表充分调参的最好全参数模型。下一阶段先冻结108开发短段的统一精度信号评价，不对中心裁剪使用整句WER，不改变G31失败门槛。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('training_acceptance',report,[OUT/'FINAL_TRAINING_ACCEPTANCE.json',OUT/'registered_checkpoints.json',ROOT/'scripts/g34_accept.py',OUT/'power_helper.json'])
    status(stage='training_accepted_development_not_started')


if __name__=='__main__':main()
