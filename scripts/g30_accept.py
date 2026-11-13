#!/usr/bin/env python3
"""Audit calibrated coefficients, every optimizer state, and fixed registration."""
import numpy as np
import torch
from g30_common import ROOT,OUT,sha,dump,now,meta,guard,archive,status


def main():
    plan=guard();done=meta(OUT/'TRAINING_COMPLETE.json');cal=meta(OUT/'CALIBRATION_COMPLETE.json')
    assert done['status']=='ok' and done['updates']==4500 and not done['development_evaluated']
    assert meta(OUT/'power_helper.json')['status']=='released'
    assert sha(OUT/'sample_indices.npy')==cal['schedule_sha256'] and sha(OUT/'calibration_records.json')==cal['records_sha256']
    assert sha(OUT/'calibration_gradients.npz')==cal['vectors_sha256']
    rows=meta(OUT/'calibration_records.json');vectors=np.load(OUT/'calibration_gradients.npz')
    assert [r['index'] for r in rows]==plan['calibration_indices']
    for r in rows:
        for objective in ('cirm','mask','shape'):
            v=vectors[f"record_{r['index']}_{objective}"];assert v.shape==(3584,) and np.isfinite(v).all()
            assert np.isclose(np.linalg.norm(v),r['objectives'][objective]['gradient_norm'],rtol=1e-12)
    for aux in ('mask','shape'):
        coefficient=.1*float(np.median([np.linalg.norm(vectors[f"record_{r['index']}_cirm"])/np.linalg.norm(vectors[f"record_{r['index']}_{aux}"]) for r in rows]))
        assert np.isclose(coefficient,cal['coefficients'][aux],rtol=1e-12)
    stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    results=[];index_hashes=set();checkpoint_registry={};rng_reference={}
    for variant in plan['variants']:
        r=meta(OUT/'train'/variant/'result.json');assert sha(OUT/'train'/variant/'result.json')==done['result_hashes'][variant]
        assert r['status']=='ok' and r['steps']==r['optimizer_updates']==1500 and not r['development_or_test_access']
        assert r['initial_digests']==cal['initial_parameter_digests'] and r['final_base_digest']==r['initial_digests']['base']
        assert r['coefficients']==cal['coefficients'] and r['schedule_sha256']==cal['schedule_sha256']
        assert r['registered']==r['checkpoints'][-1] and r['registered']['step']==1500
        assert [c['step'] for c in r['checkpoints']]==[500,1000,1500]
        assert sha(OUT/'train'/variant/'steps.jsonl')==r['training_log_sha256']
        import json
        logs=[json.loads(line) for line in (OUT/'train'/variant/'steps.jsonl').read_text().splitlines()]
        assert [l['step'] for l in logs]==list(range(50,1501,50))
        index_hashes.add(r['actual_indices_sha256'])
        for c in r['checkpoints']:
            path=ROOT/c['path'];assert sha(path)==c['sha256']
            cp=torch.load(path,map_location='cpu',weights_only=False)
            assert cp['step']==c['step'] and cp['variant']==variant and cp['seed']==plan['seed']
            coefficient=0. if variant=='cirm' else cal['coefficients']['mask' if variant=='cirm_mask' else 'shape']
            assert cp['coefficient']==coefficient
            assert cp['base_checkpoint_sha256']=='24b8c556cc52990e4d7db83da4909b7e8a1dc4a0c14880b98381bb62818beecf'
            assert cp['schedule_sha256']==cal['schedule_sha256'] and cp['plan_sha256']==sha(OUT/'frozen_plan.json')
            assert len(cp['adapter'])==4 and sum(t.numel() for t in cp['adapter'].values())==3584 and all(torch.isfinite(t).all() for t in cp['adapter'].values())
            assert len(cp['optimizer']['state'])==4
            assert all(int(state['step'])==c['step'] for state in cp['optimizer']['state'].values())
            assert all(torch.isfinite(v).all() for state in cp['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v))
            rng=(cp['torch_rng_state'],cp['cuda_rng_state'])
            if c['step'] not in rng_reference:rng_reference[c['step']]=rng
            else:assert all(torch.equal(a,b) for a,b in zip(rng_reference[c['step']],rng))
        checkpoint_registry[variant]=r['registered'];results.append(r)
    assert len(index_hashes)==1
    indices=np.load(OUT/'sample_indices.npy');import hashlib
    assert next(iter(index_hashes))==hashlib.sha256(indices.tobytes()).hexdigest()
    guard();dump(OUT/'registered_checkpoints.json',checkpoint_registry,True)
    audit={'status':'ok','utc':now(),'training_runs':3,'optimizer_updates':4500,'seed':plan['seed'],
        'registration_step':1500,'same_initial_parameters':True,'same_sample_indices':True,'base_unchanged':True,
        'coefficients_recomputed':True,'all_checkpoint_optimizer_step_counts_verified':True,
        'matched_RNG_at_checkpoints':True,'development_or_test_scored':False,'stages':stages,
        'registry_sha256':sha(OUT/'registered_checkpoints.json'),'training_complete_sha256':sha(OUT/'TRAINING_COMPLETE.json')}
    dump(OUT/'FINAL_TRAINING_ACCEPTANCE.json',audit,True)
    report=ROOT/'docs/G30_同预算混响适配训练验收.md'
    lines=['# G30 同预算混响适配训练验收','','三个目标组全部完成固定1500次更新，共4500次。六条训练输入的辅助系数由保存梯度独立重算一致。三组初始主干/LoRA、抽样序列及各checkpoint随机状态一致；主干摘要未变；所有500/1000/1500步checkpoint的参数有限、Adam步数准确。统一登记1500步，无开发或测试评分选择。', '',
        '|组|步数|Torch峰值已分配GiB|Torch峰值保留GiB|耗时秒|','|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['variant']}|1500|{r['peak_allocated_bytes']/2**30:.3f}|{r['peak_reserved_bytes']/2**30:.3f}|{r['elapsed_seconds']:.1f}|")
    lines+=['',f"lambda_mask={cal['coefficients']['mask']:.8g}，lambda_shape={cal['coefficients']['shape']:.8g}。均事前按训练输入固定公式确定，不声称最优或始终弱辅助。", '',
        '这里只完成单种子开发训练，不证明改进有效、创新或泛化。下一阶段先冻结开发评价：所有已登记组及未适配McNet，直接输出/MVDR/MWF、相同SI-SDR/PESQ/STOI与干净电平/NMSE。中心短裁剪不按整句转写做WER。更大训练预算、三种子及新完整语句确认须根据开发证据另行决定并保留强对照。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('training_acceptance',report,[OUT/'FINAL_TRAINING_ACCEPTANCE.json',OUT/'registered_checkpoints.json',ROOT/'scripts/g30_accept.py',OUT/'power_helper.json'])
    status(stage='training_accepted_development_not_scored')


if __name__=='__main__':main()
