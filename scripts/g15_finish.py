#!/usr/bin/env python3
import json,runpy
import torch
from g15_common import ROOT,OUT,BASE,now,sha,dump,status,archive


def finish():
    plan=json.loads((OUT/'frozen_plan.json').read_text());outcomes=[];systems=[];checks={};old=ROOT/'artifacts/g5_rotation_v1'
    for seed in plan['seeds']:
        folder=OUT/'train'/f'without_clean_identity_s{seed}';r=json.loads((folder/'result.json').read_text());cfg=json.loads((folder/'config.json').read_text());checkpoint=folder/'adapter_03000.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert r['status']=='ok' and r['completed_steps']==3000 and r['base_unchanged'] and r['base_digest_before']==r['base_digest_after'] and cfg['trainable_parameters']==3584
        assert saved['step']==3000 and saved['seed']==seed and sum(t.numel() for t in saved['adapter'].values())==3584
        paired={}
        for geometry in ('fixed','rotated'):
            priorfolder=old/geometry/'train'/f'without_clean_identity_s{seed}';prior=torch.load(priorfolder/'adapter_03000.pt',map_location='cpu',weights_only=False);result=json.loads((priorfolder/'result.json').read_text())
            same=saved['numpy_sampler_state']==prior['numpy_sampler_state'] and saved['identity_sampler_state']==prior['identity_sampler_state'];same_counts=r['sampler_counts']==result['sampler_counts'];assert same and same_counts
            paired[geometry]={'numpy_and_identity_sampler_states_equal':same,'sampler_counts_equal':same_counts,'CPU_torch_RNG_equal':torch.equal(saved['torch_rng_state'],prior['torch_rng_state']),
              'CUDA_torch_RNG_equal':torch.equal(saved['cuda_rng_state'],prior['cuda_rng_state']),'comparison_final_checkpoint_sha256':sha(priorfolder/'adapter_03000.pt')}
        val=json.loads((folder/'validation_03000.json').read_text());systems.append({'system':f'rigid_without_clean_identity_s{seed}','geometry':'rigid','kind':'adapter','variant':'without_clean_identity','seed':seed,'step':3000,
            'path':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint),'registration_policy':'prescribed_final_step3000_no_validation_selection'})
        outcomes.append({'seed':seed,'effective_updates':3000,'validation_final_SI':val['si_sdr_db'],'validation_final_STOI':val['stoi'],'diagnostic_engine_selected_step_not_used':r['selected']['step'],
            'elapsed_seconds':r['elapsed_seconds'],'peak_allocated_bytes':r['peak_allocated_bytes'],'peak_reserved_bytes':r['peak_reserved_bytes'],'base_unchanged':r['base_unchanged'],'sampler_checks':paired})
    checks['all_three_seeds9000_updates']=len(outcomes)==3 and sum(r['effective_updates'] for r in outcomes)==9000
    files=json.loads((OUT/'data_file_hashes.json').read_text());checks['all_prepared_data_hashes_still_match']=all(sha(ROOT/p)==h for p,h in files.items())
    ready=json.loads((OUT/'PIPELINE_READY.json').read_text());checks['all_frozen_code_unchanged']=all(sha(ROOT/p)==h for p,h in ready['files'].items())
    checks['all_sampler_counts_and_states_match_both_old_policies']=all(all(p['numpy_and_identity_sampler_states_equal'] and p['sampler_counts_equal'] for p in r['sampler_checks'].values()) for r in outcomes)
    checks['CPU_and_CUDA_RNG_match_both_old_policies']=all(all(p['CPU_torch_RNG_equal'] and p['CUDA_torch_RNG_equal'] for p in r['sampler_checks'].values()) for r in outcomes)
    b=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g15_base_check');repo=ROOT/'third_party/McNet';b['verify_upstream'](repo,repo/b['OFFLINE_CHECKPOINT']);checks['original_source_and_checkpoint_unchanged']=True
    result={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'runs':outcomes,'training_updates':9000,'completed_utc':now(),'new_test_results':False};dump(OUT/'training_audit.json',result,True)
    if result['status']!='ok':raise RuntimeError('Training audit failed; preserve artifacts,do not evaluate/register silently')
    dump(OUT/'registered_checkpoints.json',{'registered_utc':now(),'selection':'prescribed_step3000_for_all_seeds','systems':systems},True)
    lines=['# G15B 正确刚性旋转三种子训练与注册报告','','G15A数据及源/代码快照保存后，三种子各完成3000次FP32有效更新，共9000。复用原g3_train引擎，原模型源码及检查点未改，3584个LoRA参数可训练。各运行冻结基线参数摘要前后一致。',
      '', '三种子与各自旧fixed/rotated末步检查点的NumPy/identity采样状态、分类计数及CPU/CUDA随机状态全部匹配；保留了原片段、目标角序列、监督参考、缩放、优化器、批次与预算。仅训练输入的噪声方向改为同向刚性旋转。',
      '', '预定末步3000全部注册，不按本次验证SI/STOI择优。旧引擎仍输出历史selected诊断字段，已明确不用于注册；旧六个比较模型也核实为3000步。下表只是54条正确几何开发验证记录的均值，既不是新测试或人声完整性证明，也不表明新方法优于旧方法。','','|种子|有效更新|开发验证SI-SDR|开发验证STOI|秒|Torch分配峰值GiB|Torch保留峰值GiB|旧引擎诊断选步(未使用)|','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in outcomes:lines.append(f'|{r["seed"]}|3000|{r["validation_final_SI"]:.3f}|{r["validation_final_STOI"]:.4f}|{r["elapsed_seconds"]:.1f}|{r["peak_allocated_bytes"]/2**30:.3f}|{r["peak_reserved_bytes"]/2**30:.3f}|{r["diagnostic_engine_selected_step_not_used"]}|')
    lines+=['','全部准备数据SHA及执行代码SHA核验通过。当前未运行新适配器在G14或未查看语句上的测试，不能将验证数字作为效果结论。正确旋转未必胜过改变相对几何的反向增强；下一阶段必须同时保留固定/反向/刚性全部三种子与成熟MVDR对照，不选择有利角度或只报最好种子。',
       '', '先用已查看正确场景诊断，再冻结独立新语音确认协议。没有新的人工逐词标签、稳定吞字结论或结构创新声称。','', 'UTC：'+now()]
    report=ROOT/'docs/G15B_正确旋转三种子训练与注册报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[OUT/'training_audit.json',OUT/'registered_checkpoints.json',OUT/'data_file_hashes.json',ROOT/'scripts/g15_finish.py']
    for seed in plan['seeds']:
        folder=OUT/'train'/f'without_clean_identity_s{seed}';files += [folder/'config.json',folder/'result.json',folder/'steps.jsonl',folder/'validation_03000.json']
    archive('training',report,files);dump(OUT/'MAJOR_STAGE_COMPLETE.json',{'status':'ok','stage':'G15','completed_utc':now(),'report':str(report.relative_to(ROOT))},True);status(stage='complete_pending_acceptance')


if __name__=='__main__':finish()
