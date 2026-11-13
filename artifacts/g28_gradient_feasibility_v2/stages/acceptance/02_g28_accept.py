"""Independent arithmetic and provenance acceptance of the no-update probe."""
import numpy as np
from g28_gradient_v2 import ROOT,OUT,sha,dump,now,meta,archive


def main():
    done=meta(OUT/'COMPLETE.json');plan=meta(OUT/'frozen_plan.json');rows=meta(OUT/'results.json')
    assert done['status']=='ok' and done['optimizer_updates']==0 and len(rows)==18
    assert sha(OUT/'results.json')==done['results_sha256'] and sha(OUT/'gradients.npz')==done['gradients_sha256']
    assert meta(OUT/'power_helper.json')['status']=='released'
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h
    stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    arrays=np.load(OUT/'gradients.npz');expected={f"record_{r['index']}_{o}" for r in plan['records'] for o in plan['objectives']}
    assert set(arrays.files)==expected
    for r in rows:
        v=arrays[f"record_{r['index']}_{r['objective']}"];base=arrays[f"record_{r['index']}_original_cIRM"]
        assert v.shape==(3584,) and np.isfinite(v).all()
        norm=np.linalg.norm(v);assert norm>0 and np.isclose(norm,r['gradient_norm'],rtol=1e-12,atol=1e-14)
        cosine=v@base/(norm*np.linalg.norm(base));assert np.isclose(cosine,r['cosine_to_original_cIRM'],rtol=1e-12,atol=1e-14)
    check=meta(OUT/'finite_difference.json');assert check['absolute_error']<1e-9+2e-4*abs(check['analytic_gradient'])
    assert abs(check['identical_target_loss'])<1e-20
    import runpy
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g28_accept_runtime')
    repo=ROOT/'third_party/McNet';upstream=rt['verify_upstream'](repo,repo/rt['OFFLINE_CHECKPOINT'])
    audit={'status':'ok','utc':now(),'records':6,'gradient_probes':18,'optimizer_updates':0,
        'losses_with_finite_nonzero_gradients':plan['objectives'],'max_Torch_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),
        'parameter_digests_reported_unchanged':done['parameter_digests_unchanged'],'stages':stages,
        'completed_sha256':sha(OUT/'COMPLETE.json'),'failed_v1_preserved':True,
        'helper_note':'Helper started after calculation completed and immediately released; did not cover calculation period',
        'peak_scope':'16128sample FP32 forward+gradient, no optimizer; not whole GPU usage or4s batch capacity'}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G28_梯度预检验收与后续边界.md'
    text=['# G28 梯度预检验收与后续边界','','v1 CPU复数类型错误及其冻结证据保留；v2只修正dtype转换，18项有限非零梯度通过。另从保存的3584维梯度向量独立重算范数和余弦，核对有限差分、原始模型来源及所有快照哈希。优化器更新为0。', '',
        f"该1.008秒训练短段的Torch已分配显存峰值为{audit['max_Torch_allocated_bytes']/2**30:.3f} GiB，不外推长段、其他batch或完整显卡占用。", '',
        '其中一条训练记录的协方差形状目标梯度与原cIRM目标余弦为负，其余已测记录不是同一方向；这只是初始局部梯度，既不能断言二者必然冲突，也不能断言更大梯度会更好。直接掩码MSE保留为必要成熟对照，防止仅比原McNet获得虚假的创新印象。', '',
        '临时唤醒助手启动时计算已结束，立即检测到停止标记并释放；未覆盖计算全程，未修改电源计划。不会将其描述为全程保活已验证。', '',
        '后续应先准备严格训练/开发隔离的小规模混响条件，冻结训练预算及完整对照，再检验可学习性。G22/G27旧四句的真值诊断不作为选系数或选checkpoint的验证集。新目标与成熟空间统计监督有先例，暂不认定创新。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g28_accept.py',OUT/'power_helper.json'])
    dump(OUT/'status.json',{'stage':'accepted','utc':now()});print({'status':'accepted','optimizer_updates':0,'peak_GiB':audit['max_Torch_allocated_bytes']/2**30})


if __name__=='__main__':main()
