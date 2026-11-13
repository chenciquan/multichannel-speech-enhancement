import hashlib
import numpy as np
from g33_common import ROOT,OUT,sha,dump,now,meta,archive,guard,status


def main():
    plan=guard();done=meta(OUT/'PROBE_COMPLETE.json');assert done['status']=='ok' and done['optimizer_updates']==0
    assert meta(OUT/'power_helper.json')['status']=='released'
    results={};outputs={}
    for mode in plan['modes']:
        p=OUT/mode;r=meta(p/'COMPLETE.json');assert sha(p/'COMPLETE.json')==done['result_hashes'][mode]
        assert r['status']=='ok' and r['optimizer_updates']==0 and r['original_parameters_unchanged']
        assert not r['optimizer_created'] and not r['development_or_test_audio_access']
        assert sha(p/'records.json')==r['records_sha256'] and sha(p/'initial_forward_outputs.npz')==r['forwards_sha256']
        records=meta(p/'records.json');assert [v['index'] for v in records]==plan['indices']
        for v in records:
            assert v['connected_parameters']==r['trainable_parameters'] and v['frames']==23679
            assert sum(x['parameters'] for x in v['parameter_grad_norms'])==r['trainable_parameters']
            assert np.isclose(np.sqrt(sum(x['norm']**2 for x in v['parameter_grad_norms'])),v['gradient_norm'],rtol=1e-12)
        outputs[mode]=np.load(p/'initial_forward_outputs.npz');results[mode]=r
    assert results['lora']['base_digest']==results['full']['base_digest']
    assert results['lora']['trainable_parameters']==3584 and results['full']['trainable_parameters']==3328706
    errors=[]
    for index in plan['indices']:
        a=outputs['lora'][f'record_{index}'];b=outputs['full'][f'record_{index}']
        assert a.shape==b.shape and np.allclose(a,b,rtol=plan['forward_equivalence_rtol'],atol=plan['forward_equivalence_atol'])
        errors.append({'index':index,'max_abs':float(np.abs(a-b).max()),'exact':bool(np.array_equal(a,b))})
    stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    audit={'status':'ok','utc':now(),'records_each':6,'modes':2,'optimizer_updates':0,'parameters_unchanged':True,
        'forward_equivalence':errors,'results':results,'stages':stages,'power_released':True,
        'comparison_scope':'no-optimizer originalcIRM forward+gradient on1.48s trainsegments, not full training or4s benchmark'}
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',audit,True)
    report=ROOT/'docs/G33_全参数与LoRA资源预检验收.md'
    lines=['# G33 全参数与LoRA资源预检验收','','两个独立进程，各六条相同训练输入，原cIRM前向和梯度完成，开放参数全部连通且梯度有限。初始基础参数摘要一致，零残差LoRA前向与原模型在预登记容差内等价；全过程没有优化器、参数更新或开发/测试访问，原模型文件及所有来源保持不变。', '',
        '|模式|开放参数|Torch已分配峰值GiB|Torch保留峰值GiB|首例计算秒|后五例计算中位秒|两FP32 Adam矩估算MiB|','|---|---:|---:|---:|---:|---:|---:|']
    for m in plan['modes']:
        r=results[m];lines.append(f"|{m}|{r['trainable_parameters']}|{r['peak_allocated_bytes']/2**30:.3f}|{r['peak_reserved_bytes']/2**30:.3f}|{r['cold_first_compute_seconds']:.4f}|{r['warm5_compute_median_seconds']:.4f}|{r['two_FP32_Adam_moments_estimated_bytes']/2**20:.3f}|")
    lines+=['','参数节省比例不能直接当显存或速度节省比例。以上为1.48秒短段、FP32 batch1的Torch前向/梯度峰值，不含优化器状态、驱动/其他进程；Adam两矩只是按每参数8字节估算，并非实际分配或完整状态大小。计时排除模型加载、审计及CPU梯度拷贝；仅一种模式顺序、六条样本，后五例中位也不是严谨吞吐基准。', '',
        '当前结果只支持判断是否值得试做全参数微调强对照；未训练完整模型，不能推断质量或“LoRA优于全微调”。若后续训练，需登记有限学习率/预算并报告优化器实际峰值，保持同数据、相同目标与比较边界，不把一个未充分调节的全参数结果当作其最佳性能。', '',
        'G31扩展门槛仍未通过，G32仅四条旧开发整句的识别补充，不扩大统计辅助训练或认定内容保真、创新与真实阵列泛化。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g33_accept.py',OUT/'power_helper.json'])
    status(stage='accepted_no_updates');print({m:{'parameters':r['trainable_parameters'],'peak_GiB':r['peak_allocated_bytes']/2**30,'warm_seconds':r['warm5_compute_median_seconds']} for m,r in results.items()},flush=True)


if __name__=='__main__':main()
