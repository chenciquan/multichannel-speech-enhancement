"""Read accepted numerical controls; keep the original failed gate intact."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import g8_common as common
ROOT=common.ROOT;OUT=ROOT/'artifacts/g33_precision_audit_v1';OLD=ROOT/'artifacts/g33_full_gradient_probe_v1'
sha,dump,now=common.sha,common.dump,common.now


def meta(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def archive(stage,report,files):
    previous=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous


def main():
    assert meta(OUT/'FINAL_DIAGNOSTIC_AUDIT.json')['status']=='ok' and not (OLD/'FINAL_ACCEPTANCE_AUDIT.json').exists()
    plan=meta(OUT/'frozen_plan.json');summary=meta(OUT/'summary.json');comparisons=meta(OUT/'comparisons.json')
    a=np.load(OUT/'predictions.npz');w=np.load(OUT/'waveforms.npz')
    # Independently check the relative-error fields used below, beyond the acceptance gate counts.
    for r in comparisons:
        def key(m,g,rp=0):return f"{r['profile']}__{r['index']}__{m}__grad{int(g)}__repeat{rp}"
        name=r['comparison']
        if name=='cross_model_grad':x,y=key('lora',True),key('full',True)
        elif name=='cross_model_no_grad':x,y=key('lora',False),key('full',False)
        elif name in ('full_grad_flag','lora_grad_flag'):
            m=name.split('_')[0];x,y=key(m,True),key(m,False)
        else:
            m=name.split('_')[1];g=name.endswith('1');x,y=key(m,g,0),key(m,g,1)
        for data,label in ((a,'prediction'),(w,'waveform')):
            numerator=float(np.linalg.norm((data[x]-data[y]).astype(np.complex128).ravel()))
            denominator=max(float(np.linalg.norm(data[y].astype(np.complex128).ravel())),1e-30)
            assert np.isclose(numerator/denominator,r[label]['relative_RMS'],rtol=1e-12,atol=1e-15)
    repeated=[r for r in comparisons if r['comparison'].startswith('repeat_')]
    grad_flags=[r for r in comparisons if r['comparison'] in ('full_grad_flag','lora_grad_flag')]
    replay=meta(OUT/'old_G33_replay.json')
    det_controls=[]
    for left,right in (('cudnn_tf32_nondet','cudnn_tf32_det'),('cudnn_fp32_nondet','cudnn_fp32_det')):
        exact_pred=exact_wave=0;maximum=0.
        for index in plan['indices']:
            for mode in ('full','lora'):
                for grad in (0,1):
                    for repeat in (0,1):
                        suffix=f'__{index}__{mode}__grad{grad}__repeat{repeat}'
                        x,y=left+suffix,right+suffix;exact_pred+=int(np.array_equal(a[x],a[y]));exact_wave+=int(np.array_equal(w[x],w[y]))
                        maximum=max(maximum,float(np.abs(a[x]-a[y]).max()))
        det_controls.append({'left':left,'right':right,'pairs':48,'exact_predictions':exact_pred,'exact_waves':exact_wave,'prediction_max_abs':maximum})
    main_rows=[next(r for r in summary if r['profile']==p['id'] and r['comparison']=='cross_model_grad') for p in plan['profiles']]
    observations={'utc':now(),'repeat_pairs':len(repeated),'repeat_exact':sum(r['prediction']['exact'] for r in repeated),
        'grad_flag_pairs':len(grad_flags),'grad_flag_exact':sum(r['prediction']['exact'] for r in grad_flags),
        'old_replays':len(replay),'old_replays_exact':sum(r['exact'] for r in replay),'determinism_flag_controls':det_controls,
        'main_rows':main_rows,'relative_error_fields_recomputed':240,'G33_original_gate_still_failed':True}
    dump(OUT/'scientific_readback.json',observations,True)
    labels=['cuDNN\nTF32 allowed\ndet flag off','cuDNN\nTF32 off\ndet flag off','cuDNN\nTF32 allowed\ndet flag on','cuDNN\nTF32 off\ndet flag on','Native\nTF32 off']
    fig,axes=plt.subplots(2,1,figsize=(10.5,7),sharex=True)
    for ax,metric,count,title in ((axes[0],'prediction_max_abs','prediction_tolerance_pass','Compressed cIRM max absolute difference'),
                                  (axes[1],'waveform_max_abs','waveform_tolerance_pass','Reconstructed waveform max absolute difference')):
        vals=[r[metric] for r in main_rows];ax.bar(range(5),vals,color=['#b66a2c','#356d95','#b66a2c','#356d95','#6c8871'])
        ax.set_yscale('log');ax.set_ylim(min(vals)/3,max(vals)*4);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        for i,(v,r) in enumerate(zip(vals,main_rows)):ax.text(i,v*1.25,f"{r[count]}/6 within original tolerance",ha='center',fontsize=8)
    axes[1].set_xticks(range(5),labels);fig.suptitle('Zero-LoRA vs full model: same six training inputs, no parameter updates\nNo pointwise threshold was changed; pass count uses rtol=1e-5 / atol=1e-6',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9))
    for ext in ('png','pdf'):fig.savefig(OUT/f'precision_control_differences.{ext}',dpi=180)
    plt.close(fig)
    report=ROOT/'docs/G33B_科学读回与统一执行精度建议.md'
    lines=['# G33B 科学读回与统一执行精度','','本次数值矩阵已完整验收；原G33前向门槛仍未通过，未补写通过标记。独立核对了240个比较的相对误差字段。以下都是同六条训练记录的数值观察，不是模型效果或语音内容证据。', '',
        f"同路径重复预测逐位一致：{observations['repeat_exact']}/{observations['repeat_pairs']}；求导开关前后逐位一致：{observations['grad_flag_exact']}/{observations['grad_flag_pairs']}；新默认相似配置精确复现原G33：{observations['old_replays_exact']}/{observations['old_replays']}。因此这批观察不支持把差异归因于同路径前向随机性或是否建立求导图。", '',
        '|配置|压缩cIRM最大差|原容差通过/6|重建波形最大差|原容差通过/6|','|---|---:|---:|---:|---:|']
    for r in main_rows:lines.append(f"|{r['profile']}|{r['prediction_max_abs']:.8g}|{r['prediction_tolerance_pass']}|{r['waveform_max_abs']:.8g}|{r['waveform_tolerance_pass']}|")
    lines+=['','关闭cuDNN TF32许可后，跨模型最大差显著减小；deterministic开关在这批输入/两次重复中没有改变结果，逐配置48对比较见scientific_readback。说明后端精度许可是本次差异的重要影响因素；没有跟踪到具体算子或GPU指令，不能断言每个RNN实际使用TF32，残余约数e−6的cIRM差异也尚未定位到某一实现细节。无cuDNN控制同样仍有cIRM小残差，并非精度真值。', '',
        '数学相同的浮点图不保证逐位相同，且后端组织可影响结果，见[PyTorch v2.5.1 数值精度说明](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/numerical_accuracy.rst)。这条一般规则不消除原失败，也不证明残差对训练轨迹、ASR或人耳无害。', '',
        '## 后续采用什么边界', '',
        '建议后续全参数/LoRA强对照统一使用cuDNN启用、TF32许可均关闭、benchmark=False、cuDNN deterministic=False的明确进程配置，并固定版本与随机种子。选择依据是训练输入上的数值定位，不是开发成绩；它不启用全局确定性算法、不改系统环境。该配置在本六条的重建波形接口通过了事前G33B固定容差，压缩cIRM仅3/6通过原逐元素容差，二者分别保留。', '',
        '新训练若采用此配置，必须重新跑普通LoRA原cIRM匹配对照，不能把旧TF32许可配置训练的G30权重直接当作完全匹配基线。原始权重摘要相同、零B残差与完整源代码验证是初始模型依据；不要求两种优化方法训练轨迹一致，也不假称cIRM数值已完全相等。新实验可以研究开发质量及实际Adam资源，但不得称原G33公平性门槛已过。', '',
        '若进一步做全参数微调，应限定并登记少量学习率，统一1500步与共享训练索引，保留全部点，不把差的单点当作全微调最好水平；新增匹配LoRA后才能讨论该配置下的参数/资源/质量权衡。尚无全参数适配后的质量结果，不扩大G31未过门槛的统计辅助训练。', '',
        '![数值后端控制](../artifacts/g33_precision_audit_v1/precision_control_differences.png)', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('scientific_readback',report,[OUT/'scientific_readback.json',OUT/'precision_control_differences.png',OUT/'precision_control_differences.pdf',ROOT/'scripts/g33b_readback.py'])
    print({k:observations[k] for k in ('repeat_exact','repeat_pairs','grad_flag_exact','grad_flag_pairs','old_replays_exact','old_replays')},flush=True)


if __name__=='__main__':main()
