from g33_common import ROOT,OUT,sha,read,dump,now,meta,archive,status


def main():
    prior=ROOT/'artifacts/g32_complete_dev_v1';assert meta(prior/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    assert (prior/'stages/scientific_readback/archive_manifest.json').exists()
    assert meta(ROOT/'artifacts/g31_development_signals_v1/primary_decision.json')['extension_gate_passed'] is False
    OUT.mkdir(exist_ok=False)
    names=['scripts/g33_common.py','scripts/g33_prepare.py','scripts/g33_probe.py','scripts/g33_queue.py','scripts/g33_accept.py',
        'scripts/g28_gradient_v2.py','scripts/mcnet_infer.py','src/mcse/adaptation.py','scripts/g8_common.py',
        'data/g29_v1/train_x.npy','data/g29_v1/train_y.npy','data/g29_v1/train_manifest.jsonl',
        'artifacts/g29_room_development_v1/FINAL_ACCEPTANCE_AUDIT.json','artifacts/g30_room_objectives_v1/frozen_plan.json',
        'artifacts/g32_complete_dev_v1/FINAL_ACCEPTANCE_AUDIT.json']
    indices=meta(ROOT/'artifacts/g30_room_objectives_v1/frozen_plan.json')['calibration_indices']
    records=read(ROOT/'data/g29_v1/train_manifest.jsonl');assert len(indices)==6 and len(set(indices))==6
    plan={'utc':now(),'modes':['lora','full'],'indices':indices,'training_records':[records[i] for i in indices],
        'seed':20261113,'FP32':True,'batch':1,'frames':23679,'loss':'unchanged original compressedcIRM MSE',
        'optimizer_updates':0,'optimizer_created':False,'max_mode_seconds':600,'max_queue_seconds':1500,
        'comparison':'same source CP and same six training inputs; reset forwardRNG perindex, independent processes permode',
        'forward_equivalence_rtol':1e-5,'forward_equivalence_atol':1e-6,
        'memory':'torch allocated/reserved at end of forward+autograd.grad, no Adam state;2FP32moment bytes estimated separately',
        'timing':'synchronized forward+gradient only, excludes audit/CPUcopies/sourcehashes; firstcase cold reported separately, oneordernotthroughputbenchmark',
        'no_development_or_test_audio':True,'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G33A_全参数与LoRA训练资源预检协议.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G33A 全参数与LoRA训练资源预检\n\n目的：为“小参数适配有何实际资源价值”补全参数微调成熟对照的可行性证据，不继续扩G31未通过门槛的统计辅助训练。相同原McNet CP、G29六条固定训练记录、原cIRM目标、FP32 batch1。LoRA用相同两处空间FC rank4/alpha4；完整模式开放原模型全部参数。各模式独立进程，逐记录重置前向随机种子。\n\n只计算前向及autograd.grad，不创建优化器、更新参数或读开发/测试音频。每模式600秒，总队列1500秒；非有限或未连接参数/OOM保留并停止，不自动调batch/精度或重试。保存原始参数摘要不变、前向与梯度范数/连通性、实际Torch已分配/保留峰值与同步计算耗时。\n\n检查初始零残差LoRA与原模型前向在预登记容差内等价；完整模式未改变模型结构。显存不含Adam状态、驱动/其他进程；2个FP32矩的字节数另作估算，不当作实测训练峰值。首例冷启动与后五例分开，一种运行顺序的短预检不作正式吞吐基准，也不外推4秒梯度或更大batch。官方预训练配置的Adam lr=.001/指数衰减不等于小数据微调最优；后续若训练，学习率与同预算对照另登记，不拿一组差的全参数结果当其最好水平。\n')
    archive('protocol',report,[OUT/'frozen_plan.json']+[ROOT/n for n in names[:9]])
    status(stage='prepared_no_updates');print({'status':'prepared','indices':indices})


if __name__=='__main__':main()
