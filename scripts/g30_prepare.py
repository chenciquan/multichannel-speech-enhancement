#!/usr/bin/env python3
import numpy as np
from g30_common import ROOT,OUT,DATA,sha,read,dump,now,meta,archive,guard,status


def main():
    accepted=ROOT/'artifacts/g29_room_development_v1/FINAL_ACCEPTANCE_AUDIT.json'
    assert meta(accepted)['status']=='ok' and meta(accepted)['full_replays']==684
    OUT.mkdir(exist_ok=False)
    records=read(DATA/'train_manifest.jsonl');templates={r['id']:r for r in meta(ROOT/'artifacts/g29_room_development_v1/frozen_plan.json')['templates']}
    calibration=[]
    for room in ('train0','train1','train2'):
        for field in ('point','multi8'):calibration.append(next(r['index'] for r in records if templates[r['template_id']]['room_id']==room and r['field']==field))
    assert len(set(calibration))==6
    names=['scripts/g30_prepare.py','scripts/g30_common.py','scripts/g30_calibrate.py','scripts/g30_train.py','scripts/g30_queue.py',
           'scripts/g30_accept.py','scripts/g28_gradient_v2.py','scripts/mcnet_infer.py','src/mcse/adaptation.py',
           'scripts/g29_common.py','scripts/g8_common.py','configs/g31_development_protocol.json',
           'data/g29_v1/train_x.npy','data/g29_v1/train_y.npy','data/g29_v1/train_manifest.jsonl',
           'artifacts/g29_room_development_v1/FINAL_ACCEPTANCE_AUDIT.json']
    plan={'utc':now(),'seed':20261113,'steps_each':1500,'variants':['cirm','cirm_mask','cirm_shape'],
        'learning_rate':.0003,'optimizer':'Adam','gradient_clip':5.,'batch':1,'FP32':True,'AMP':False,
        'rank':4,'alpha':4,'trainable_parameters':3584,'checkpoint_steps':[500,1000,1500],
        'registration':'Prescribed1500 for all variants, no development/model selection during training',
        'sampling':'Uniform fixed schedule of1500 indices from576, same for all variants; no validation/test in trainer',
        'calibration_indices':calibration,'calibration_aux_strength':.1,
        'coefficient_rule':'lambda=.1*median_over6(||grad originalcIRM||2/||grad auxiliary||2), separate maskedMSE and shape; initialsameLoRA, no updates, no dev/test score access',
        'losses':{'cirm':'original compressed cIRM MSE','cirm_mask':'cIRM+lambda_mask*power_ratio_mask_MSE',
                  'cirm_shape':'cIRM+lambda_shape*.5*(speech+noise true-mask-mixture covariance trace-shape loss)'},
        'truth_scope':'Auxiliary statistic target uses oracle power mask applied to noisy mixture, NOT true6ch source SCM',
        'max_calibration_seconds':600,'max_run_seconds':1200,'max_queue_seconds':4500,
        'failure_policy':'Preserve and stop on timeout, nonfinite, wrong step count, unequal initial/base digests or data modification; no silent retry',
        'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'frozen_plan.json',plan,True)
    rng=np.random.default_rng(plan['seed']);schedule=rng.integers(0,576,size=1500,dtype=np.int64)
    with (OUT/'sample_indices.npy').open('xb') as f:np.save(f,schedule)
    report=ROOT/'docs/G30A_混响轻量适配同预算目标对照协议.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G30A 混响轻量适配同预算目标对照协议\n\n仅使用G29固定576训练短段，三组同一官方McNet初始化、同两处空间FC LoRA(rank4/alpha4，3584参数)、同种子20261113、同1500条抽样索引。Adam lr3e−4、FP32、batch1、梯度裁剪5。组1原压缩cIRM；组2加简单功率比掩码MSE；组3加真值掩码产生的混合协方差形状监督。后两者非新发明，保留原重建项以约束直接语音输出。\n\n辅助尺度在任何更新和开发评分前计算：取每个训练房间×point/multi8的首条，共6条，在相同初始前向求各目标梯度范数，lambda=.1×六条主/辅助梯度范数比的中位数。掩码和统计目标分别校准，.1事前固定为较弱初始辅助强度；不是最优系数，也不保证逐例梯度都较弱。零/非有限梯度停止。此尺度平衡是工程控制，不宣称创新。\n\n每组1500次有效更新，保存500/1000/1500步及日志，只登记1500步，过程中不加载开发/测试语音，不按成绩选checkpoint或加训。保留全部组及不利结果；主干摘要必须逐位不变，三组初始适配器摘要及抽样一致。校准600秒、每组1200秒、总队列4500秒上限，失败留存并停止，不自动重试。\n\n这是单种子有限开发试验，不是最终效果验证。完成训练归档后再登记开发评价，直接输出和成熟MVDR/MWF均保留，干净RMS/NMSE另算。开发短裁剪不许整句WER。G22/G27旧四句不进入校准、训练或选择；未来新完整语句确认及三种子扩展另设门槛。\n')
    archive('protocol',report,[OUT/'frozen_plan.json',OUT/'sample_indices.npy']+[ROOT/n for n in names[:9]])
    dump(OUT/'READY_FOR_CALIBRATION.json',{'status':'ok','utc':now(),'plan_sha256':sha(OUT/'frozen_plan.json'),'schedule_sha256':sha(OUT/'sample_indices.npy')},True)
    status(stage='prepared_calibration_not_started');print({'status':'prepared','calibration_indices':calibration})


if __name__=='__main__':main()
