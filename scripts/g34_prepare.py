"""Freeze all implementation and resource/initialization gates before updates."""
import shutil
from g34_common import ROOT,OUT,sha,meta,dump,now,archive,status


def main():
    assert meta(ROOT/'artifacts/g33_precision_audit_v1/FINAL_DIAGNOSTIC_AUDIT.json')['status']=='ok'
    assert meta(ROOT/'artifacts/g30_room_objectives_v1/FINAL_TRAINING_ACCEPTANCE.json')['status']=='ok'
    assert meta(ROOT/'artifacts/g29_room_development_v1/FINAL_ACCEPTANCE_AUDIT.json')['full_replays']==684
    assert not (ROOT/'artifacts/g33_full_gradient_probe_v1/FINAL_ACCEPTANCE_AUDIT.json').exists()
    OUT.mkdir(exist_ok=False)
    shutil.copy2(ROOT/'artifacts/g30_room_objectives_v1/sample_indices.npy',OUT/'sample_indices.npy')
    names=['scripts/g34_common.py','scripts/g34_prepare.py','scripts/g34_initial.py','scripts/g34_train.py',
        'scripts/g34_queue.py','scripts/g34_accept.py','scripts/g8_common.py','scripts/mcnet_infer.py','src/mcse/adaptation.py',
        'data/g29_v1/train_x.npy','data/g29_v1/train_y.npy','data/g29_v1/train_manifest.jsonl',
        'artifacts/g29_room_development_v1/FINAL_ACCEPTANCE_AUDIT.json',
        'artifacts/g30_room_objectives_v1/FINAL_TRAINING_ACCEPTANCE.json',
        'artifacts/g30_room_objectives_v1/sample_indices.npy',
        'artifacts/g33_precision_audit_v1/FINAL_DIAGNOSTIC_AUDIT.json',
        'artifacts/g33_precision_audit_v1/scientific_readback.json','docs/G34_统一精度全参数强对照准备草案.md']
    runs=[{'id':'lora_3e4','mode':'lora','lr':3e-4,'parameters':3584}]+[
        {'id':name,'mode':'full','lr':lr,'parameters':3328706} for name,lr in
        [('full_3e5',3e-5),('full_1e4',1e-4),('full_3e4',3e-4)]]
    plan={'utc':now(),'seed':20261113,'runs':runs,'steps_each':1500,'total_updates':6000,
        'optimizer':'Adam','betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'gradient_clip':5.,'batch':1,
        'precision':'FP32,no AMP; cuDNN enabled,TF32 false; matmulTF32 false; benchmark false; cuDNN deterministic false',
        'rank':4,'alpha':4,'objective':'original compressed cIRM MSE; no auxiliary','checkpoint_steps':[500,1000,1500],
        'registration_step':1500,'selection':False,'initial_indices':[0,3,10,5,7,2],
        'initial_wave_rtol':1e-5,'initial_wave_atol':1e-6,'initial_cIRM_residual':'record without reclassifying G33 failed gate',
        'max_initial_seconds':1200,'max_run_seconds':2400,'max_queue_seconds':11200,
        'schedule_sha256':sha(OUT/'sample_indices.npy'),
        'training_pool':'G29 576 fixed TRAIN records; same G30 sampling1500; no development/test audio loaded',
        'rng':'Each independent construction seed20261113; reset seed20261113 after construction before training; initialcheck separateprocess',
        'failure_policy':'first OOM,nonfinite,unconnected parameter,data change,initialwave failure or timeout: preserve and stop; no automatic retry/precision/batch change',
        'resource_scope':'synchronized step GPU/API wall time separately from process/import/hash/I-O wall time; Adamstate actually allocated',
        'limitations':'single seed/limited LR points/small oldtrain; not fully tuned strongest fullmodel; no content or innovation claim',
        'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G34A_统一精度适配强对照冻结协议.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G34A 统一精度适配强对照冻结协议\n\n固定G29训练池与G30同一1500索引。统一FP32并关闭cuDNN/matmul的TF32许可，其他实际开关见frozen_plan。独立运行普通LoRA lr3e−4与全参数lr3e−5/1e−4/3e−4，原cIRM目标、Adam、batch1、clip5，各1500次，共6000更新。保存500/1000/1500步，只登记1500步，无开发/测试音频或分数选择，三个全参数学习率均保留。构造后重置同种子。\n\n先另进程检查六条固定训练输入的初始基础参数摘要、LoRA零B及重建波形接口原容差1e−5/1e−6。cIRM残差另列，旧G33验收失败不改判；该六条检查不保证全部训练输入逐位等价。无优化器的初始化检查不算训练。\n\n初始化子进程1200秒，各训练2400秒，总队列11200秒（含导入/I-O）；首个OOM、非有限、未连通参数、超时或数据变化保存即停，不自动重试。实际Adam状态、逐步有限性/索引、所有checkpoint及显存记录验收后才进入开发评价。同步逐步计时与总墙时分开，不把WSL挂载盘加载时间当GPU计算时间。无新依赖、数据下载、驱动或系统配置修改。\n\n这是成熟强对照的有限开发实验，不扩大G31失败的统计辅助方案。单种子、三学习率不是充分调参，不按好看结果隐藏其他点，也不证明人声误删、创新或发表能力。\n')
    archive('protocol',report,[OUT/'frozen_plan.json',OUT/'sample_indices.npy']+[ROOT/n for n in names[:9]])
    dump(OUT/'READY.json',{'utc':now(),'status':'ok','plan_sha256':sha(OUT/'frozen_plan.json')},True)
    status(stage='prepared_no_updates');print('G34 protocol frozen; no optimizer updates',flush=True)


if __name__=='__main__':main()
