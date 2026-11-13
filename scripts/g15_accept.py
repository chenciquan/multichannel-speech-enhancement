#!/usr/bin/env python3
import json
from g15_common import ROOT,OUT,now,sha,dump,archive


def main():
    result=json.loads((OUT/'training_audit.json').read_text());assert json.loads((OUT/'MAJOR_STAGE_COMPLETE.json').read_text())['status']=='ok'
    plan=json.loads((OUT/'frozen_plan.json').read_text());reg=json.loads((OUT/'registered_checkpoints.json').read_text());prior=json.loads((ROOT/'artifacts/g5_rotation_v1/stages/training/archive_manifest.json').read_text());engine=next(r for r in prior['files'] if r['source']=='scripts/g3_train.py')
    checks={'training_audit_ok':result['status']=='ok','all3_registered3000steps':len(reg['systems'])==3 and all(r['step']==3000 for r in reg['systems']),
        'all_registered_checkpoint_hashes_match':all(sha(ROOT/r['path'])==r['checkpoint_sha256'] for r in reg['systems']),
        'same_engine_as_archived_G5trainer':sha(ROOT/'scripts/g3_train.py')==engine['sha256'],
        'temporary_awake_released':json.loads((OUT/'power_helper.json').read_text())['status']=='released'}
    prep=json.loads((OUT/'stages/preparation/archive_manifest.json').read_text());checks['data_report_saved_before_all_training_updates']=all(prep['saved_utc']<json.loads((OUT/'train'/f'without_clean_identity_s{seed}'/'config.json').read_text())['started_at'] for seed in plan['seeds'])
    n=0;valid=True
    for p in (OUT/'stages').glob('*/archive_manifest.json'):
        for r in json.loads(p.read_text())['files']:valid&=sha(p.parent/r['snapshot'])==r['sha256'];n+=1
    checks['all_prior_snapshot_hashes_match']=bool(valid);final={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'snapshot_files_checked':n,'completed_utc':now()};assert final['status']=='ok'
    report=ROOT/'docs/G15_训练阶段验收与后续评估边界.md';report.write_text('# G15 训练阶段验收与后续评估边界\n\n三个正确刚性旋转种子均完成3000次有效更新，总9000次；注册检查点与SHA核验通过。原McNet参数未变，NumPy采样状态/分类计数及CPU/CUDA随机状态与旧fixed和counter同种子一致。训练器SHA与G5归档版本完全相同，复用的是G5实际使用的现有版本。\n\n2048训练输入和54旧验证输入逐样本复现，新参考麦输入/监督不变；原数据和模型保留。数据报告先于训练，阶段快照与临时保持唤醒释放核验通过。G14已有98项完整测试通过，此后共享数值源码未修改，G15不重复整套测试；本阶段以全部数组还原、训练有效步数/梯度有限性/冻结基线和随机状态实际核验补充。\n\n本阶段只证明正确训练按协议完成，不证明性能更好。末步3000全部注册，原引擎selected字段未使用；所有不利验证结果保留。后续先对已查看正确几何开发条件诊断，再做新说话人小规模确认。不能把已查看的G14重新称为未经触碰最终测试，也不预设正确刚性训练优于反向联合分布增强。\n\nUTC：'+now()+'\n',encoding='utf-8');dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',final,True);archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',OUT/'registered_checkpoints.json',ROOT/'scripts/g15_accept.py']);print(json.dumps(final))


if __name__=='__main__':main()
