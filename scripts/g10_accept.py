#!/usr/bin/env python3
"""Post-stage acceptance with full tests, resource and metadata revision records."""
import json
from pathlib import Path
import numpy as np
from g10_common import ROOT,OUT,now,sha,read,dump,archive
from mcse.asr_scoring import corpus_score


def main():
    result=json.loads((OUT/'evaluation/summary.json').read_text());log=OUT/'verification/unit_tests.log';text=log.read_text();amend=json.loads((OUT/'code_revisions/pre_asr/amendment.json').read_text())
    checks={'G10_audit_ok':json.loads((OUT/'completion_audit.json').read_text())['status']=='ok',
      'full77_tests_passed':'Ran 77 tests' in text and '\nOK' in text,
      'temporary_awake_released':json.loads((OUT/'power_helper.json').read_text())['status']=='released',
      'report_only_revision_before_ASR':amend['before_ASR'] and sha(OUT/'code_revisions/pre_asr/initial_g10_asr.py')==amend['initial_sha256']}
    actual=next(r for r in json.loads((OUT/'stages/whisper/archive_manifest.json').read_text())['files'] if r['source']=='scripts/g10_asr.py')
    checks['Whisper_used_archived_corrected_ASR_version']=actual['sha256']==amend['current_sha256']
    total=0;valid=True
    for p in (OUT/'stages').glob('*/archive_manifest.json'):
        for r in json.loads(p.read_text())['files']:valid &= sha(p.parent/r['snapshot'])==r['sha256'];total+=1
    checks['all_stage_snapshots_match']=bool(valid)
    verdict={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'snapshot_files_checked':total,'completed_utc':now()}
    if verdict['status']!='ok':raise RuntimeError(json.dumps(verdict))
    rows=read(OUT/'evaluation/combined_signals.jsonl');wi={(r['system'],r['sample_id']):r for r in read(OUT/'evaluation/Whisper/asr_scores.jsonl')};ci={(r['system'],r['sample_id']):r for r in read(OUT/'evaluation/CTC/asr_scores.jsonl')}
    registry=json.loads((OUT/'systems.json').read_text());per_seed={}
    for system in [r['system'] for r in registry['direct'] if r['kind']=='adapter']+[r['system'] for r in registry['beam'] if r.get('seed') is not None]:
        per_seed[system]={}
        for condition in ['clean']+list(result['cells']):
            for t in (-40,20,70):
                group=[r for r in rows if r['system']==system and r['target_azimuth']==t and ((condition=='clean' and r['condition']=='clean') or (condition!='clean' and r['noise_id']==condition and r['condition']!='clean'))];assert group
                per_seed[system][condition+'|'+str(t)]={'records':len(group),'si_sdr_db':float(np.mean([r['metrics']['si_sdr_db'] for r in group])),
                  'Whisper':corpus_score(wi[(r['system'],r['sample_id'])]['score'] for r in group),'CTC':corpus_score(ci[(r['system'],r['sample_id'])]['score'] for r in group)}
    seed_path=OUT/'evaluation/per_seed_readback.json';dump(seed_path,{'notice':'Every frozen adapter seed,descriptive readback only,no winner selection','systems':per_seed},True)
    def merged(group,t):
        values=[result['cells'][noise][group][str(t)] for noise in result['cells']]
        return {'SI':float(np.mean([r['metrics']['si_sdr_db'] for r in values])),
          **{engine:corpus_score(r[engine] for r in values)['wer'] for engine in ('Whisper','CTC')}}
    lines=['# G10 阶段验收与简明读回','','16条新语句、4位新说话人的15系统比较及双ASR均已完成并归档。完整77项测试通过，阶段快照、源/输入/输出/权重哈希和缓存来源已审计。实验没有新训练，不代表创新成立或真实词语完整性已证实。',
       '', '下表合并四种既有噪声，仅作便于阅读的点估计；详细分噪声、每人和4说话人bootstrap区间在主报告/机器汇总，不能用平均数掩盖反例。SI-SDR正差好，WER负差好。','','|比较|方向|ΔSI-SDR dB|ΔWhisper WER百分点|ΔCTC WER百分点|','|---|---:|---:|---:|---:|']
    comparisons=[('rotated - fixed','rotated','fixed'),('rotated_MVDR - fixed_MVDR','rotated_mvdr','fixed_mvdr')]+[(m+'_MVDR - direct',m+'_mvdr',m) for m in ('mcnet','fixed','rotated')]
    for label,a,b in comparisons:
        for t in (-40,20,70):
            av,bv=merged(a,t),merged(b,t);lines.append(f'|{label}|{t}|{av["SI"]-bv["SI"]:+.2f}|{(av["Whisper"]-bv["Whisper"])*100:+.1f}|{(av["CTC"]-bv["CTC"])*100:+.1f}|')
    lines+=['','干净识别基准使用渲染后的PCM24参考，每源句一次；必要时包含共同防削波缩放，并非直接使用未经处理的原FLAC。机器汇总notice中的unscaled-equivalent不能理解为幅度逐样本相同，以实际参考路径、哈希和common_scale为准。',
      '', '资源记录：变长批次曾观察设备总显存7,830MiB/8,188MiB；该数包含系统及缓存，不是Torch活动分配峰值，无法仅凭nvidia-smi区分缓存与活动张量。批次成功不等于可忽略8GB余量限制，G6固定长度资源数不能直接代替本批。未修改驱动、CUDA、系统电源计划或重启模型进程。',
      '', '执行中、ASR尚未启动时修正了两处报告归档元数据：缓存优先级文字补上G9；Whisper快照文件由误指g9_asr.py改为实际g10_asr.py。初版代码及前后哈希和说明已保存，未改数值流程、系统、权重、选句或解码计分。原PIPELINE_READY记录保留，不覆盖历史。',
      '', '后续决策应优先依据新说话人的分条件事实。已查看本批后不能再以它调参并继续称最终盲测；真实混响/硬件阵列、更多独立说话人及人工词级验证仍缺。文献补查已明确低秩增强适配、音素损失和神经波束形成都有先例。',
      '', '每个适配器种子的分噪声/方向及干净输入读回另存[per_seed_readback.json](../artifacts/g10_fresh_voice_v1/evaluation/per_seed_readback.json)，保留全部三种子，不挑赢家。',
      '', '[主报告](G10_新说话人直接输出与MVDR机制验证报告.md)、[文献补查](研究补查_轻量适配与神经波束形成先例_20261112.md)、[完整测试日志](../artifacts/g10_fresh_voice_v1/verification/unit_tests.log)。','', 'UTC：'+now()]
    report=ROOT/'docs/G10_阶段验收与简明读回.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',verdict,True)
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',log,seed_path,OUT/'runtime_resource_observation.json',OUT/'runtime_resource_after_signal.json',OUT/'code_revisions/pre_asr/amendment.json',OUT/'code_revisions/pre_asr/initial_g10_asr.py',OUT/'PIPELINE_READY.json',Path(__file__)]);print(json.dumps(verdict))


if __name__=='__main__':main()
