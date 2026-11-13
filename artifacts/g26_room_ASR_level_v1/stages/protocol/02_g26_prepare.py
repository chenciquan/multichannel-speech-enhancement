#!/usr/bin/env python3
"""Freeze full matched-room auxiliary ASR inputs without duplicating audio."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib
import json
import time
from collections import defaultdict
from g25_prepare import ROOT,sha,read,dump,now
import g8_common as common
from g19_scores import waveform
from mcse.asr_level import standardize_level

OUT=ROOT/'artifacts/g26_room_ASR_level_v1';SOURCE=ROOT/'artifacts/g22_matched_rooms_v1';CONTROL=ROOT/'artifacts/g25_asr_level_control_v1'


def archive(stage,report,files):
    previous=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous


def main():
    assert json.loads((CONTROL/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    decision=ROOT/'docs/G25_科学读回与完整房间辅评分决策.md'
    with decision.open('x',encoding='utf-8') as f:
        f.write('# G25 科学读回与完整房间辅评分决策\n\n480个规范化基准的两次识别，两个后端规范化文本变化都为0；所有二进制增益版本在标准化后输入字节一致。规范化评分相对raw有变化，但没有统一改善。正确刚性组MVDR/MWF的Whisper WER从37.89%/45.56%变为38.44%/45.44%，CTC从26.56%/40.11%变为27.78%/39.00%；这批子集中主要排序未变，不能把变化称为增强模型更好。\n\n下一步保留同一预处理（目标RMS=.05、峰值上限=.95），对G22全部16488条输出及相应参考增加独立的ASR辅助评分。G25只覆盖一个噪声点源子集，不能据此判断其余场型、噪声和模型的排序；完整配对读回用于回答该问题。参数不根据新分数重选。\n\n原增强输出、质量指标和raw-ASR表保留，两个评分条件并列。仅复用相同模型、相同预处理身份、相同规范化PCM哈希的缓存；不混用raw评分。不重新运行增强模型或训练，不创建音频副本全集，规范化可从原波形和冻结代码重建。\n')
    previous=common.OUT
    try:common.OUT=CONTROL;common.archive('next_full_room_decision',decision,[CONTROL/'FINAL_ACCEPTANCE_AUDIT.json'])
    finally:common.OUT=previous
    OUT.mkdir(exist_ok=False)
    config=json.loads((CONTROL/'frozen_plan.json').read_text())['normalization']
    assert sha(ROOT/'src/mcse/asr_level.py')==config['code_sha256']
    names=['scripts/g26_prepare.py','src/mcse/asr_level.py','artifacts/g22_matched_rooms_v1/evaluation/combined_signals.jsonl',
           'artifacts/g22_matched_rooms_v1/FINAL_ACCEPTANCE_AUDIT.json','artifacts/g25_asr_level_control_v1/frozen_plan.json',
           'artifacts/g25_asr_level_control_v1/FINAL_ACCEPTANCE_AUDIT.json']
    plan={'frozen_utc':now(),'normalization':config,'records_each_ASR':16488,'reference_cases':504,
          'max_data_seconds':1800,'training':False,'new_enhancement_inference':False,
          'scope':'Posthoc auxiliary ASR regime for all G22 outputs; raw quality and ASR unchanged',
          'cache_policy':'Only compatible G25 normalized observations and new exact normalized PCM hashes',
          'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G26A_完整房间统一电平辅评分协议.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G26A 完整房间统一电平辅评分协议\n\n对原G22全部16488输出和504个参考条件固定使用G25规则与同一预处理ID，模型、解码选项和增强参数不变。原音频、raw-ASR及质量指标不改。统一电平ASR另存为辅助评价条件，不选择更低分覆盖旧表。\n\n规范化在内存中重建，清单保存原文件SHA、规范化PCM SHA、预处理ID与增益诊断，不再复制整套音频。仅复用同一预处理的相同PCM输入；未缓存输出及参考必须解码。先数据/哈希报告归档，再执行固定两个识别器。\n\n分析保留所有三种子、方向、噪声、场型、干净控制和房间配对，特别检查主要方法排序和原有房间效应是否保留。后验辅评分不能称未见数据验证或新的增强算法。四人四句的统计限制不变。\n')
    archive('protocol',report,[OUT/'frozen_plan.json',ROOT/'scripts/g26_prepare.py',CONTROL/'FINAL_ACCEPTANCE_AUDIT.json'])
    start=time.monotonic();signals=read(SOURCE/'evaluation/combined_signals.jsonl')
    assert len(signals)==16488
    fields=('system','sample_id','source_id','speaker_id','transcript','condition','room_id','requested_rt60','target_azimuth',
            'noise_id','noise_field','relative_angle','matched_room_pair','frames','output_path','output_sha256',
            'reference_path','reference_sha256','oracle_clean_information','deployable')
    references={};reference_cases={};zero_pairs=defaultdict(dict);peak_limited=0;rows=[]
    with (OUT/'manifest.jsonl').open('x',encoding='utf-8') as log:
        for i,r in enumerate(signals,1):
            assert time.monotonic()-start<plan['max_data_seconds']
            original,key=waveform(r['output_path'],r['output_sha256'])
            normalized,info=standardize_level(original,config['target_rms'],config['peak_limit'])
            output_key=hashlib.sha256(normalized.tobytes()).hexdigest();peak_limited+=info['peak_limited']
            ref=r['reference_path']
            if ref not in references:
                a,_=waveform(ref,r['reference_sha256']);n,diag=standardize_level(a,config['target_rms'],config['peak_limit'])
                references[ref]=(hashlib.sha256(n.tobytes()).hexdigest(),diag,r['reference_sha256'])
            rk,ri,rh=references[ref];assert rh==r['reference_sha256']
            reference_cases[r['sample_id']]=rk
            row={field:r[field] for field in fields}
            row.update(raw_output_samples_sha256=key,normalized_output_samples_sha256=output_key,
                       normalized_reference_samples_sha256=rk,normalization_id=config['preprocessing_id'],
                       output_level_diagnostics=info,reference_level_diagnostics=ri,ASR_regime='level_control_v1')
            log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
            if r['requested_rt60']==0:zero_pairs[r['system'],r['matched_room_pair']][r['room_id']]=(output_key,rk)
            if i%1008==0:dump(OUT/'status.json',{'stage':'data','records':i,'total':16488,'utc':now()})
    assert len(reference_cases)==504 and len(zero_pairs)==2748
    assert all(set(v)=={'reference','new'} and v['reference']==v['new'] for v in zero_pairs.values())
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    dump(OUT/'reference_normalized_inputs.json',references,True)
    done={'status':'ok','utc':now(),'records':16488,'reference_cases':504,'RT0_exact_normalized_pairs':2748,
          'peak_limited_output_records':peak_limited,'manifest_sha256':sha(OUT/'manifest.jsonl'),
          'reference_inputs_sha256':sha(OUT/'reference_normalized_inputs.json'),'elapsed_seconds':time.monotonic()-start,
          'ASR_run':False,'new_audio_files':0}
    dump(OUT/'PREPARATION_COMPLETE.json',done,True)
    report=ROOT/'docs/G26B_完整规范化输入清单与负控制.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G26B 完整规范化输入清单与负控制\n\n16488个输出及504参考条件已从原波形重建规范化PCM哈希与增益诊断。2748对RT0的规范化输出/参考键完全一致。触发峰值上限的输出记录数为 '+str(peak_limited)+'；这些记录的RMS可能低于目标，仍按同一固定规则保留。\n\n没有生成新音频全集、运行增强网络或训练，尚未运行本阶段ASR。不根据输入指标删样本或改参数，后续按清单哈希重建并评分，原始音频和成绩保持不变。\n')
    archive('preparation',report,[OUT/'PREPARATION_COMPLETE.json',OUT/'manifest.jsonl',OUT/'reference_normalized_inputs.json',ROOT/'scripts/g26_prepare.py'])
    dump(OUT/'status.json',{'stage':'prepared_normalized_full_ASR_not_started','utc':now()});print(done)


if __name__=='__main__':main()
