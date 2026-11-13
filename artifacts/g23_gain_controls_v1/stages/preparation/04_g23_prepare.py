#!/usr/bin/env python3
"""Frozen positive scalar controls; no enhancement inference or optimization."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import g8_common as common
from mcse.evaluation import si_sdr

ROOT=common.ROOT;OUT=ROOT/'artifacts/g23_gain_controls_v1';G22=ROOT/'artifacts/g22_matched_rooms_v1'
sha,dump,read,now=common.sha,common.dump,common.read,common.now


def archive(stage,report,files):
    old=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=old


def main():
    assert json.loads((G22/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    decision=ROOT/'docs/G22_验收后增益控制阶段决策.md'
    with decision.open('x',encoding='utf-8') as f:
        f.write('# G22 验收后的增益控制阶段决策\n\nG22全部信号、双ASR与房间配对已验收。较强反射的新房间中，正确刚性组MWF与MVDR的Whisper插入计数为1086与171；文本错误排序不能直接称为声音内容丢失。\n\n下一阶段仅做固定正标量增益控制，保留归一化数字波形的相对结构，检验ASR评分对幅度的敏感性。使用预先固定−6/−12dB及0dB原输出，不根据成绩找最优增益。不训练、不重新增强、不加入TSOS成绩；它不是新增强方法或听感不变性证明。\n')
    old=common.OUT
    try:common.OUT=G22;common.archive('next_gain_control_decision',decision,[G22/'FINAL_ACCEPTANCE_AUDIT.json'])
    finally:common.OUT=old
    OUT.mkdir(exist_ok=False);(OUT/'gain_audio').mkdir()
    cases=[c for c in read(ROOT/'data/g22_v1/manifest.jsonl') if c['noise_id']=='OOFFICE' and c['noise_field']=='point'
           and c['relative_angle']==-80 and c['requested_rt60'] in (.15,.35)]
    assert len(cases)==48 and len({c['source_id'] for c in cases})==4
    registry=json.loads((G22/'signal_systems.json').read_text())
    rigid=[s['system'] for s in registry['direct'] if s.get('geometry')=='rigid'];assert len(rigid)==3
    systems=['noisy']+[s+'_mask_'+m for s in rigid for m in ('mvdr','mwf')]+['truth_covariance_mvdr','truth_covariance_mwf']
    pins=['scripts/g23_prepare.py','artifacts/g22_matched_rooms_v1/FINAL_ACCEPTANCE_AUDIT.json',
          'artifacts/g22_matched_rooms_v1/evaluation/combined_signals.jsonl','data/g22_v1/manifest.jsonl',
          'artifacts/g22_matched_rooms_v1/signal_systems.json','src/mcse/evaluation.py']
    plan={'frozen_utc':now(),'selection':'OOFFICE point relative -80, all3angles, RT.15/.35, bothrooms, all4sources',
          'cases':48,'base_systems':systems+['clean_reference'],'gain_dB':[0,-6,-12],
          'base_pairs':480,'new_waveforms':960,'expected_pairs_each_ASR':1440,
          'relative_roundtrip_error_max':1e-6,'SI_SDR_change_tolerance_db':.001,
          'exact_reference_SI_SDR':'Degenerate comparison; exclude clean_reference from SI-SDR invariance checks, retain all ASR cases',
          'gain_convention':'float64 multiplication by 10**(dB/20), cast float32; no clipping or per-file normalization',
          'purpose':'Exploratory evaluator control; no best gain selection, no model claim or human intelligibility invariance',
          'hashes':{n:sha(ROOT/n) for n in pins}}
    dump(OUT/'frozen_plan.json',plan,True)
    protocol=ROOT/'docs/G23A_识别器增益敏感性控制协议.md'
    with protocol.open('x',encoding='utf-8') as f:
        f.write('# G23A 识别器增益敏感性控制协议\n\n固定48条件：四源语音、两房间、请求RT=.15/.35、全部三个方向、OOFFICE点源相对−80。保留干净参考、带噪参考麦、正确刚性全部三种子的MVDR/MWF及两个真值诊断，共10种输出。选取基于场景标签，不按识别错误逐例挑选。\n\n0dB复用原文件，−6/−12dB仅对整段乘固定正标量，存FLOAT WAV，不削波、不修复或重新推理。共480基准对，960新音频，1440评分/后端。浮点缩放往返相对误差≤1e−6；非干净参考SI-SDR差≤0.001dB。干净参考的理想SI-SDR比较退化，单列并只核对波形结构，所有ASR仍保留。\n\n后续沿用固定双识别器，以同模型/PCM哈希复用缓存，0dB应全部命中已有缓存。比较S/D/I与假设文本变化，不根据结果选最佳增益。实验是评估器诊断，不是增强创新；降低播放电平可能影响人耳可听度，不声称听感不变。未实现或复现TSOS。\n')
    archive('protocol',protocol,[OUT/'frozen_plan.json',ROOT/'scripts/g23_prepare.py'])
    signals={(r['system'],r['sample_id']):r for r in read(G22/'evaluation/combined_signals.jsonl')}
    rows=[];round_errors=[];si_errors=[]
    with (OUT/'manifest.jsonl').open('x',encoding='utf-8') as log:
        for c in cases:
            ref,sr=sf.read(c['reference_path'],dtype='float32');assert sr==16000 and sha(c['reference_path'])==c['reference_sha256']
            for system in systems+['clean_reference']:
                if system=='clean_reference':path=c['reference_path'];digest=c['reference_sha256']
                else:path=signals[system,c['sample_id']]['output_path'];digest=signals[system,c['sample_id']]['output_sha256']
                assert sha(path)==digest
                original,rate=sf.read(path,dtype='float32');assert rate==16000 and original.shape==ref.shape and np.isfinite(original).all()
                baseline_si=None if system=='clean_reference' else si_sdr(ref,original)
                energy=float(original.astype(float)@original.astype(float));assert energy>0
                for db in plan['gain_dB']:
                    gain=10**(db/20);result=(original.astype(float)*gain).astype('float32')
                    relative=float(np.linalg.norm(result.astype(float)/gain-original)/np.sqrt(energy))
                    assert relative<=plan['relative_roundtrip_error_max'];round_errors.append(relative)
                    delta=None if baseline_si is None else si_sdr(ref,result)-baseline_si
                    if delta is not None:assert abs(delta)<=plan['SI_SDR_change_tolerance_db'];si_errors.append(abs(delta))
                    if db==0:output=Path(path);assert np.array_equal(result,original)
                    else:
                        output=OUT/'gain_audio'/f'{system}_{c["sample_id"]}_m{abs(db)}.wav'
                        with output.open('xb') as f:sf.write(f,result,16000,format='WAV',subtype='FLOAT')
                        saved,rr=sf.read(output,dtype='float32');assert rr==16000 and np.array_equal(saved,result)
                    row={**c,'base_system':system,'gain_db':db,'gain':gain,'base_path':str(path),'base_sha256':digest,
                         'output_path':str(output),'output_sha256':sha(output),'relative_roundtrip_error':relative,
                         'si_sdr_change_db':delta,'base_si_sdr_db':baseline_si,
                         'base_peak':float(np.abs(original).max()),'scaled_peak':float(np.abs(result).max()),
                         'oracle_clean_information':system.startswith('truth_covariance_'),'control_only':True}
                    log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
    assert len(rows)==len({(r['base_system'],r['sample_id'],r['gain_db']) for r in rows})==1440
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    done={'status':'ok','utc':now(),'rows':1440,'base_pairs':480,'new_waveforms':960,
          'max_relative_roundtrip_error':max(round_errors),'max_SI_SDR_change_db':max(si_errors),
          'manifest_sha256':sha(OUT/'manifest.jsonl'),'ASR_run':False}
    dump(OUT/'PREPARATION_COMPLETE.json',done,True)
    report=ROOT/'docs/G23B_固定增益波形控制验证.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G23B 固定增益波形控制验证\n\n生成960条缩放音频并复用480个0dB基准，共1440条件。正标量缩放往返相对误差最大 '+f"{done['max_relative_roundtrip_error']:.3g}"+'；非退化比较的SI-SDR变化最大 '+f"{done['max_SI_SDR_change_db']:.3g}"+'dB，均通过事先上限。干净参考保留，但不以其退化SI-SDR比较误判结构改变。\n\n本阶段未运行ASR、增强网络或训练，尚无增益影响识别的实验结论。所有原始G22文件保持原样。\n')
    archive('preparation',report,[OUT/'PREPARATION_COMPLETE.json',OUT/'manifest.jsonl',OUT/'frozen_plan.json',ROOT/'scripts/g23_prepare.py'])
    dump(OUT/'status.json',{'stage':'prepared_ASR_not_started','utc':now()})
    print(done)


if __name__=='__main__':main()
