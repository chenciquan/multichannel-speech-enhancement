#!/usr/bin/env python3
"""ASR-only level control and exact input-invariance validation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib
import json
import subprocess
import sys
import numpy as np
import soundfile as sf
import g8_common as common
from g19_scores import waveform
from mcse.asr_level import standardize_level

ROOT=common.ROOT;OUT=ROOT/'artifacts/g25_asr_level_control_v1';PARENT=ROOT/'artifacts/g24_gain_repeat_v1'
sha,read,dump,now=common.sha,common.read,common.dump,common.now


def archive(stage,report,files):
    old=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=old


def main():
    assert json.loads((PARENT/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    assert (PARENT/'stages/scientific_readback/archive_manifest.json').exists()
    OUT.mkdir(exist_ok=False);(OUT/'audio').mkdir()
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'src')
    with (OUT/'normalization_unit_tests.log').open('x') as log:
        test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_asr_level.py','-v'],
                            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    assert test.returncode==0,'Level-control tests failed; preserve log'
    config={'target_rms':.05,'peak_limit':.95,'formula':'min(target_rms/RMS(x),peak_limit/maxabs(x)); zero remains zero',
            'precision':'float32 input -> float64 coefficient and multiply -> float32 output',
            'code_sha256':sha(ROOT/'src/mcse/asr_level.py')}
    config['preprocessing_id']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    names=['scripts/g25_prepare.py','src/mcse/asr_level.py','tests/test_asr_level.py',
           'artifacts/g24_gain_repeat_v1/manifest.jsonl','artifacts/g24_gain_repeat_v1/FINAL_ACCEPTANCE_AUDIT.json']
    plan={'frozen_utc':now(),'base_pairs':480,'normalized_waveforms':480,'expected_observations_each_ASR':960,
          'normalization':config,'gain_variants_checked_per_base':4,'observations':['normalized_first','normalized_second'],
          'fresh_ASR_every_observation':True,'apply_to_enhancement_outputs':False,'raw_results_preserved':True,
          'selection':'Exactly all G24 base pairs, no new sample selection or performance-based normalization parameters',
          'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'frozen_plan.json',plan,True)
    protocol=ROOT/'docs/G25A_识别输入电平控制协议.md'
    with protocol.open('x',encoding='utf-8') as f:
        f.write('# G25A 识别输入电平控制协议\n\n预先固定目标RMS=.05、峰值上限=.95，取两个正比例系数的较小值，无削波。参数是本次评估控制约定，没有根据WER选择。只改变送入ASR的副本，增强模型、原音频、原ASR和RMS/NMSE指标均保留。\n\n保留G24全部480基准。先逐一验证原幅度/半幅/四分之一幅/重复原幅度经标准化后的float32完全相同；每个基准只保存一个规范化输入，并计划两遍独立识别，共960观察/后端，不把相同输入别名计为新的独立句子。预处理ID和参数另存，不与raw-ASR成绩混并。\n\n4项数学/边界测试先运行并保留日志。控制只检验全局电平混杂，不能修复内容、改变信噪比或证明增强方法更好；不作最优电平、部署或人耳可懂度结论。\n')
    archive('protocol',protocol,[OUT/'frozen_plan.json',OUT/'normalization_unit_tests.log',ROOT/'src/mcse/asr_level.py',ROOT/'tests/test_asr_level.py'])
    previous=read(PARENT/'manifest.jsonl');assert len(previous)==1920
    rows=[];invariance=[];cases=set()
    fields=('base_system','sample_id','source_id','speaker_id','transcript','room_id','requested_rt60','target_azimuth',
            'noise_id','noise_field','relative_angle','reference_path','reference_sha256','condition','frames','oracle_clean_information')
    with (OUT/'manifest.jsonl').open('x',encoding='utf-8') as log:
        for offset in range(0,len(previous),4):
            block=previous[offset:offset+4]
            assert [r['observation'] for r in block]==['original_first','half','quarter','original_second']
            normalized=[];original=None;info=None
            for i,r in enumerate(block):
                a,key=waveform(r['output_path'],r['output_sha256']);assert key==r['output_samples_sha256']
                n,details=standardize_level(a,.05,.95);normalized.append(n)
                if i==0:original=a;info=details
            assert all(n.tobytes()==normalized[0].tobytes() for n in normalized)
            base=block[0];canonical=normalized[0];path=OUT/'audio'/f"{base['base_system']}_{base['sample_id']}.wav"
            with path.open('xb') as f:sf.write(f,canonical,16000,format='WAV',subtype='FLOAT')
            recovered,key=waveform(str(path),sha(path));assert recovered.tobytes()==canonical.tobytes()
            invariance.append({'base_system':base['base_system'],'sample_id':base['sample_id'],
                               'variants_exact':4,'normalized_samples_sha256':key,'level_diagnostics':info})
            for observation in plan['observations']:
                record={field:base[field] for field in fields}
                record.update(observation=observation,raw_source_path=base['output_path'],raw_source_sha256=base['output_sha256'],
                              raw_source_samples_sha256=base['output_samples_sha256'],output_path=str(path),output_sha256=sha(path),
                              output_samples_sha256=key,normalization_id=config['preprocessing_id'],
                              level_diagnostics=info,ASR_regime='level_control_v1',control_only=True)
                log.write(json.dumps(record,allow_nan=False)+'\n');rows.append(record)
            cases.add(base['sample_id'])
    assert len(rows)==960 and len(invariance)==480 and len(cases)==48
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    dump(OUT/'input_invariance.json',invariance,True)
    done={'status':'ok','utc':now(),'base_pairs':480,'manifest_rows':960,'gain_variant_inputs_checked':1920,
          'all_normalized_binary_variants_exact':True,'unit_tests':4,
          'unique_normalized_samples':len({r['normalized_samples_sha256'] for r in invariance}),
          'peak_limited_bases':sum(r['level_diagnostics']['peak_limited'] for r in invariance),
          'manifest_sha256':sha(OUT/'manifest.jsonl'),'input_invariance_sha256':sha(OUT/'input_invariance.json'),'ASR_run':False}
    dump(OUT/'PREPARATION_COMPLETE.json',done,True)
    report=ROOT/'docs/G25B_识别电平输入不变性验证.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G25B 识别电平输入不变性验证\n\n4项测试通过：目标RMS、峰值约束、二进制增益不变性、零/无效输入边界。全部480基准的四种G24增益观察，经固定标准化后逐样本和字节完全一致；480条规范化副本已保存，960个独立重复识别观察待执行。\n\n本验证证明输入级的全局增益控制，不是增强模型改善。尚未运行本阶段ASR，因此不提前报告识别稳定性或WER变化。原增强输出与raw-ASR未改。\n')
    archive('preparation',report,[OUT/'PREPARATION_COMPLETE.json',OUT/'input_invariance.json',OUT/'manifest.jsonl',ROOT/'scripts/g25_prepare.py'])
    dump(OUT/'status.json',{'stage':'prepared_normalized_ASR_not_started','utc':now()});print(done)


if __name__=='__main__':main()
