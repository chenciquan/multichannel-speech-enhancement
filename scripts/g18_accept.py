#!/usr/bin/env python3
"""Final evidence readback, including actual decay and finite-tail diagnostics."""
import json
import numpy as np
from g18_common import ROOT,OUT,now,sha,dump,archive
from mcse.asr_scoring import corpus_score


def main():
    assert json.loads((OUT/'MAJOR_STAGE_COMPLETE.json').read_text())['status']=='ok';ev=OUT/'evaluation';s=json.loads((ev/'summary.json').read_text());audit=json.loads((OUT/'completion_audit.json').read_text());ready=json.loads((OUT/'PIPELINE_READY.json').read_text());plan=json.loads((OUT/'frozen_plan.json').read_text())
    asset=json.loads((ROOT/'artifacts/g12_room_stress_v2/ASSET_COMPLETE.json').read_text());checks={'completion_audit_ok':audit['status']=='ok','all_frozen_sources_match':all(sha(ROOT/n)==h for n,h in ready['files'].items()),
        'original_RIR_modules_unchanged':all(sha(p)==h for p,h in asset['installed_module_files_sha256'].items()),
        'shared_runtime_matches_G16QA':sha(ROOT/'src/mcse/runtime.py')==json.loads((ROOT/'artifacts/g16_rigid_dev_v1/RUNTIME_QA.json').read_text())['current_runtime_sha256'],
        'temporary_awake_released':json.loads((OUT/'power_helper.json').read_text())['status']=='released'}
    valid=True;n=0
    for p in (OUT/'stages').glob('*/archive_manifest.json'):
        for r in json.loads(p.read_text())['files']:valid&=sha(p.parent/r['snapshot'])==r['sha256'];n+=1
    checks['all_prior_snapshot_hashes_match']=bool(valid);final={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'snapshot_files_checked':n,'completed_utc':now()};assert final['status']=='ok'
    readback={'cells':{},'all_seed_rigid_minus_counter':{},'RIR_decay':{},'resources_all_neural_systems':json.loads((ev/'runtime_resources.json').read_text()),
        'scope':'One fixed simulated room,reused8utterances4speakers,nearfieldRT0control;not fresh voices or broad realroomconfirmation'}
    reg=json.loads((OUT/'systems.json').read_text());maps={g:{r['seed']:r['system'] for r in reg['direct'] if r.get('geometry')==g and r['kind']=='adapter'} for g in ('fixed','rotated','rigid')};assert all(len(v)==3 for v in maps.values())
    rirs=json.loads((OUT/'RIR_manifest.json').read_text())
    for rt in plan['requested_RT60_seconds']:
        rk=str(rt);readback['cells'][rk]={};readback['all_seed_rigid_minus_counter'][rk]={}
        for g in s['clean'][rk]:
            readback['cells'][rk][g]={}
            for t in (-40,20,70):
                a=[models[g][str(t)] for models in s['cells'][rk].values()];readback['cells'][rk][g][str(t)]={'si_sdr_db':float(np.mean([r['metrics']['si_sdr_db'] for r in a])),
                    'Whisper_WER':corpus_score(r['Whisper'] for r in a)['wer'],'CTC_WER':corpus_score(r['CTC'] for r in a)['wer']}
        for seed in sorted(maps['rigid']):
            readback['all_seed_rigid_minus_counter'][rk][str(seed)]={}
            for suffix in ('','_mask_mvdr'):
                effects={}
                for t in (-40,20,70):
                    a=s['per_seed'][rk][maps['rigid'][seed]+suffix][str(t)];b=s['per_seed'][rk][maps['rotated'][seed]+suffix][str(t)]
                    effects[str(t)]={'si_sdr_db':a['metrics']['si_sdr_db']-b['metrics']['si_sdr_db'],'Whisper_WER':a['Whisper']['wer']-b['Whisper']['wer'],'CTC_WER':a['CTC']['wer']-b['CTC']['wer']}
                readback['all_seed_rigid_minus_counter'][rk][str(seed)][suffix or 'direct']=effects
        a=[r for r in rirs if r['requested_rt60']==rt];values=[r['measured_T20_extrapolated_RT60'] for r in a if r['measured_T20_extrapolated_RT60'] is not None];assert len(a)==33
        readback['RIR_decay'][rk]={'positions':len(a),'valid_T20_fits':len(values),'T20_extrapolated_seconds_min_median_max':[float(np.min(values)),float(np.median(values)),float(np.max(values))] if values else None,
            'maximum_last20ms_energy_fraction':max(r['truncated_480ms_last20ms_energy_fraction'] for r in a),'notice':'Requested Sabine label,position-dependent T20 estimate,and finite-tail energy are different quantities.'}
    lines=['# G18 单房间压力验收与简明读回','','21系统×504=10584条信号及各后端10584条评分全覆盖，全部源/输入/输出/RIR/权重SHA、双ASR重计分、掩码来源和阶段顺序通过；原生成器模块、共享推理代码及阶段快照核验通过，临时保持唤醒已释放。没有新依赖、下载、训练或新语音。',
        '', '复用G17的8句/4人。固定房间内转动源星座，墙体不转；RT0是同距离近场无反射对照，不等同G17平面波，八源星座也不同。各RT共享完整0.48秒响应窗和总时长，因此RT0末端也保留背景噪声；这不同G17的短保护窗，跨阶段差值不能全归因反射。参考为混响语音，任务是降噪，未评估干声去混响。下表合并两旧录音/全部三种子，格内SI-SDR dB / Whisper WER / CTC WER；主报告分噪声，机器文件含每说话人/场型/种子及效应。','','|请求RT|系统|−40°|20°|70°|','|---:|---|---:|---:|---:|']
    for rt,models in readback['cells'].items():
        for g,v in models.items():lines.append(f'|{rt}|{g}|'+ '|'.join(f'{r["si_sdr_db"]:.2f} / {r["Whisper_WER"]:.1%} / {r["CTC_WER"]:.1%}' for r in v.values())+'|')
    lines+=['','## 请求参数、实际衰减估计与有限窗口','','|请求RT秒|源位置数|有效T20拟合数|T20外推秒最小/中位/最大|末20ms能量最大比例|','|---:|---:|---:|---|---:|']
    for rt,r in readback['RIR_decay'].items():lines.append(f'|{rt}|{r["positions"]}|{r["valid_T20_fits"]}|{r["T20_extrapolated_seconds_min_median_max"]}|{r["maximum_last20ms_energy_fraction"]:.8g}|')
    lines+=['','0.48秒是有限响应窗，保留全部有限卷积尾部不能保证自然无限衰减完整；T20外推也不是直接实测T60。SNR沿用混音的未量化参考活动帧，量化后同mask核验，另记录重算PCM活动帧差异，不改定义以匹配成绩。',
        '', '## 完整变长批次资源','','|模型|Torch分配峰值GiB|Torch保留峰值GiB|','|---|---:|---:|']
    for r in readback['resources_all_neural_systems']:lines.append(f'|{r["system"]}|{r["peak_allocated_bytes"]/2**30:.3f}|{r["peak_reserved_bytes"]/2**30:.3f}|')
    lines+=['','以上是逐系统FP32/batch1的Torch统计，不是整机峰值或流式时延。全部对比在同一输入下成组进行，但跨G17/G18改变了近场、星座和房间，不能将跨阶段差值全归因为反射。',
        '', '四位说话人、多次使用的句子、两旧噪声和单间模拟房不足以支持真实/广泛房间泛化。WER下降不直接证明辅音/词尾完整；当前无新人工逐词或盲听标签。所有负结果与三种子保留，不预设正确训练胜出。下一阶段根据全体结果收窄问题，不凭均值继续增加模块。',
        '', '[主报告](G18_固定房间反射压力测试报告.md)、[机器读回](../artifacts/g18_room_stress_v1/evaluation/readback.json)、[完整审计](../artifacts/g18_room_stress_v1/completion_audit.json)。','', 'UTC：'+now()]
    report=ROOT/'docs/G18_单房间压力验收与简明读回.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(ev/'readback.json',readback,True);dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',final,True);archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ev/'readback.json',ROOT/'scripts/g18_accept.py']);print(json.dumps(final))


if __name__=='__main__':main()
