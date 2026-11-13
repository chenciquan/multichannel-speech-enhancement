#!/usr/bin/env python3
import json
import numpy as np
from g16_common import ROOT,OUT,now,sha,dump,archive
from mcse.asr_scoring import corpus_score


def main():
    assert json.loads((OUT/'MAJOR_STAGE_COMPLETE.json').read_text())['status']=='ok';ev=OUT/'evaluation';s=json.loads((ev/'summary.json').read_text());audit=json.loads((OUT/'completion_audit.json').read_text());ready=json.loads((OUT/'PIPELINE_READY.json').read_text())
    checks={'completion_audit_ok':audit['status']=='ok','full98_runtime_tests_passed':'Ran 98 tests' in (OUT/'full98_runtime_tests.log').read_text() and '\nOK' in (OUT/'full98_runtime_tests.log').read_text(),
        'all_frozen_sources_match':all(sha(ROOT/n)==h for n,h in ready['files'].items()),'temporary_awake_released':json.loads((OUT/'power_helper.json').read_text())['status']=='released'}
    valid=True;n=0
    for p in (OUT/'stages').glob('*/archive_manifest.json'):
        for r in json.loads(p.read_text())['files']:valid&=sha(p.parent/r['snapshot'])==r['sha256'];n+=1
    checks['all_prior_snapshots_match']=bool(valid);final={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'snapshot_files_checked':n,'completed_utc':now()};assert final['status']=='ok'
    readback={'cells':{},'all_seed_contrasts':{},'resources_new_models_only':json.loads((ev/'runtime_resources.json').read_text()),'scope':'Development only,8reused sources4speakers;not fresh confirmation'}
    for g in s['clean']:
        readback['cells'][g]={}
        for t in (-40,20,70):
            v=[models[g][str(t)] for models in s['cells'].values()];readback['cells'][g][str(t)]={'si_sdr_db':float(np.mean([r['metrics']['si_sdr_db'] for r in v])),
                'Whisper_WER':corpus_score(r['Whisper'] for r in v)['wer'],'CTC_WER':corpus_score(r['CTC'] for r in v)['wer']}
    reg=json.loads((OUT/'systems.json').read_text());systems=reg['baseline']+reg['direct'];maps={geom:{r['seed']:r['system'] for r in systems if r['kind']=='adapter' and r.get('geometry')==geom} for geom in ('fixed','rotated','rigid')};assert all(len(v)==3 for v in maps.values())
    for seed in sorted(maps['rigid']):
        readback['all_seed_contrasts'][str(seed)]={}
        for comparison in ('fixed','rotated'):
            readback['all_seed_contrasts'][str(seed)][comparison]={}
            for suffix in ('','_mask_mvdr'):
                effects={}
                for t in (-40,20,70):
                    a=s['per_seed'][maps['rigid'][seed]+suffix][str(t)];b=s['per_seed'][maps[comparison][seed]+suffix][str(t)]
                    effects[str(t)]={'si_sdr_db':a['metrics']['si_sdr_db']-b['metrics']['si_sdr_db'],'Whisper_WER':a['Whisper']['wer']-b['Whisper']['wer'],'CTC_WER':a['CTC']['wer']-b['CTC']['wer']}
                readback['all_seed_contrasts'][str(seed)][comparison][suffix or 'direct']=effects
    lines=['# G16 开发诊断验收与三策略简明读回','','21系统×168=3528条信号及每后端3528评分全覆盖；其中复用2520条G14基线，仅新增1008条。全部基线记录不变、音频/权重SHA、双ASR重计分、掩码来源和阶段顺序通过；注册变更后的98项测试、三新模型验证指标复现、两旧模型波形复现、快照核验及临时保持唤醒释放完成。',
        '', '开发集8句/4人已经查看，不是新独立确认。fixed、counter、rigid都用同种子/采样/预算，rigid为本次正确刚性训练；旧counter继续保留作另一联合方向分布增强，不能仅因更正几何而断言其无效。下表合并两噪声、全部三种子，格内SI-SDR dB / Whisper WER / CTC WER。','','|系统|−40°|20°|70°|','|---|---:|---:|---:|']
    for g,v in readback['cells'].items():lines.append('|'+g+'|'+ '|'.join(f'{r["si_sdr_db"]:.2f} / {r["Whisper_WER"]:.1%} / {r["CTC_WER"]:.1%}' for r in v.values())+'|')
    lines+=['','## 新三个模型的整批资源','','|模型|Torch分配峰值GiB|Torch保留峰值GiB|','|---|---:|---:|']
    for r in readback['resources_new_models_only']:lines.append(f'|{r["system"]}|{r["peak_allocated_bytes"]/2**30:.3f}|{r["peak_reserved_bytes"]/2**30:.3f}|')
    lines+=['','该资源表不混入旧基线的历史耗时，不声称端到端实时。每种子的效应及所有不利结果另存readback/summary；一项WER下降不等于词语更完整，也不能从开发均值宣称普遍更优。',
        '', '下一阶段须把当前全部权重、预算与输出形式冻结，另立新语音确认协议；新说话人源、语句ID和源SHA排除项目已用数据，不按增强/识别成绩挑选。已查看语句不再次包装成最终验证。混响与真实硬件适用范围仍待研究，不预设新训练胜出或保证论文发表。',
        '', '[主报告](G16_三种方向训练策略开发诊断报告.md)、[机器读回](../artifacts/g16_rigid_dev_v1/evaluation/readback.json)、[完整审计](../artifacts/g16_rigid_dev_v1/completion_audit.json)。','', 'UTC：'+now()]
    report=ROOT/'docs/G16_开发诊断验收与三策略简明读回.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(ev/'readback.json',readback,True);dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',final,True);archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ev/'readback.json',ROOT/'scripts/g16_accept.py']);print(json.dumps(final))


if __name__=='__main__':main()
