#!/usr/bin/env python3
import json
import numpy as np
from g17_common import ROOT,OUT,now,sha,dump,archive
from mcse.asr_scoring import corpus_score


def main():
    assert json.loads((OUT/'MAJOR_STAGE_COMPLETE.json').read_text())['status']=='ok';ev=OUT/'evaluation';s=json.loads((ev/'summary.json').read_text());audit=json.loads((OUT/'completion_audit.json').read_text());ready=json.loads((OUT/'PIPELINE_READY.json').read_text())
    checks={'completion_audit_ok':audit['status']=='ok','unchanged_shared_runtime_since_G16QA':sha(ROOT/'src/mcse/runtime.py')==json.loads((ROOT/'artifacts/g16_rigid_dev_v1/RUNTIME_QA.json').read_text())['current_runtime_sha256'],
        'all_frozen_sources_match':all(sha(ROOT/n)==h for n,h in ready['files'].items()),'temporary_awake_released':json.loads((OUT/'power_helper.json').read_text())['status']=='released'}
    valid=True;n=0
    for p in (OUT/'stages').glob('*/archive_manifest.json'):
        for r in json.loads(p.read_text())['files']:valid&=sha(p.parent/r['snapshot'])==r['sha256'];n+=1
    checks['all_prior_snapshots_match']=bool(valid);final={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'snapshot_files_checked':n,'completed_utc':now()};assert final['status']=='ok'
    readback={'cells':{},'all_seed_contrasts':{},'resources_all_neural_models':json.loads((ev/'runtime_resources.json').read_text()),'scope':'Fresh8utterances4project speakers;old2noiserecordings;correct nominalfreefield;not broadrealroomconfirmation'}
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
    lines=['# G17 新说话人确认验收与三策略简明读回','','21系统×168=3528条全部本轮生成的信号及每后端3528评分全覆盖。新说话人/源ID/源SHA排除、音频/权重SHA、双ASR重计分、掩码来源、方法冻结及阶段顺序通过；沿用未变更共享数值代码的G16注册QA/98项测试，快照核验及临时保持唤醒释放完成。',
        '', '本轮4位说话人/8句在取数前排除了本项目全部已用说话人和源，全部21方法与步数先冻结；8句/4人仍不是数千独立样本，噪声录音并非新来源。fixed、counter、rigid使用同种子/采样/预算，rigid为正确刚性训练，counter为另一联合方向分布增强，不预设谁优胜。下表合并两噪声、全部三种子，格内SI-SDR dB / Whisper WER / CTC WER。','','|系统|−40°|20°|70°|','|---|---:|---:|---:|']
    for g,v in readback['cells'].items():lines.append('|'+g+'|'+ '|'.join(f'{r["si_sdr_db"]:.2f} / {r["Whisper_WER"]:.1%} / {r["CTC_WER"]:.1%}' for r in v.values())+'|')
    lines+=['','## 全部十个神经系统的整批资源','','|模型|Torch分配峰值GiB|Torch保留峰值GiB|','|---|---:|---:|']
    for r in readback['resources_all_neural_models']:lines.append(f'|{r["system"]}|{r["peak_allocated_bytes"]/2**30:.3f}|{r["peak_reserved_bytes"]/2**30:.3f}|')
    lines+=['','全部本轮神经系统实测，不声称端到端实时。每种子的效应及所有不利结果另存readback/summary；一项WER下降不等于词语更完整，也不能从四人短语音均值宣称普遍更优。',
        '', '本轮查看后也成为已查看数据；未来如调参，不能继续称全新最终测试。基础识别器对公开语料的预训练曝光未独立审计。下一项高价值风险是混响压力而非追加恢复模块；需在固定房间内清楚区分源相对几何、近场传播及反射变化。真实硬件/中文/广泛说话人范围仍待研究，不保证论文发表。',
        '', '[主报告](G17_三种方向训练策略新说话人确认报告.md)、[机器读回](../artifacts/g17_fresh_confirmation_v1/evaluation/readback.json)、[完整审计](../artifacts/g17_fresh_confirmation_v1/completion_audit.json)。','', 'UTC：'+now()]
    report=ROOT/'docs/G17_新说话人确认验收与三策略简明读回.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(ev/'readback.json',readback,True);dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',final,True);archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ev/'readback.json',ROOT/'scripts/g17_accept.py']);print(json.dumps(final))


if __name__=='__main__':main()
