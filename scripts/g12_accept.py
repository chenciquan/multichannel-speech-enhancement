#!/usr/bin/env python3
"""Read completed G12 evidence; do not repeat neural inference or tests."""
import json
import numpy as np
from g12_common import ROOT,OUT,now,sha,dump,archive
from mcse.asr_scoring import corpus_score


def main():
    assert json.loads((OUT/'MAJOR_STAGE_COMPLETE.json').read_text())['status']=='ok'
    audit=json.loads((OUT/'completion_audit.json').read_text());summary=json.loads((OUT/'evaluation/summary.json').read_text());ready=json.loads((OUT/'PIPELINE_READY.json').read_text())
    check={'completion_audit_ok':audit['status']=='ok','whole91_test_suite_passed':'Ran 91 tests' in (OUT/'reverberation_unit_tests.log').read_text() and '\nOK' in (OUT/'reverberation_unit_tests.log').read_text(),
        'all_pipeline_sources_unchanged':all(sha(ROOT/name)==h for name,h in ready['files'].items()),
        'temporary_awake_released':json.loads((OUT/'power_helper.json').read_text())['status']=='released'}
    count=0;valid=True
    for manifest in (OUT/'stages').glob('*/archive_manifest.json'):
        for r in json.loads(manifest.read_text())['files']:valid&=sha(manifest.parent/r['snapshot'])==r['sha256'];count+=1
    check['all_prior_G12_stage_snapshot_hashes_match']=bool(valid)
    asset=json.loads((OUT/'ASSET_COMPLETE.json').read_text())
    check['installed_RIR_files_unchanged']=all(sha(name)==h for name,h in asset['installed_module_files_sha256'].items())
    reg=json.loads((OUT/'systems.json').read_text());fixed=[r['system'] for r in reg['direct'] if r.get('geometry')=='fixed'];rotated=[r['system'] for r in reg['direct'] if r.get('geometry')=='rotated']
    checks={'status':'ok' if all(check.values()) else 'failed','checks':check,'snapshot_files_checked':count,'completed_utc':now()}
    if checks['status']!='ok':raise RuntimeError(json.dumps(checks))
    def aggregate(rt,g,t):
        rows=[models[g][str(t)] for models in summary['cells'][str(rt)].values()]
        return {'si_sdr_db':float(np.mean([r['metrics']['si_sdr_db'] for r in rows])),
            'Whisper_WER':corpus_score(r['Whisper'] for r in rows)['wer'],'CTC_WER':corpus_score(r['CTC'] for r in rows)['wer']}
    readback={'protocol':'Aggregate both recorded noises,all three adapter seeds; not new independent sentences','conditions':{},'all_seed_effects':{},'resources':json.loads((OUT/'evaluation/runtime_resources.json').read_text())}
    lines=['# G12 阶段验收与混响压力简明读回','','7560条信号及每后端7560条评分全覆盖，全部三种子、源/音频/RIR/权重SHA及双ASR重计分通过；91项测试已在房间数据生成前完成，验收不重跑推理/识别。所有阶段快照与临时保持唤醒释放核验通过。',
        '', '复用8句/4人、单房间、两条旧噪声录音，参考是混响语音。请求RT不是每位置实测T60；有限0.48秒RIR尾部记录另存。不是新独立语音、多房间或真实硬件验证，也未确认稳定吞字。',
        '', '以下合并两噪声，仅作读回；主报告分噪声、机器汇总还分说话人、场型、每种子。格内SI-SDR dB / Whisper WER / CTC WER。','','|请求RT|系统|−40°|20°|70°|','|---:|---|---:|---:|---:|']
    for rt in (0,.15,.35):
        readback['conditions'][str(rt)]={}
        for g in ('mcnet','fixed','rotated','mcnet_mvdr','fixed_mvdr','rotated_mvdr'):
            values={str(t):aggregate(rt,g,t) for t in (-40,20,70)};readback['conditions'][str(rt)][g]=values
            lines.append(f'|{rt}|{g}|'+ '|'.join(f'{v["si_sdr_db"]:.2f} / {v["Whisper_WER"]:.1%} / {v["CTC_WER"]:.1%}' for v in values.values())+'|')
        readback['all_seed_effects'][str(rt)]={}
        fixed_by_seed={s['seed']:s['system'] for s in reg['direct'] if s['system'] in fixed};rotated_by_seed={s['seed']:s['system'] for s in reg['direct'] if s['system'] in rotated}
        assert len(fixed_by_seed)==3 and set(fixed_by_seed)==set(rotated_by_seed)
        for seed in sorted(fixed_by_seed):
            f,r=fixed_by_seed[seed],rotated_by_seed[seed];se=str(seed);readback['all_seed_effects'][str(rt)][se]={}
            for suffix in ('','_mask_mvdr'):
                readback['all_seed_effects'][str(rt)][se][suffix or 'direct']={}
                for t in (-40,20,70):
                    def one(name):
                        a=summary['per_seed'][str(rt)][name][str(t)]
                        return {'si_sdr_db':a['metrics']['si_sdr_db'],'Whisper_WER':a['Whisper']['wer'],'CTC_WER':a['CTC']['wer']}
                    left,right=one(r+suffix),one(f+suffix);readback['all_seed_effects'][str(rt)][se][suffix or 'direct'][str(t)]={k:left[k]-right[k] for k in left}
    lines+=['','## 资源与解释边界','','每系统在完整变长504例运行期间记录Torch峰值，切换模型前释放缓存；下表不是整机显存和流式时延。','','|系统|Torch分配峰值GiB|Torch保留峰值GiB|','|---|---:|---:|']
    for r in readback['resources']:lines.append(f'|{r["system"]}|{r["peak_allocated_bytes"]/1024**3:.3f}|{r["peak_reserved_bytes"]/1024**3:.3f}|')
    lines+=['','混响和近场传播同时区别于旧自由场数据，无反射条件用于本轮内部控制，不能直接把与G10的差值全称为混响影响。指标一致变化只提供条件证据，不等于词尾或辅音逐个保真。下一阶段必须基于完整负结果收窄问题，不能先设恢复模块或宣称已有组件组合具有创新。',
        '', '[主报告](G12_单房间混响压力测试报告.md)、[机器读回](../artifacts/g12_room_stress_v2/evaluation/readback.json)、[完整审计](../artifacts/g12_room_stress_v2/completion_audit.json)。','', 'UTC：'+now()]
    report=ROOT/'docs/G12_阶段验收与混响压力简明读回.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(OUT/'evaluation/readback.json',readback,True);dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',checks,True)
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',OUT/'evaluation/readback.json',OUT/'reverberation_unit_tests.log',ROOT/'scripts/g12_accept.py']);print(json.dumps(checks))


if __name__=='__main__':main()
