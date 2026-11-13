#!/usr/bin/env python3
"""Read back paired effects, run the final98 suite once, archive acceptance."""
import json,os,subprocess,sys
import numpy as np
from g14_common import ROOT,OUT,now,sha,dump,archive
from mcse.asr_scoring import corpus_score


def main():
    assert json.loads((OUT/'MAJOR_STAGE_COMPLETE.json').read_text())['status']=='ok';ev=OUT/'evaluation';summary=json.loads((ev/'summary.json').read_text());audit=json.loads((OUT/'completion_audit.json').read_text());ready=json.loads((OUT/'PIPELINE_READY.json').read_text())
    assert json.loads((OUT/'power_helper.json').read_text())['status']=='released'
    log=OUT/'full98_unit_tests.log'
    with log.open('x') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},stdout=f,stderr=subprocess.STDOUT,timeout=300,check=True)
    checks={'completion_audit_ok':audit['status']=='ok','full98_tests_passed':'Ran 98 tests' in log.read_text() and '\nOK' in log.read_text(),
        'all_frozen_pipeline_sources_unchanged':all(sha(ROOT/n)==h for n,h in ready['files'].items()),'temporary_awake_released':True}
    sources=json.loads((OUT/'selected_sources.json').read_text());prep=json.loads((OUT/'PREPARATION_COMPLETE.json').read_text());plan=json.loads((OUT/'frozen_plan.json').read_text())
    checks.update(all_eight_source_FLAC_hashes_match=all(sha(ROOT/'data/g10_v1/sources'/r['path'])==r['sha256'] for r in sources),
        data_manifest_hash_matches_preparation=sha(ROOT/'data/g14_v1/manifest.jsonl')==prep['manifest_sha256'],
        beamforming_source_unchanged=sha(ROOT/'src/mcse/beamforming.py')==plan['beamforming_sha256'],
        frozen_G10_manifest_unchanged=sha(ROOT/'data/g10_v1/manifest.jsonl')==plan['G10_manifest_sha256'])
    oldregistry=ROOT/'artifacts/g10_fresh_voice_v1/systems.json';entries=json.loads((ROOT/'artifacts/g10_fresh_voice_v1/stages/preparation/archive_manifest.json').read_text())['files'];entry=next(r for r in entries if r['source']==str(oldregistry.relative_to(ROOT)))
    checks['all15_systems_match_archived_G10_registry']=sha(oldregistry)==sha(OUT/'systems.json')==entry['sha256']
    valid=True;n=0
    for p in (OUT/'stages').glob('*/archive_manifest.json'):
        for r in json.loads(p.read_text())['files']:valid&=sha(p.parent/r['snapshot'])==r['sha256'];n+=1
    checks['all_prior_snapshots_match']=bool(valid)
    final={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'snapshot_files_checked':n,'completed_utc':now()};assert final['status']=='ok'
    reg=json.loads((OUT/'systems.json').read_text());readback={'cells':{},'all_seed_legacy_training_contrasts':{},'negative_controls':summary['negative_controls'],'resources':json.loads((ev/'runtime_resources.json').read_text())}
    groups=('mcnet','fixed','legacy_rotated','mcnet_mvdr','fixed_mvdr','legacy_rotated_mvdr')
    for protocol,noises in summary['cells'].items():
        readback['cells'][protocol]={}
        for g in groups:
            readback['cells'][protocol][g]={}
            for t in (-40,20,70):
                rows=[models[g][str(t)] for models in noises.values()];readback['cells'][protocol][g][str(t)]={'si_sdr_db':float(np.mean([r['metrics']['si_sdr_db'] for r in rows])),
                   'Whisper_WER':corpus_score(r['Whisper'] for r in rows)['wer'],'CTC_WER':corpus_score(r['CTC'] for r in rows)['wer']}
        fixed={r['seed']:r['system'] for r in reg['direct'] if r.get('geometry')=='fixed'};rot={r['seed']:r['system'] for r in reg['direct'] if r.get('geometry')=='rotated'};assert len(fixed)==3 and set(fixed)==set(rot)
        readback['all_seed_legacy_training_contrasts'][protocol]={}
        for seed in sorted(fixed):
            values={}
            for suffix in ('','_mask_mvdr'):
                values[suffix or 'direct']={}
                for t in (-40,20,70):
                    a=summary['per_seed'][protocol][rot[seed]+suffix][str(t)];b=summary['per_seed'][protocol][fixed[seed]+suffix][str(t)]
                    values[suffix or 'direct'][str(t)]={'si_sdr_db':a['metrics']['si_sdr_db']-b['metrics']['si_sdr_db'],'Whisper_WER':a['Whisper']['wer']-b['Whisper']['wer'],'CTC_WER':a['CTC']['wer']-b['CTC']['wer']}
            readback['all_seed_legacy_training_contrasts'][protocol][str(seed)]=values
    lines=['# G14 阶段验收与正确几何简明读回','','5040条信号及每识别器5040条评分覆盖、全部源/音频/权重SHA、双ASR重计分和阶段顺序通过；98项完整测试通过，原快照核验与临时保持唤醒释放完成。',
        '', '这次只纠正测试几何，冻结已有模型。legacy_rotated模型实际接受旧反向旋转训练，不能当作正确整场景旋转重训。8句/4人及两旧录音已查看，本轮不是新独立最终验证。',
        '', '下表合并两录音、全部三种子，仅便于读回；每格SI-SDR dB / Whisper WER / CTC WER。主报告分噪声，机器文件还分说话人、场型和每种子。','','|测试协议|系统|−40°|20°|70°|','|---|---|---:|---:|---:|']
    for protocol,models in readback['cells'].items():
        for g,values in models.items():lines.append(f'|{protocol}|{g}|'+ '|'.join(f'{v["si_sdr_db"]:.2f} / {v["Whisper_WER"]:.1%} / {v["CTC_WER"]:.1%}' for v in values.values())+'|')
    nc=summary['negative_controls'];lines+=['',f'1080对协议不变负控制：最大波形差{nc["maximum_abs_difference"]:.3g}，完全相同{nc["exact_waveforms"]}对，任一后端假设不同{nc["any_backend_hypothesis_difference_pairs"]}对。',
      '', '测试场景难度变化不是算法改进收益；旧训练策略差异不能自动解释为正确刚性旋转训练。无新人工逐词标签或稳定吞字证据。后续若训练正确旋转适配器，须固定步骤/采样/损失/验证规则，并与旧固定和反向训练明确比较，另用未查看语音确认。',
      '', '## 完整变长批次资源','','|模型|Torch分配峰值GiB|Torch保留峰值GiB|','|---|---:|---:|---:|']
    for r in readback['resources']:lines.append(f'|{r["system"]}|{r["peak_allocated_bytes"]/1024**3:.3f}|{r["peak_reserved_bytes"]/1024**3:.3f}|')
    lines+=['','以上是逐系统FP32/batch1的Torch统计，不是整机峰值或流式时延。每系统释放缓存，原模型、驱动/CUDA均未改。',
      '', '[主报告](G14_正确几何配对自由场诊断报告.md)、[更正入口](G13A_旋转坐标审计与历史结论更正.md)、[机器读回](../artifacts/g14_corrected_geometry_v1/evaluation/readback.json)。','', 'UTC：'+now()]
    report=ROOT/'docs/G14_阶段验收与正确几何简明读回.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(ev/'readback.json',readback,True);dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',final,True)
    archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',ev/'readback.json',log,ROOT/'scripts/g14_accept.py']);print(json.dumps(final))


if __name__=='__main__':main()
