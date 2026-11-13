"""Freeze the room comparison and verify the filter on saved frontend outputs."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import g18_common
from g19_common import ROOT, OUT, PREV, now, sha, read, dump, status, archive
from mcse.wiener import mask_mwf


def metadata(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stage_decision():
    decision = PREV / 'stages/next_mwf_decision/archive_manifest.json'
    if decision.exists():
        return
    assert metadata(PREV / 'FINAL_ACCEPTANCE_AUDIT.json')['status'] == 'ok'
    assert (PREV / 'stages/ASR_diagnostics/archive_manifest.json').is_file()
    report = ROOT / 'docs/G18_房间结果与滤波对照决策.md'
    message = """# G18 房间实验结论与下一步

21 个系统在 504 个条件上获得完整评分。以目标 20°、刚性训练加 MVDR 为例，反射从 0 增至请求 RT 0.35 秒，SI-SDR 从 7.02 dB 降到 2.08 dB，CTC WER 从 5.2% 升到 54.3%。同组无噪声参考的 CTC WER 则从 1.6% 升至 6.3%（合并三方向）。噪声和反射共存时的退化值得单独研究。请求 0.35 秒时，33 个声源位置的 T20 外推中位数为 0.459 秒。

刚性和反向训练的识别差异随条件变化。请求 RT 0.15 秒、办公室噪声、70° 的 72 对 MVDR 输出共计 567 个参考词。Whisper 的总错误差为 193 个，插入差占 153 个。CTC 的总错误差为 21 个，插入计数方向相反。此项事后核查定位了识别分差的来源，人耳对这些词的判断仍待采集。

下一步沿用全部十个神经前端及其输出，用标准多通道 Wiener 滤波检验输出滤波器的影响。主参数固定为 μ=1，训练步数和掩码取法沿用现有实验。MWF 用共同帧数估计两个协方差，原 MVDR 使用分别归一化的协方差。配对差反映这两种完整滤波配置的差异。除 SI-SDR、PESQ、STOI 和双后端 WER，干净控制还需核查输出增益和幅度误差。

该比较的适用范围是已查看的一个模拟房间、4 位说话人和两条噪声录音。G18 的 UTC 时间戳间隔和单调计时器分别为 2073.08 秒、1599.66 秒，差值的原因尚未核实。阶段耗时据此保留两种记录。
"""
    report.write_text(message, encoding='utf-8')
    g18_common.archive('next_mwf_decision', report, [
        ROOT / 'configs/g19_mwf_plan.json', PREV / 'FINAL_ACCEPTANCE_AUDIT.json',
        PREV / 'evaluation/posthoc_ASR_insertions.json', PREV / 'timing_clock_observation.json',
    ])


def main():
    stage_decision()
    plan_path = ROOT / 'configs/g19_mwf_plan.json'
    anchor = metadata(PREV / 'stages/next_mwf_decision/archive_manifest.json')
    assert sha(plan_path) == next(row['sha256'] for row in anchor['files'] if row['source'] == 'configs/g19_mwf_plan.json')
    OUT.mkdir(exist_ok=False)
    plan = metadata(plan_path)
    plan.update(frozen_utc=now(), G18_combined_signals_sha256=sha(PREV / 'evaluation/combined_signals.jsonl'),
                G18_ASR_score_sha256={e: sha(PREV / 'evaluation' / e / 'asr_scores.jsonl') for e in ('Whisper', 'CTC')},
                G18_ASR_provenance_sha256={e: sha(PREV / 'evaluation' / e / 'provenance.json') for e in ('Whisper', 'CTC')},
                filter_sha256=sha(ROOT / 'src/mcse/wiener.py'))
    assert plan['G18_combined_signals_sha256'] == metadata(PREV / 'evaluation/BEAM_COMPLETE.json')['combined_sha256']
    dump(OUT / 'frozen_plan.json', plan, True)
    status(stage='preflight')
    log = ROOT / 'artifacts/wiener_preflight_unit_v1.log'
    assert 'Ran 6 tests' in log.read_text() and 'OK' in log.read_text()
    cases = read(ROOT / 'data/g18_v1/manifest.jsonl')
    source = metadata(PREV / 'selected_sources.json')[0]['source_id']
    selected = [c for c in cases if c['source_id'] == source and c['target_azimuth'] == 20 and (
        (c['noise_id'] == 'OOFFICE' and c['relative_angle'] == -80) or
        (c['condition'] == 'clean' and c['requested_rt60'] == .35))]
    assert len(selected) == 4
    rows = read(PREV / 'evaluation/combined_signals.jsonl')
    indexed = {(r['system'], r['sample_id']): r for r in rows}
    frontends = [r['system'] for r in metadata(PREV / 'systems.json')['direct'] if r['kind'] != 'noisy']
    torch.set_num_threads(2)
    probes = []
    for c in selected:
        assert sha(c['noisy_path']) == c['input_sha256']
        x, rate = sf.read(c['noisy_path'], dtype='float32', always_2d=True)
        for frontend in frontends:
            direct = indexed[frontend, c['sample_id']]
            old_filter = indexed[frontend + '_mask_mvdr', c['sample_id']]
            assert sha(direct['output_path']) == old_filter['mask_estimate_sha256'] == direct['output_sha256']
            mono, mono_rate = sf.read(direct['output_path'], dtype='float32')
            y, info = mask_mwf(x, mono, mu=plan['mu'])
            difference = abs(info['mask_mean'] - old_filter['beamforming_diagnostics']['mask_mean'])
            assert difference <= 1e-12 and info['solve_relative_residual_max'] <= 1e-8
            assert rate == mono_rate == 16000 and y.shape == mono.shape and np.isfinite(y).all()
            probes.append({'sample_id': c['sample_id'], 'frontend': frontend, 'mask_mean_abs_diff': difference, **info})
    assert len(probes) == 40
    dump(OUT / 'preflight.json', {'status': 'ok', 'completed_utc': now(), 'probes': probes}, True)
    report = ROOT / 'docs/G19A_多通道Wiener对照协议与验证.md'
    report.write_text(f"""# G19A 多通道 Wiener 对照协议与验证

本阶段复用 G18 的 504 个房间条件。每个前端的直接输出同时用作 MWF 和已有 MVDR 的掩码估计来源。十个前端各产生 504 个新输出，全部种子进入比较。

主实验固定 μ=1。两个加权混合协方差使用同一帧数作分母，求解使用复数共轭输出、参考麦索引 4 及迹相关的对角加载。该配置的比较范围包括协方差估计约定和滤波器求解。干净条件的 RMS 增益和归一化平方误差补充 SI-SDR，以观察语音电平变化。

6 项解析测试通过，覆盖复数相位、常数掩码、秩亏和全零输入等情况。40 个真实文件探测覆盖十个前端，保存前端输出的哈希与原 MVDR 的掩码来源一致。掩码均值最大绝对差为 {max(r['mask_mean_abs_diff'] for r in probes):.3g}，线性方程最大相对残差为 {max(r['solve_relative_residual_max'] for r in probes):.3g}。

后续分别保存信号、Whisper、CTC 及配对报告。推理权重、实验语句和识别选项沿用 G18。此阶段属于成熟方法的配对实验。理论来源见[研究补查](研究补查_失真加权Wiener成熟对照_20261112.md)。
""", encoding='utf-8')
    archive('preflight', report, [OUT / 'preflight.json', OUT / 'frozen_plan.json', log, plan_path,
                                   ROOT / 'src/mcse/wiener.py', ROOT / 'tests/test_wiener.py', Path(__file__)])
    status(stage='preflight_complete')


if __name__ == '__main__':
    main()
