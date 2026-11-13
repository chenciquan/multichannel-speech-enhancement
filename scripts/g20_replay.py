#!/usr/bin/env python3
"""Reconstruct archived room components; no training, decoding or new mixtures."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time
from functools import lru_cache
import numpy as np
import soundfile as sf
import g8_common as common
from mcse.components import pcm24_field
from mcse.reverberation import full_speech, continuous_noise
from mcse.spatial_noise import highpass40

ROOT = common.ROOT
OUT = ROOT / 'artifacts/g20_room_components_v1'
G18 = ROOT / 'artifacts/g18_room_stress_v1'
G19 = ROOT / 'artifacts/g19_room_mwf_v1'
sha, dump, now, read = common.sha, common.dump, common.now, common.read


def archive(stage, report, files):
    previous = common.OUT
    try:
        common.OUT = OUT
        common.archive(stage, report, files)
    finally:
        common.OUT = previous


def main():
    assert json.loads((G19 / 'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status'] == 'ok'
    decision = ROOT / 'docs/G19_结果读回与房间分量诊断决策.md'
    decision.write_text('# G19 结果读回与下一阶段决策\n\n固定 μ=1 的成熟 MWF 未形成整体优势。72 个细分格中，23 格 SI-SDR 更高，0 格两个 ASR 的 WER 同时下降。参数不据此调整，全部负结果保留。\n\n下一步先独立还原 G18 的六通道语音及噪声，再检验固定滤波器中的语音失真、残余噪声及误差交叉项。真值仅进入诊断，不用于估计掩码或权重。这不能证明吞字、掩码成因或跨房间泛化；也不是新算法。先归档还原结果，再进入完整滤波分解。\n', encoding='utf-8')
    previous = common.OUT
    try:
        common.OUT = G19
        common.archive('next_component_decision', decision, [G19 / 'FINAL_ACCEPTANCE_AUDIT.json'])
    finally:
        common.OUT = previous
    OUT.mkdir(exist_ok=False)
    names = ['scripts/g20_replay.py', 'src/mcse/components.py', 'src/mcse/reverberation.py',
             'src/mcse/spatial_noise.py', 'scripts/g18_build.py',
             'data/g18_v1/manifest.jsonl', 'artifacts/g18_room_stress_v1/selected_sources.json',
             'artifacts/g18_room_stress_v1/noise_sources.json', 'artifacts/g18_room_stress_v1/RIR_manifest.json',
             'artifacts/g19_room_mwf_v1/FINAL_ACCEPTANCE_AUDIT.json']
    plan = {'frozen_utc': now(), 'max_seconds': 1200, 'expected_cases': 504,
            'input_reference_match': 'exact PCM24 decoded samples',
            'quantization_residual_bound': 2 / 2**23,
            'components': 'S_q=PCM24(full_speech*common_scale); N_eff=X_pcm-S_q (float64)',
            'source_hashes': {name: sha(ROOT / name) for name in names},
            'next': 'Only after replay archive: recompute blind weights, reproduce saved mixed outputs, freeze each weight and filter components.'}
    dump(OUT / 'frozen_replay_plan.json', plan, True)
    protocol = ROOT / 'docs/G20A_房间分量还原协议.md'
    with protocol.open('x', encoding='utf-8') as f:
        f.write('# G20A 房间分量还原协议\n\n保留 G18 所有 504 例（432 带噪、72 干净）。从保存的实际 RIR、源 FLAC、整条高通后的噪声及已冻结区间，按原运算顺序还原混合。六通道混合和第5麦参考均须量化后逐样本完全一致。\n\n正式诊断分量定义为量化六通道语音 S_q 与 X_pcm−S_q；后者包含噪声及量化差额，须另报相对连续物理噪声的偏差，以 2 个 PCM24 LSB 为预定上限。仅验证数据，不运行增强网络或识别器。\n')
    archive('protocol', protocol, [OUT / 'frozen_replay_plan.json', ROOT / 'scripts/g20_replay.py'])
    start = time.monotonic()
    verified = {}
    def check(path, expected):
        path = str(path)
        if path not in verified:
            assert sha(path) == expected, path
            verified[path] = expected
        else:
            assert verified[path] == expected
    sources = {r['source_id']: r for r in json.loads((G18 / 'selected_sources.json').read_text())}
    rirs = {r['path']: r for r in json.loads((G18 / 'RIR_manifest.json').read_text())}
    @lru_cache(maxsize=20)
    def response(path):
        check(path, rirs[path]['sha256'])
        return np.load(path)
    @lru_cache(maxsize=12)
    def speech(source, path):
        row = sources[source]
        flac = ROOT / 'data/g17_v1/sources' / row['path']
        check(flac, row['sha256'])
        mono, sr = sf.read(flac, dtype='float64')
        assert sr == 16000 and len(mono) == row['samples']
        return full_speech(mono, response(path))
    noises = {}
    for row in json.loads((G18 / 'noise_sources.json').read_text()):
        check(row['path'], row['sha256'])
        wave, sr = sf.read(row['path'], dtype='float64')
        assert sr == 16000
        noises[row['noise_id']] = highpass40(wave)
    cases = read(ROOT / 'data/g18_v1/manifest.jsonl')
    results = []
    with (OUT / 'replay.jsonl').open('x', encoding='utf-8') as log:
        for c in cases:
            assert time.monotonic() - start < plan['max_seconds']
            s = speech(c['source_id'], c['speech_RIR_path'])
            assert s.shape == (c['frames'], 6)
            n = np.zeros_like(s)
            k = len(c['noise_RIR_paths'])
            assert k == len(c['noise_intervals'])
            for path, (lo, hi) in zip(c['noise_RIR_paths'], c['noise_intervals']):
                n += continuous_noise(noises[c['noise_id']][lo:hi], response(path),
                                      c['noise_preroll_samples'], c['frames']) / np.sqrt(k)
            mixed = pcm24_field((s + c['noise_gain'] * n) * c['common_scale'])
            sq = pcm24_field(s * c['common_scale'])
            check(c['noisy_path'], c['input_sha256'])
            check(c['reference_path'], c['reference_sha256'])
            saved, sr = sf.read(c['noisy_path'], dtype='float32', always_2d=True)
            ref, rr = sf.read(c['reference_path'], dtype='float32')
            assert sr == rr == 16000 and np.array_equal(mixed, saved) and np.array_equal(sq[:, 4], ref), c['sample_id']
            effective = saved.astype(np.float64) - sq.astype(np.float64)
            physical = (c['noise_gain'] * n) * c['common_scale']
            delta = float(np.abs(effective - physical).max())
            assert delta <= plan['quantization_residual_bound']
            row = {key: c[key] for key in ('sample_id', 'source_id', 'speaker_id', 'requested_rt60', 'noise_field', 'target_azimuth')}
            row.update(input_exact=True, reference_exact=True,
                       effective_noise_minus_physical_max_abs=delta,
                       effective_noise_minus_physical_max_LSB=delta * 2**23,
                       component_closure_max_abs=float(np.abs(sq.astype(np.float64) + effective - saved).max()))
            assert row['component_closure_max_abs'] == 0
            log.write(json.dumps(row) + '\n'); log.flush(); results.append(row)
            if len(results) % 72 == 0:
                dump(OUT / 'status.json', {'stage': 'replay', 'done': len(results), 'total': 504, 'utc': now()})
    assert len(results) == 504
    assert all(sha(ROOT / name) == digest for name, digest in plan['source_hashes'].items())
    complete = {'status': 'ok', 'completed_utc': now(), 'cases': len(results),
                'input_exact_cases': 504, 'reference_exact_cases': 504,
                'quantization_max_LSB': max(r['effective_noise_minus_physical_max_LSB'] for r in results),
                'closure_max_abs': 0, 'elapsed_seconds': time.monotonic() - start,
                'replay_sha256': sha(OUT / 'replay.jsonl')}
    dump(OUT / 'REPLAY_COMPLETE.json', complete, True)
    dump(OUT / 'verified_files.json', verified, True)
    report = ROOT / 'docs/G20A_房间分量还原验证报告.md'
    with report.open('x', encoding='utf-8') as f:
        f.write('# G20A 房间分量还原验证报告\n\n全部504例六通道混合及第5麦参考均逐样本完全还原；S_q 与有效噪声严格闭合。有效噪声相对连续物理噪声的最大偏差为 ' + f"{complete['quantization_max_LSB']:.6f}" + ' 个 PCM24 LSB，低于事先固定的2 LSB上限。\n\n本阶段仅验证诊断分量来源，不是滤波效果、语音误删或创新性证据。后续应先重现旧混合输出，再固定权重滤分量，并计入平方误差交叉项。未运行网络、ASR或训练。\n')
    archive('replay', report, [OUT / 'REPLAY_COMPLETE.json', OUT / 'replay.jsonl', OUT / 'verified_files.json', ROOT / 'scripts/g20_replay.py'])
    dump(OUT / 'status.json', {'stage': 'replay_archived_pending_filter_diagnostic', 'utc': now()})


if __name__ == '__main__':
    main()
