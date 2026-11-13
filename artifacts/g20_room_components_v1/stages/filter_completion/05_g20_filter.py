#!/usr/bin/env python3
"""Conditional component diagnostics of the archived room MWF/MVDR outputs."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import json
import time
import traceback
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from g20_replay import ROOT, OUT, G18, G19, sha, dump, now, read, archive
from mcse.beamforming import _input, _stft, _wave, REFERENCE
from mcse.components import pcm24_field, component_metrics
from mcse.reverberation import full_speech
from mcse.wiener import masked_covariances, wiener_weights


def main():
    assert json.loads((OUT / 'REPLAY_COMPLETE.json').read_text())['status'] == 'ok'
    assert (OUT / 'stages/replay/archive_manifest.json').exists()
    folder = OUT / 'filter'; folder.mkdir(exist_ok=False)
    pinned = ['scripts/g20_filter.py', 'src/mcse/components.py', 'src/mcse/beamforming.py',
              'src/mcse/wiener.py', 'src/mcse/reverberation.py',
              'artifacts/g19_room_mwf_v1/evaluation/combined_signals.jsonl',
              'artifacts/g20_room_components_v1/REPLAY_COMPLETE.json',
              'data/g18_v1/manifest.jsonl']
    plan = {'frozen_utc': now(), 'hashes': {n: sha(ROOT / n) for n in pinned},
            'expected_rows': 8640, 'max_seconds': 3600,
            'saved_output_tolerance': 1e-6, 'linearity_relative_peak_tolerance': 1e-6,
            'scope': '432 noisy cases x 10 fixed frontends x 2 filters; no clean reference in weights',
            'preflight_rule': 'first source, OOFFICE point -80, target20, all RT, ten frontends, both filters:60 rows'}
    dump(folder / 'frozen_plan.json', plan, True)
    protocol = ROOT / 'docs/G20B_固定房间滤波分量诊断协议.md'
    with protocol.open('x', encoding='utf-8') as f:
        f.write('# G20B 固定房间滤波分量诊断协议\n\nG20A 全部504例还原归档后，仅分析432个带噪条件、全部10前端及两滤波器，共8640项。先由原混合和已存前端估计重算盲权重并复现旧输出，再以同一个固定权重分别处理 S_q 和 X_pcm−S_q。\n\n先按固定规则检查60项并归档，再运行完整覆盖。报告语音 NMSE、目标增益、噪声衰减、分量输出 SNR，以及总误差的语音/噪声/交叉项。所有值为条件线性诊断，不定位词语缺失，也不声称真实场景或跨房间泛化。μ及正则保持原值，不训练、不运行ASR。\n')
    archive('filter_protocol', protocol, [folder / 'frozen_plan.json', ROOT / 'scripts/g20_filter.py'])
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    started = time.monotonic()
    signals = {(r['system'], r['sample_id']): r for r in read(G19 / 'evaluation/combined_signals.jsonl')}
    registry = json.loads((G18 / 'systems.json').read_text())
    frontends = {r['system']: r for r in registry['direct'] if r['kind'] != 'noisy'}
    assert len(frontends) == 10
    sources = {r['source_id']: r for r in json.loads((G18 / 'selected_sources.json').read_text())}
    cases = [c for c in read(ROOT / 'data/g18_v1/manifest.jsonl') if c['condition'] != 'clean']
    assert len(cases) == 432
    verified = json.loads((OUT / 'verified_files.json').read_text())
    checked = {}
    def verify(path, digest):
        path = str(path)
        if path not in checked:
            assert sha(path) == digest, path
            checked[path] = digest
        else:
            assert checked[path] == digest
    def wave(path, digest):
        verify(path, digest)
        a, sr = sf.read(path, dtype='float32')
        assert sr == 16000 and np.isfinite(a).all()
        return a
    mono = {}
    def evaluate(c):
        assert time.monotonic() - started < plan['max_seconds']
        src = sources[c['source_id']]
        source_path = ROOT / 'data/g17_v1/sources' / src['path']
        verify(source_path, src['sha256'])
        if src['source_id'] not in mono:
            mono[src['source_id']], sr = sf.read(source_path, dtype='float64'); assert sr == 16000
        verify(c['speech_RIR_path'], verified[c['speech_RIR_path']])
        sq = pcm24_field(full_speech(mono[src['source_id']], np.load(c['speech_RIR_path'])) * c['common_scale'])
        x = wave(c['noisy_path'], c['input_sha256'])
        ref = wave(c['reference_path'], c['reference_sha256'])
        assert np.array_equal(sq[:, 4], ref)
        neff = x.astype(np.float64) - sq.astype(np.float64)
        X = _stft(_input(x)); S = _stft(_input(sq)); N = _stft(_input(neff))
        for name, identity in frontends.items():
            direct = signals[name, c['sample_id']]
            e = wave(direct['output_path'], direct['output_sha256'])
            E = _stft(torch.from_numpy(e.astype(np.float64)))
            power = E.abs().square(); m = power / (power + (X[4] - E).abs().square() + 1e-10)
            # Clean components are never used to estimate either filter.
            for method in ('mvdr', 'mwf'):
                if method == 'mvdr':
                    ps = AF.psd(X, m, normalize=True, eps=1e-10)
                    pn = AF.psd(X, 1-m, normalize=True, eps=1e-10)
                    w = AF.mvdr_weights_souden(ps, pn, REFERENCE, diagonal_loading=True, diag_eps=1e-7, eps=1e-8)
                else:
                    ps, pn = masked_covariances(X, m); w, _ = wiener_weights(ps, pn, mu=1.)
                filter_wave = lambda z: _wave(AF.apply_beamforming(w, z), c['frames'])
                y, ys, yn = (filter_wave(z) for z in (X, S, N))
                system = name + '_mask_' + method
                saved_row = signals[system, c['sample_id']]
                saved = wave(saved_row['output_path'], saved_row['output_sha256'])
                error = float(np.abs(y.astype(float)-saved).max())
                linearity = float(np.abs(y.astype(float)-ys.astype(float)-yn.astype(float)).max())
                assert error <= plan['saved_output_tolerance']
                assert linearity <= plan['linearity_relative_peak_tolerance']*(1+float(np.abs(y).max()))
                metrics = component_metrics(ref, neff[:, 4], ys, yn)
                assert metrics['snr_accounting_error_db'] < 1e-8
                d = ys.astype(float) - ref.astype(float); noise = yn.astype(float)
                energy = float(ref.astype(float) @ ref.astype(float))
                dist = float(d @ d) / energy; residual = float(noise @ noise) / energy
                cross = float(2*(d @ noise)) / energy
                total = float((d+noise) @ (d+noise)) / energy
                accounting = abs(total-dist-residual-cross)
                assert accounting <= 1e-10*(1+total)
                metrics.update(target_error_nmse=dist, residual_noise_nmse=residual,
                               error_cross_term=cross, reconstructed_total_nmse=total,
                               error_accounting_abs=accounting)
                yield {**{k:c[k] for k in ('sample_id','source_id','speaker_id','requested_rt60','noise_field','noise_id','target_azimuth','relative_angle')},
                       'frontend': name, 'system': system, 'method': method,
                       'group': identity.get('geometry', identity['kind']), 'seed': identity.get('seed'),
                       'metrics': metrics, 'saved_output_max_abs': error, 'linearity_max_abs': linearity}
    first = next(iter(sources))
    probe = [c for c in cases if c['source_id']==first and c['noise_id']=='OOFFICE'
             and c['noise_field']=='point' and c['relative_angle']==-80 and c['target_azimuth']==20]
    assert len(probe) == 3
    probes = [r for c in probe for r in evaluate(c)]
    assert len(probes) == 60
    dump(folder / 'preflight.json', probes, True)
    preport = ROOT / 'docs/G20B_滤波分量预检报告.md'
    with preport.open('x', encoding='utf-8') as f:
        f.write('# G20B 滤波分量预检\n\n固定60项均通过：还原参考完全一致、旧混合输出误差≤1e−6、分量线性叠加及能量交叉项核算成立。真值未用于权重。预检结果先归档后才开始全部8640项。\n')
    archive('filter_preflight', preport, [folder / 'preflight.json', folder / 'frozen_plan.json'])
    complete = []
    with (folder / 'components.jsonl').open('x', encoding='utf-8') as log:
        for i, c in enumerate(cases, 1):
            for r in evaluate(c):
                log.write(json.dumps(r, allow_nan=False)+'\n'); complete.append(r)
            log.flush()
            if i % 24 == 0:
                dump(OUT / 'status.json', {'stage':'filter_components','cases':i,'rows':len(complete),'total_rows':8640,'utc':now()})
    assert len(complete) == 8640 and not torch.cuda.is_initialized()
    assert all(sha(ROOT / n) == h for n,h in plan['hashes'].items())
    buckets = defaultdict(list)
    for r in complete: buckets[(r['requested_rt60'],r['group'],r['method'])].append(r)
    measures = ('speech_error_nmse','target_gain_db','noise_reduction_db','component_output_snr_db',
                'target_error_nmse','residual_noise_nmse','error_cross_term','reconstructed_total_nmse')
    summary = [{'rt':rt,'group':group,'method':method,'rows':len(rows),
                **{m:float(np.mean([r['metrics'][m] for r in rows])) for m in measures}}
               for (rt,group,method),rows in sorted(buckets.items())]
    dump(folder / 'summary.json', summary, True)
    audit = {'status':'ok','rows':len(complete),'completed_utc':now(), 'CPU_only':True,
             'saved_output_max_abs':max(r['saved_output_max_abs'] for r in complete),
             'linearity_max_abs':max(r['linearity_max_abs'] for r in complete),
             'elapsed_seconds':time.monotonic()-started,'components_sha256':sha(folder/'components.jsonl'),
             'summary_sha256':sha(folder/'summary.json')}
    dump(folder / 'COMPLETE.json', audit, True); dump(folder / 'verified_files.json', checked, True)
    lines = ['# G20 固定房间滤波分量诊断报告','','完成8640项固定权重条件诊断。每项先复现原混合输出，再以同一权重处理量化语音和有效噪声。未训练或重新识别。真值仅用于诊断。','',
             '|请求RT|前端组|滤波|例数|语音NMSE|目标增益dB|噪声衰减dB|分量输出SNR dB|残余噪声NMSE|误差交叉项|',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['rt']}|{r['group']}|{r['method']}|{r['rows']}|"+'|'.join(f"{r[m]:.4f}" for m in measures[:4]+('residual_noise_nmse','error_cross_term'))+'|')
    lines += ['', '表中均为逐例指标算术平均；dB量是逐例dB后平均。负目标增益表示能量衰减，不能直接解释为吞字。带噪NMSE分解须同时计入语音误差、残余噪声与交叉项。有效噪声包含G20A已核验的量化差额。',
              '', '本诊断解释既定滤波器的条件分量响应，不能把差异因果归于神经掩码，也不支持多房间、真实阵列、语言内容或创新性结论。全部三训练种子保留在逐例记录，语句重复不作为独立样本。', '', now()]
    report = ROOT / 'docs/G20_固定房间滤波分量诊断报告.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('filter_completion',report,[folder/'COMPLETE.json',folder/'summary.json',folder/'verified_files.json',folder/'frozen_plan.json',ROOT/'scripts/g20_filter.py'])
    dump(OUT/'status.json',{'stage':'filter_complete_pending_readback','utc':now()})


if __name__ == '__main__':
    try:
        with torch.inference_mode(): main()
    except Exception as e:
        if OUT.exists(): dump(OUT/'filter_failure.json',{'utc':now(),'error':repr(e)})
        traceback.print_exc(); raise
    finally:
        if OUT.exists(): (OUT/'keep_awake_stop').write_text('G20 component job ended\n')
