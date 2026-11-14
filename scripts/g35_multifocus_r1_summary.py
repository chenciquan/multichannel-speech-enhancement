"""Aggregate every approved round-one row; plotting/reporting only, no forwards."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import numpy as np
from g35_multifocus_r1 import ROOT, OUT, SEEDS, METRICS, read, meta, sha, dump

ANGLES = list(range(-90, 91, 10))


def csv_write(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def groups(rows, key):
    result = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return result


def average(rows, metric):
    return statistics.fmean(r['metrics'][metric] for r in rows)


def summarize():
    from matplotlib import font_manager
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert meta(OUT / 'DELIVERY_AUDIT.json')['status'] == 'passed'
    rows = read(OUT / 'signals.jsonl'); assert len(rows) == 6384
    noisy = [r for r in rows if r['condition'] != 'clean']
    clean = [r for r in rows if r['condition'] == 'clean']
    assert len(noisy) == 5472 and len(clean) == 912
    destination = OUT / 'curves'; destination.mkdir(exist_ok=False)
    cells = groups(noisy, lambda r: (r['system'], r['target_azimuth']))
    profiles = []
    for (system, angle), items in sorted(cells.items()):
        assert len(items) == 48
        profiles.append({'system': system, 'arm': items[0]['arm'], 'seed': items[0]['seed'],
                         'angle': angle, 'n': 48, **{m: average(items, m) for m in METRICS}})
    csv_write(destination / 'per_seed_19_angles.csv', profiles)
    bygroup = groups(noisy, lambda r: (r['system'], r['target_azimuth'], r['noise_id'],
                                      r['condition'], str(r['relative_angle'])))
    group_rows = [{'system': s, 'arm': items[0]['arm'], 'seed': items[0]['seed'], 'angle': a,
                   'noise': n, 'field': f, 'relative': rel, 'n': len(items),
                   **{m: average(items, m) for m in METRICS}}
                  for (s, a, n, f, rel), items in sorted(bygroup.items())]
    assert len(group_rows) == 684 and all(r['n'] == 8 for r in group_rows)
    csv_write(destination / 'noise_field_groups.csv', group_rows)
    means = {(arm, a): {m: statistics.fmean(p[m] for p in profiles if p['arm'] == arm and p['angle'] == a)
                            for m in METRICS} for arm in ('A', 'B') for a in ANGLES}
    csv_write(destination / 'mean_19_angles.csv', [
        {'arm': arm, 'angle': a, **means[arm, a]} for arm in ('A', 'B') for a in ANGLES])
    summary = []
    for arm in ('A', 'B'):
        for m in METRICS:
            v = np.array([means[arm, a][m] for a in ANGLES])
            inside = np.array([-45 <= a <= 75 for a in ANGLES])
            summary.append({'arm': arm, 'metric': m, 'mean_19': float(v.mean()), 'worst': float(v.min()),
                'worst_angles': ','.join(str(a) for a, x in zip(ANGLES, v) if x == v.min()),
                'best': float(v.max()), 'best_angles': ','.join(str(a) for a, x in zip(ANGLES, v) if x == v.max()),
                'range': float(np.ptp(v)), 'angle_std_population': float(v.std()),
                'inside_mean': float(v[inside].mean()), 'outside_mean': float(v[~inside].mean())})
    csv_write(destination / 'summary_19.csv', summary)
    seed_summaries = []
    for system in sorted({r['system'] for r in noisy}):
        for m in METRICS:
            pr = sorted([p for p in profiles if p['system'] == system], key=lambda p: p['angle'])
            v = np.array([p[m] for p in pr])
            seed_summaries.append({'system': system, 'arm': pr[0]['arm'], 'seed': pr[0]['seed'], 'metric': m,
                'mean_19': float(v.mean()), 'worst': float(v.min()), 'worst_angle': pr[int(v.argmin())]['angle'],
                'best': float(v.max()), 'best_angle': pr[int(v.argmax())]['angle'],
                'range': float(np.ptp(v)), 'angle_std_population': float(v.std())})
    csv_write(destination / 'per_seed_summary.csv', seed_summaries)
    lookup = {(r['system'], r['sample_id']): r for r in noisy}
    paired = []
    for seed in SEEDS:
        for a in [r for r in noisy if r['system'] == f'A_s{seed}']:
            b = lookup[f'B_s{seed}', a['sample_id']]
            assert a['input_sha256'] == b['input_sha256'] and a['reference_sha256'] == b['reference_sha256']
            paired.append({'seed': seed, 'sample_id': a['sample_id'], 'angle': a['target_azimuth'],
                'speaker': a['speaker_id'], 'noise': a['noise_id'], 'field': a['condition'],
                'relative': a['relative_angle'], **{m: b['metrics'][m] - a['metrics'][m] for m in METRICS}})
    assert len(paired) == 2736
    csv_write(destination / 'all_paired_differences.csv', paired)
    seed_deltas = [{'seed': s, 'angle': a, **{m: statistics.fmean(p[m] for p in paired if p['seed'] == s and p['angle'] == a)
                    for m in METRICS}} for s in SEEDS for a in ANGLES]
    csv_write(destination / 'per_seed_19_differences.csv', seed_deltas)
    gd = groups(paired, lambda r: (r['angle'], r['noise'], r['field'], str(r['relative'])))
    group_deltas = [{'angle': a, 'noise': n, 'field': f, 'relative': rel, 'n': len(v),
                     **{m: statistics.fmean(p[m] for p in v) for m in METRICS}}
                    for (a, n, f, rel), v in sorted(gd.items())]
    csv_write(destination / 'noise_field_differences.csv', group_deltas)
    speaker_cells = groups(noisy, lambda r: (r['system'], r['target_azimuth'], r['speaker_id']))
    csv_write(destination / 'speaker_groups.csv', [
        {'system': s, 'angle': a, 'speaker': sp, 'n': len(v), **{m: average(v, m) for m in METRICS}}
        for (s, a, sp), v in sorted(speaker_cells.items())])
    clean_cells = groups(clean, lambda r: (r['system'], r['target_azimuth']))
    clean_profiles = [{'system': s, 'arm': v[0]['arm'], 'seed': v[0]['seed'], 'angle': a, 'n': len(v),
        **{m: average(v, m) for m in METRICS},
        **{m: statistics.fmean(r['clean_fidelity'][m] for r in v) for m in ('NMSE', 'RMS_gain')}}
        for (s, a), v in sorted(clean_cells.items())]
    assert len(clean_profiles) == 114 and all(p['n'] == 8 for p in clean_profiles)
    csv_write(destination / 'clean_per_seed_19_angles.csv', clean_profiles)
    clean_table = []
    for s, v in sorted(groups(clean, lambda r: r['system']).items()):
        assert len(v) == 152
        clean_table.append({'system': s, 'arm': v[0]['arm'], 'seed': v[0]['seed'], 'n': 152,
            **{m: average(v, m) for m in METRICS},
            **{m: statistics.fmean(r['clean_fidelity'][m] for r in v) for m in ('NMSE', 'RMS_gain')}})
    csv_write(destination / 'clean_controls.csv', clean_table)
    run_results = [meta(OUT / 'train' / f'{arm}_s{s}' / 'result.json') for s in SEEDS for arm in ('A', 'B')]
    content_shift = []
    for s in SEEDS:
        aa = next(r for r in run_results if r['arm'] == 'A' and r['seed'] == s)['actual_sampling']
        bb = next(r for r in run_results if r['arm'] == 'B' and r['seed'] == s)['actual_sampling']
        for field in ('field_counts', 'noise_counts', 'SNR_counts', 'speaker_counts', 'source_counts'):
            for value in sorted(set(aa[field]) | set(bb[field])):
                content_shift.append({'seed': s, 'attribute': field, 'value': value,
                    'A_visits': aa[field].get(value, 0), 'B_visits': bb[field].get(value, 0),
                    'B_minus_A': bb[field].get(value, 0) - aa[field].get(value, 0)})
    csv_write(destination / 'training_content_shift.csv', content_shift)
    # Independently use raw rows for each mean-cell check; no model calls.
    max_error = max(abs(means[arm, a][m] - average([r for r in noisy if r['arm'] == arm and r['target_azimuth'] == a], m))
                    for arm in ('A', 'B') for a in ANGLES for m in METRICS)
    assert max_error <= 1e-12
    dump(OUT / 'summary_values.json', {'summary': summary, 'clean': clean_table,
         'pair_count': len(paired), 'independent_curve_max_error': max_error}, True)
    font = '/mnt/c/Windows/Fonts/msyh.ttc'; assert Path(font).is_file()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = {'A': '#2569ad', 'B': '#df6b20'}
    labels = {'A': 'A：几何保持式多方位 LoRA（新匹配）', 'B': 'B：五重点方位联合 LoRA'}
    units = {'si_sdr_db': 'SI-SDR (dB)', 'pesq_wb': 'PESQ-WB', 'stoi': 'STOI', 'NMSE': 'NMSE', 'RMS_gain': 'RMS增益'}

    def decorate(ax, zero=False):
        ax.axvspan(-45, 75, color='#b9c8bd', alpha=.12)
        for center in (-40, 0, 10, 30, 70):
            ax.axvspan(center - 5, center + 5, color=colors['B'], alpha=.10)
        ax.axvline(20, color='#666', ls=':', lw=1)
        if zero:
            ax.axhline(0, color='#333', lw=.7)
        ax.set_xlim(-90, 90); ax.set_xticks(ANGLES); ax.grid(alpha=.2)

    def savefig(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(destination / f'{name}.{ext}', dpi=300)
        plt.close(fig)

    old = list(csv.DictReader((ROOT / 'artifacts/g35_azimuth_scan_v1/curves/mean_19_angles.csv').open(encoding='utf-8-sig')))
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for ax, m in zip(axes, METRICS):
        decorate(ax)
        for arm in ('A', 'B'):
            for s in SEEDS:
                ax.plot(ANGLES, [next(p[m] for p in profiles if p['system'] == f'{arm}_s{s}' and p['angle'] == a)
                                 for a in ANGLES], color=colors[arm], alpha=.23, lw=.8)
            ax.plot(ANGLES, [means[arm, a][m] for a in ANGLES], '-o', color=colors[arm], lw=2, ms=3, label=labels[arm])
        for family, label, color in [('fixed', '历史单方位 +20°（G35既有）', '#47844b'),
                                     ('mcnet', '官方McNet（G35既有）', '#222'), ('noisy', '参考麦（G35既有）', '#888')]:
            ax.plot(ANGLES, [float(next(r[m] for r in old if r['family'] == family and int(r['angle']) == a)) for a in ANGLES],
                    '--', color=color, lw=1, alpha=.8, label=label)
        ax.set_ylabel(units[m])
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].set_title('首轮匹配开发评价：完整19点；粗线=三种子均值，细线=所有种子')
    axes[-1].set_xlabel('目标水平方位（度；色带为五重点训练邻域，虚线为历史 +20°）')
    savefig(fig, 'three_metrics_19_angles')
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, m in zip(axes, METRICS):
        decorate(ax, True)
        for s in SEEDS:
            ax.plot(ANGLES, [next(d[m] for d in seed_deltas if d['seed'] == s and d['angle'] == a) for a in ANGLES],
                    color=colors['B'], alpha=.35, lw=1)
        ax.plot(ANGLES, [means['B', a][m] - means['A', a][m] for a in ANGLES], '-o', color=colors['B'], ms=3, lw=2)
        ax.set_ylabel('B−A ' + units[m])
    axes[0].set_title('固定曝光方案的配对差：B − A（同种子、同G35输入）')
    axes[-1].set_xlabel('目标水平方位（度；连接线不代表连续角度保证）')
    savefig(fig, 'paired_differences_19_angles')
    spaces = [('point', '-80'), ('point', '25'), ('multi8', 'None')]
    for m in METRICS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        for i, noise in enumerate(('OOFFICE', 'NPARK')):
            for j, (field, relative) in enumerate(spaces):
                ax = axes[i, j]; decorate(ax)
                for arm in ('A', 'B'):
                    for seed in SEEDS:
                        ax.plot(ANGLES, [next(r[m] for r in group_rows if r['system'] == f'{arm}_s{seed}' and
                            r['angle'] == a and r['noise'] == noise and r['field'] == field and r['relative'] == relative)
                            for a in ANGLES], color=colors[arm], alpha=.2, lw=.7)
                    values = [statistics.fmean(r[m] for r in group_rows if r['arm'] == arm and r['angle'] == a
                              and r['noise'] == noise and r['field'] == field and r['relative'] == relative) for a in ANGLES]
                    ax.plot(ANGLES, values, '-o', color=colors[arm], ms=2, label=arm)
                ax.set_title(f'{noise} / {field} / 相对角 {relative}'); ax.set_ylabel(units[m]); ax.tick_params(axis='x', labelsize=7)
        axes[0, 0].legend(); fig.suptitle('六种噪声/空间组合：全部种子与19角', y=1.01)
        savefig(fig, 'groups_' + m)
    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
    for ax, m in zip(axes, ('si_sdr_db', 'pesq_wb', 'stoi', 'NMSE', 'RMS_gain')):
        decorate(ax)
        if m == 'RMS_gain':
            ax.axhline(1, color='#333', ls='--', lw=.8)
        for arm in ('A', 'B'):
            for seed in SEEDS:
                ax.plot(ANGLES, [next(p[m] for p in clean_profiles if p['system'] == f'{arm}_s{seed}' and p['angle'] == a)
                                 for a in ANGLES], color=colors[arm], lw=.7, alpha=.25)
            ax.plot(ANGLES, [statistics.fmean(p[m] for p in clean_profiles if p['arm'] == arm and p['angle'] == a)
                            for a in ANGLES], '-o', color=colors[arm], label=arm, ms=3)
        ax.set_ylabel(units[m])
    axes[0].set_title('全部19角干净控制；NMSE越低越好，RMS增益参照1'); axes[0].legend()
    axes[-1].set_xlabel('目标水平方位（度）'); savefig(fig, 'clean_fidelity_19_angles')
    write_report(summary, profiles, seed_summaries, means, seed_deltas, group_deltas,
                 clean_table, paired, run_results, content_shift)
    dump(OUT / 'SUMMARY_COMPLETE.json', {'status': 'passed', 'independent_max_error': max_error,
        'files': {str(p.relative_to(ROOT)): sha(p) for p in destination.iterdir() if p.is_file()}}, True)


def write_report(summary, profiles, seed_summaries, means, seed_deltas, group_deltas,
                 clean_table, paired, runs, content_shift):
    report = ROOT / 'docs/G35_五重点方位LoRA首轮匹配训练与19角开发评价报告.md'
    assert not report.exists(), 'Never overwrite an existing report'
    plan = meta(OUT / 'execution_manifest.json'); audit = meta(OUT / 'DELIVERY_AUDIT.json')
    lines = ['# G35 五重点方位LoRA首轮匹配训练与19角开发评价报告', '',
        '本轮依据用户明确执行授权，直接完成必要准入、A/B六次匹配训练与已有G35开发评价。没有另开待审核协议阶段，没有ASR、参数搜索、第二比例、G34恢复或其他候选训练。历史数据/代码/权重/报告保留。', '',
        '## 1. 实际完成与唯一研究变量', '',
        'A采用原G15采样；B采用五个互斥左闭右开邻域：[-45,-35)、[-5,5)、[5,15)、[25,35)、[65,75)度。B各重点300次、背景1500次，每种子3000次，三种子完整，未生成精确角音频或改标签。所有输入来自现有2048条G15正确几何池。', '',
        '两组均从官方McNet和同种子初始LoRA开始，不是续训旧适配器。freq/narr两处输出FC、rank=4、alpha=4、3584参数、原cIRM mask_loss、Adam学习率0.0003、batch1、梯度裁剪5、无clean辅助保持一致；FP32、两处TF32许可关闭，使用同一显式后端。每500步保存，统一登记3000末步，不按开发分数挑点或挑种子；未插入额外开发验证前向。', '',
        f'六次训练共18000次有效更新；开发输出及信号评价6384行（5472带噪、912干净），全部六模型×1064条已有G35输入。准入含一次训练前向/反向但零优化器更新，以及一次既有G35输入接口增强；不混入正式成绩。数据/源码/权重保护、全部输出哈希及覆盖核查通过，受保护历史文件{audit["protected_files"]}个。', '',
        '## 2. 采样公平性与实际资源', '',
        'B在CPU索引层用整数运输分配精确配额，尽量保持原采样：每个优化步与A匹配field、noise_id、SNR及point基准噪声角；multi8的angle不是物理方向，未当作单一噪声角。各目标组point数量按A实际point比例作整数分配，接近1:1。该离散分配目标仅用于采样，不进入神经损失或推理，不是新的降噪模块。', '',
        '九个训练数组/来源条件不变，数据不裁剪、不归一化、不替换。语句裁片、具体噪声段和multi8方向仍可能因索引变化而不同；不能把粗类别匹配说成逐条基础混合曝光完全相同。完整来源次数与变动见training_content_shift.csv。', '',
        '|组/种子|更新|秒|Torch峰值分配GiB|峰值保留GiB|point/multi8|不同索引|',
        '|---|---|---|---|---|---|---|']
    for r in runs:
        s = r['actual_sampling']
        lines.append(f'|{r["arm"]}/{r["seed"]}|3000|{r["seconds"]:.2f}|{r["peak_allocated_bytes"]/1024**3:.3f}|{r["peak_reserved_bytes"]/1024**3:.3f}|{s["field_counts"]["point"]}/{s["field_counts"]["multi8"]}|{s["unique_indices"]}|')
    lines += ['', '各A/B同种子初始主干、适配器摘要及CPU/CUDA RNG摘要完全一致，主干训练前后摘要不变。完整逐步损失、梯度范数、样本索引/角度/噪声/SNR、优化器与随机状态、GPU采样记录均保留。Torch峰值是真实分配器峰值，整卡显存仅作采样记录，不冒充连续整卡峰值或实时部署测量。', '',
              '|种子|A五重点合计|B五重点合计|B每重点/背景|源句计数变动(半L1)|说话人计数变动(半L1)|',
              '|---|---|---|---|---|---|']
    for seed in SEEDS:
        a = next(r for r in runs if r['arm'] == 'A' and r['seed'] == seed)['actual_sampling']
        source_change = sum(abs(r['B_minus_A']) for r in content_shift if r['seed'] == seed and r['attribute'] == 'source_counts') / 2
        speaker_change = sum(abs(r['B_minus_A']) for r in content_shift if r['seed'] == seed and r['attribute'] == 'speaker_counts') / 2
        lines.append(f'|{seed}|{3000-a["target_group_counts"]["background"]}|1500|300×5 / 1500|{source_change:g}|{speaker_change:g}|')
    lines += ['', 'field、噪声、SNR及point噪声类别全局计数和逐步类别匹配均核验；内容曝光变化如实保留，不能据此完全隔离物理角度因果。B重点占比由A原有约四成提高至五成，并不是从完全没有重点曝光开始。+20°位于背景，其实际邻近访问仍见完整采样记录，不称未见角插值。', '',
        '## 3. 完整19角信号结果', '',
        '仅带噪条件汇总，每角48条件/模型；三种子等权。Δ=B−A，正值表示B分数更高；STOI是原始0—1分数，非WER。', '',
        '![完整19角曲线](../artifacts/g35_multifocus_round1_v1/curves/three_metrics_19_angles.png)', '',
        '![同种子配对差](../artifacts/g35_multifocus_round1_v1/curves/paired_differences_19_angles.png)', '',
        '|组|指标|19角均值|最差值/角|最好值/角|角间极差|角间标准差|',
        '|---|---|---|---|---|---|---|']
    for r in summary:
        lines.append(f'|{r["arm"]}|{r["metric"]}|{r["mean_19"]:.4f}|{r["worst"]:.4f}/{r["worst_angles"]}°|{r["best"]:.4f}/{r["best_angles"]}°|{r["range"]:.4f}|{r["angle_std_population"]:.4f}|')
    lines += ['', '最差方向均值不是逐例最低值；角间标准差不是样本不确定性。', '',
              '|角度|A SI-SDR/dB|B SI-SDR/dB|ΔSI/dB|ΔPESQ|ΔSTOI|', '|---|---|---|---|---|---|']
    for a in ANGLES:
        aa, bb = means['A', a], means['B', a]
        lines.append(f'|{a:+d}°|{aa["si_sdr_db"]:.4f}|{bb["si_sdr_db"]:.4f}|{bb["si_sdr_db"]-aa["si_sdr_db"]:+.4f}|{bb["pesq_wb"]-aa["pesq_wb"]:+.4f}|{bb["stoi"]-aa["stoi"]:+.4f}|')
    focus = {-40, 0, 10, 30, 70}
    lines += ['', '|观察集合|角数|平均ΔSI/dB|平均ΔPESQ|平均ΔSTOI|', '|---|---|---|---|---|']
    for label, aa in [('五重点中心', sorted(focus)), ('其余14角', [a for a in ANGLES if a not in focus]),
                      ('历史优势+20°', [20]), ('左侧范围外', [-90, -80, -70, -60, -50]), ('右侧范围外', [80, 90])]:
        d = {m: statistics.fmean(means['B', a][m] - means['A', a][m] for a in aa) for m in METRICS}
        lines.append(f'|{label}|{len(aa)}|{d["si_sdr_db"]:+.4f}|{d["pesq_wb"]:+.4f}|{d["stoi"]:+.4f}|')
    lines += ['', '重点+10/+30的恢复、+20保持、−40/+70收益和−90/+90边缘风险均以全表为准，不因属于重点而忽略负值。历史单方位曲线是同一G35开发输入上的既有参照，但训练数值历史与本轮新匹配A/B不同，不把跨轮差异计作新方案的唯一因果收益。', '',
              '## 4. 全部三个种子', '', '|组/种子|平均SI/dB|最差SI/角|平均PESQ|平均STOI|', '|---|---|---|---|---|']
    for arm in ('A', 'B'):
        for seed in SEEDS:
            ss = [r for r in seed_summaries if r['system'] == f'{arm}_s{seed}']
            d = {r['metric']: r for r in ss}
            lines.append(f'|{arm}/{seed}|{d["si_sdr_db"]["mean_19"]:.4f}|{d["si_sdr_db"]["worst"]:.4f}/{d["si_sdr_db"]["worst_angle"]}°|{d["pesq_wb"]["mean_19"]:.4f}|{d["stoi"]["mean_19"]:.4f}|')
    lines += ['', '|角度|ΔSI种子20261112|20261113|20261114|PESQ正收益种子|STOI正收益种子|', '|---|---|---|---|---|---|']
    for a in ANGLES:
        ds = [next(r for r in seed_deltas if r['seed'] == s and r['angle'] == a) for s in SEEDS]
        lines.append('|' + '|'.join([f'{a:+d}°', *[f'{d["si_sdr_db"]:+.4f}' for d in ds],
                     f'{sum(d["pesq_wb"]>0 for d in ds)}/3', f'{sum(d["stoi"]>0 for d in ds)}/3']) + '|')
    lines += ['', '每个种子每角三项完整数值见per_seed_19_angles.csv及per_seed_19_differences.csv，不以某种子或正收益计数代替全量结果。', '',
              '## 5. 六种噪声/空间组合及逐例退化', '',
              '|噪声/场型/相对角|平均ΔSI/dB|平均ΔPESQ|平均ΔSTOI|SI退化角数|PESQ退化角数|STOI退化角数|',
              '|---|---|---|---|---|---|---|']
    for key, v in sorted(groups(group_deltas, lambda r: (r['noise'], r['field'], r['relative'])).items()):
        d = {m: statistics.fmean(r[m] for r in v) for m in METRICS}
        lines.append('|' + '/'.join(key) + '|' + '|'.join([f'{d[m]:+.4f}' for m in METRICS] +
                     [str(sum(r[m] < 0 for r in v)) + '/19' for m in METRICS]) + '|')
    for m in METRICS:
        lines += ['', f'![六组{m}](../artifacts/g35_multifocus_round1_v1/curves/groups_{m}.png)']
    lines += ['', '|指标|逐例B−A>0|逐例B−A<0|最不利差|对应种子/样本|', '|---|---|---|---|---|']
    for m in METRICS:
        worst = min(paired, key=lambda r: r[m])
        lines.append(f'|{m}|{sum(r[m]>0 for r in paired)}/2736|{sum(r[m]<0 for r in paired)}/2736|{worst[m]:.4f}|{worst["seed"]}/{worst["sample_id"]}|')
    lines += ['', '全部114个“角×噪声/场型”差值与684个逐模型分组均值另存CSV，所有不利记录保留。每组每角只有8条源语句；条件、种子、角不是独立说话人样本，不作总体显著性声称。', '',
              '## 6. 干净语音保真', '',
              '|组/种子|SI/dB|PESQ|STOI|NMSE|RMS增益|', '|---|---|---|---|---|---|']
    for r in clean_table:
        lines.append(f'|{r["system"]}|{r["si_sdr_db"]:.4f}|{r["pesq_wb"]:.4f}|{r["stoi"]:.4f}|{r["NMSE"]:.6f}|{r["RMS_gain"]:.6f}|')
    lines += ['', '![全角干净控制](../artifacts/g35_multifocus_round1_v1/curves/clean_fidelity_19_angles.png)', '',
              'NMSE=Σ(y−s)²/Σs²，RMS增益=√(Σy²/Σs²)，对保存后FLOAT输出与共同参考计算；越低NMSE、越接近1的幅度是不同参照，不用尺度不敏感的SI替代。每模型152干净条件，全19角逐种子数据见clean_per_seed_19_angles.csv。不能把能量变化直接等同吞字或听感结论。', '',
              '## 7. 是否值得继续研究与边界', '']
    focus_d = {m: statistics.fmean(means['B', a][m] - means['A', a][m] for a in sorted(focus)) for m in METRICS}
    clean_d = {m: statistics.fmean(r[m] for r in clean_table if r['arm'] == 'B') -
                  statistics.fmean(r[m] for r in clean_table if r['arm'] == 'A') for m in ('NMSE', 'RMS_gain')}
    positive = {m: sum(means['B', a][m] > means['A', a][m] for a in ANGLES) for m in METRICS}
    lines.append(f'本固定方案的五重点中心平均差为SI {focus_d["si_sdr_db"]:+.4f} dB、PESQ {focus_d["pesq_wb"]:+.4f}、STOI {focus_d["stoi"]:+.4f}；全19角正收益点数为{positive["si_sdr_db"]}/{positive["pesq_wb"]}/{positive["stoi"]}（依次SI/PESQ/STOI，分母均19）。干净NMSE均值变化{clean_d["NMSE"]:+.6f}，RMS增益均值变化{clean_d["RMS_gain"]:+.6f}。')
    lines += ['', '这只是一次固定曝光设计的开发性读回，未预登记临床意义/非劣效数值门槛，不追认某个均值为“成功阈值”。是否值得继续，要同时看重点恢复、+20和非重点代价、六组/三种子一致性及干净保真；不能只报重点平均。任何局部收益都不证明连续方位、真实阵列、新噪声/房间或最终独立泛化。', '',
              '本轮没有新ASR/试听证据，不能把SI/PESQ/STOI变化称作WER或吞字恢复。梯度冲突、容量不足或曝光因果机制未在本轮直接测量。完整负结果交用户审核，完成后停止，不自动试第二比例、继续训练或开新方向。', '',
              '## 8. 来源、完整产物与复现', '',
              '执行manifest与代码哈希在训练前写入；新目录独立且拒绝已有路径，不修改历史注册表。A/B初始摘要、逐步日志、采样表、全部500步快照、最终优化器/随机状态、输入/参考/输出SHA及资源记录均保留。所有模型使用同一G35输入和评分代码，未增益调整、对齐、裁剪或额外滤波。', '',
              '[执行清单](../artifacts/g35_multifocus_round1_v1/execution_manifest.json) · [采样检查](../artifacts/g35_multifocus_round1_v1/sampler_checks.json) · [GPU准入](../artifacts/g35_multifocus_round1_v1/preflight.json) · [六次训练](../artifacts/g35_multifocus_round1_v1/TRAIN_COMPLETE.json) · [全部逐例信号](../artifacts/g35_multifocus_round1_v1/signals.jsonl) · [交付审计](../artifacts/g35_multifocus_round1_v1/DELIVERY_AUDIT.json)', '',
              '[19角CSV](../artifacts/g35_multifocus_round1_v1/curves/mean_19_angles.csv) · [各随机种子](../artifacts/g35_multifocus_round1_v1/curves/per_seed_19_angles.csv) · [全部配对差](../artifacts/g35_multifocus_round1_v1/curves/all_paired_differences.csv) · [分组](../artifacts/g35_multifocus_round1_v1/curves/noise_field_groups.csv) · [干净](../artifacts/g35_multifocus_round1_v1/curves/clean_controls.csv) · [训练内容曝光变化](../artifacts/g35_multifocus_round1_v1/curves/training_content_shift.csv)', '',
              '全部图同时保存300 dpi PNG、PDF、SVG。连接线只助阅读，不平滑、不表示连续角保证。数据仍是旧G17八句/四人的已查看开发集；最终确认和新ASR仍需单独批准。']
    with report.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    dump(OUT / 'continuation_facts.json', {'focus_center_differences': focus_d,
        'positive_angle_counts': positive, 'clean_B_minus_A': clean_d,
        'no_preregistered_performance_success_threshold': True,
        'report_sha256_at_generation': sha(report)}, True)
