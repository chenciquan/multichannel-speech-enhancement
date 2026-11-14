"""Layout-only figures from completed CSVs; no model, inference, or scoring."""
import csv
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/g35_multifocus_round1_v1/curves'
SEEDS = (20261112, 20261113, 20261114)
ANGLES = list(range(-90, 91, 10))
METRICS = ('si_sdr_db', 'pesq_wb', 'stoi')
LABELS = ('SI-SDR (dB)', 'PESQ-WB', 'STOI')
COLORS = {'A': '#2569ad', 'B': '#df6b20'}


def read(name):
    with (DEST / name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def decorate(ax):
    ax.axvspan(-45, 75, color='#b9c8bd', alpha=.12)
    for center in (-40, 0, 10, 30, 70):
        ax.axvspan(center - 5, center + 5, color=COLORS['B'], alpha=.10)
    ax.axvline(20, color='#666', ls=':', lw=1)
    ax.set_xlim(-90, 90); ax.set_xticks(ANGLES)
    ax.grid(alpha=.2); ax.tick_params(axis='x', labelsize=7)


def save(fig, name):
    fig.tight_layout(rect=(0, 0, 1, .96))
    for ext in ('png', 'pdf', 'svg'):
        p = DEST / f'{name}.{ext}'
        assert not p.exists(), 'No figure replacement'
        fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name(),
        'axes.unicode_minus': False, 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    rows = read('noise_field_groups.csv')
    assert len(rows) == 684
    spaces = [('point', '-80', '点噪声，相对角 −80°'),
              ('point', '25', '点噪声，相对角 +25°'),
              ('multi8', 'None', 'multi8，八方向噪声')]
    for metric, label in zip(METRICS, LABELS):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        for i, noise in enumerate(('OOFFICE', 'NPARK')):
            for j, (field, relative, title) in enumerate(spaces):
                ax = axes[i, j]; decorate(ax)
                for arm in ('A', 'B'):
                    values = []
                    for seed in SEEDS:
                        v = [float(next(r[metric] for r in rows if r['system'] == f'{arm}_s{seed}'
                              and int(r['angle']) == a and r['noise'] == noise and r['field'] == field
                              and r['relative'] == relative)) for a in ANGLES]
                        values.append(v); ax.plot(ANGLES, v, color=COLORS[arm], alpha=.2, lw=.7)
                    ax.plot(ANGLES, [statistics.fmean(v[k] for v in values) for k in range(19)],
                            '-o', color=COLORS[arm], ms=2, label=arm)
                ax.set_title(f'{noise} / {title}'); ax.set_ylabel(label)
                if i == 1:
                    ax.set_xlabel('目标水平方位（度）')
        axes[0, 0].legend()
        fig.suptitle('六种噪声/空间组合：完整19角与全部三个种子（粗线为均值）', y=.995)
        save(fig, 'groups_' + metric + '_layout_r1')
    rows = read('per_seed_19_angles.csv')
    assert len(rows) == 114
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True, sharey='row')
    for j, seed in enumerate(SEEDS):
        for i, (metric, label) in enumerate(zip(METRICS, LABELS)):
            ax = axes[i, j]; decorate(ax)
            for arm in ('A', 'B'):
                v = [float(next(r[metric] for r in rows if r['system'] == f'{arm}_s{seed}' and int(r['angle']) == a)) for a in ANGLES]
                ax.plot(ANGLES, v, '-o', color=COLORS[arm], ms=2.5, lw=1.6, label=arm)
            ax.set_ylabel(label)
            if i == 0:
                ax.set_title(f'种子 {seed}')
            if i == 2:
                ax.set_xlabel('目标水平方位（度）')
    axes[0, 0].legend()
    fig.suptitle('六份新模型逐种子比较：A原采样，B五重点曝光；每列同一初始种子', y=.995)
    save(fig, 'three_seeds_separate')
    print('CSV-only layout review complete; no model/scoring calls')


if __name__ == '__main__':
    main()
