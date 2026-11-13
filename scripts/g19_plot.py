"""Plot archived paired filter differences, with all training seeds visible."""
import os
os.environ['MPLBACKEND']='Agg'
import json
import matplotlib.pyplot as plt
import numpy as np
from g19_common import ROOT, OUT, sha, archive


def main():
    source=OUT/'evaluation/summary.json'
    audit=json.loads((OUT/'completion_audit.json').read_text())
    assert sha(source)==audit['summary_sha256']
    summary=json.loads(source.read_text())
    groups=('mcnet','fixed','counter','rigid')
    colors=('#555555','#D55E00','#0072B2','#009E73')
    columns=(('si_sdr_db','SI-SDR difference (dB)'),('Whisper_wer_pp','Whisper WER difference (pp)'),('CTC_wer_pp','CTC WER difference (pp)'))
    fig, axes=plt.subplots(1,3,figsize=(12.4,3.8),layout='constrained')
    for ax,(metric,label) in zip(axes,columns):
        for group,color in zip(groups,colors):
            per=[r for r in summary['per_seed'] if r['group']==group]
            for frontend in sorted({r['frontend'] for r in per}):
                selected=sorted([r for r in per if r['frontend']==frontend],key=lambda r:r['rt'])
                ax.plot(range(3),[r['mwf_minus_mvdr'][metric] for r in selected],color=color,alpha=.3,linewidth=.8)
            means=sorted([r for r in summary['by_rt'] if r['group']==group],key=lambda r:r['rt'])
            ax.plot(range(3),[r['mwf_minus_mvdr'][metric] for r in means],color=color,marker='o',label=group,linewidth=1.8)
        ax.axhline(0,color='black',linewidth=.8,linestyle='--')
        ax.set_xticks(range(3),['0','0.15','0.35']);ax.set_xlabel('Requested RT (s)');ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('G19: MWF minus MVDR — same frontend and room input\nThin lines: individual seeds; bold: grouped paired difference',fontsize=11)
    png=OUT/'evaluation/filter_deltas.png';pdf=OUT/'evaluation/filter_deltas.pdf'
    assert not png.exists() and not pdf.exists()
    fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig)
    report=ROOT/'docs/G19_配对差值图说明.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G19 配对差值图\n\n![固定滤波对照](../artifacts/g19_room_mwf_v1/evaluation/filter_deltas.png)\n\n图从已验收汇总读取，不重算增强或识别。纵轴均为 MWF 减 MVDR；SI-SDR 正值更好，两个 WER 差负值更好。细线显示各训练种子，粗线显示组内配对差；原始 McNet 仅一个前端。横轴为请求 RT，不是各位置实测衰减时间。\n\n重复语句和训练种子并非独立样本，图不提供置信区间或显著性结论。滤波估计与求解同时有差别。固定μ=1未显示整体优势，不依据此图调参。PDF可用于后续论文排版。\n')
    archive('paired_visualization',report,[png,pdf,ROOT/'scripts/g19_plot.py'])


if __name__=='__main__':main()
