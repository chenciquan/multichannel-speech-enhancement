"""Display fresh exact-gain hypothesis changes without implying content deletion."""
import os
os.environ['MPLBACKEND']='Agg'
import json
import numpy as np
import matplotlib.pyplot as plt
from g24_prepare import ROOT,OUT,sha,archive


def main():
    rec=OUT/'recognition'
    assert json.loads((OUT/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    audit=json.loads((rec/'completion_audit.json').read_text())
    assert sha(rec/'summary.json')==audit['summary_sha256']
    summary=json.loads((rec/'summary.json').read_text())
    families=('clean_reference','noisy','rigid_mvdr','rigid_mwf','truth_covariance_mvdr','truth_covariance_mwf')
    labels=('Clean\nreference','Noisy','Rigid\nMVDR','Rigid\nMWF','True-cov\nMVDR','True-cov\nMWF')
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
    for ax,engine in zip(axes,('Whisper','CTC')):
        for offset,observation,color,label in ((-.18,'half','#0072B2','Exact gain 1/2'),(.18,'quarter','#D55E00','Exact gain 1/4')):
            values=[next(r for r in summary['gain_changes'] if r['engine']==engine and r['family']==family and r['observation']==observation) for family in families]
            ax.bar(np.arange(6)+offset,[100*r['normalized_changes']/r['pairs'] for r in values],width=.34,color=color,label=label)
        repeat=next(r for r in summary['repeatability'] if r['engine']==engine)
        ax.set_title(f"{engine}: original cross-run / within-run changes = {repeat['cross_run_normalized_changes']} / {repeat['within_run_normalized_changes']}",fontsize=9)
        ax.set_xticks(range(6),labels,fontsize=8);ax.set_ylim(0,100)
        ax.set_ylabel('Normalized hypothesis changes (%)');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('G24: independently decoded, exactly reversible gain controls\nChanges are versus the fresh original; repeated conditions are not independent utterances',fontsize=11)
    png=rec/'exact_gain_hypothesis_changes.png';pdf=rec/'exact_gain_hypothesis_changes.pdf'
    assert not png.exists() and not pdf.exists();fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig)
    report=ROOT/'docs/G24_精确增益识别变化图说明.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G24 精确增益识别变化图\n\n![独立精确增益控制](../artifacts/g24_gain_repeat_v1/recognition/exact_gain_hypothesis_changes.png)\n\n图由已验收结果绘制，纵轴为规范化假设文本相对本次原幅度发生变化的系统/条件对比例。每个刚性适配组144对、其余48对，均重复使用四人四句场景，不是独立语句的大样本概率。\n\n原幅度跨次和同次变化均为0；半幅/四分之一幅可逐样本精确反向还原。文本变化率不等于WER或人声丢失率，净错误变化可能因不同错误相互抵消而很小。真值协方差及干净参考仅作诊断。完整WER、S/D/I见阶段报告，不从图中选择最佳增益或宣称增强模型更好。\n')
    archive('gain_visualization',report,[png,pdf,ROOT/'scripts/g24_plot.py'])


if __name__=='__main__':main()
