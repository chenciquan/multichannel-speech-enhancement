"""Publication-exportable view of the two oracle diagnostic filters."""
import os
os.environ['MPLBACKEND']='Agg'
import json
import matplotlib.pyplot as plt
from g21_truth_covariance import ROOT,OUT,sha,archive


def main():
    rec=OUT/'recognition';source=rec/'summary.json'
    assert sha(source)==json.loads((rec/'completion_audit.json').read_text())['summary_sha256']
    rows=json.loads(source.read_text())['by_rt']
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.7),layout='constrained')
    metrics=('si_sdr_db','Whisper','CTC')
    for method,color in (('mvdr','#0072B2'),('mwf','#D55E00')):
        selected=sorted([r for r in rows if r['group']=='truth_covariance' and r['method']==method],key=lambda r:r['rt'])
        for ax,key in zip(axes,metrics):
            values=[r[key] if key=='si_sdr_db' else 100*r[key]['wer'] for r in selected]
            ax.plot(range(3),values,color=color,marker='o',label=method.upper())
    for ax,engine in zip(axes[1:],metrics[1:]):
        reference=json.loads((rec/engine/'reference_summary.json').read_text())
        ax.plot(range(3),[100*reference[str(rt)]['wer'] for rt in (0,.15,.35)],color='#555555',linestyle='--',label='Clean room reference')
    for ax,title,label in zip(axes,('Signal similarity','Whisper tiny.en','Greedy CTC'),('SI-SDR (dB)','WER (%)','WER (%)')):
        ax.set_title(title);ax.set_ylabel(label);ax.set_xlabel('Requested RT (s)')
        ax.set_xticks(range(3),['0','0.15','0.35']);ax.grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('G21: true-source covariance diagnostics — not deployable or a theoretical bound',fontsize=11)
    png=rec/'truth_covariance_quality_and_WER.png';pdf=rec/'truth_covariance_quality_and_WER.pdf'
    assert not png.exists() and not pdf.exists();fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig)
    report=ROOT/'docs/G21_真值统计质量与识别图说明.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G21 真值统计质量与识别图\n\n![真值统计诊断](../artifacts/g21_truth_covariance_v1/recognition/truth_covariance_quality_and_WER.png)\n\n由已验收汇总绘制，未重新计算输出或ASR。仅比较两种使用真实分量协方差的诊断滤波器，虚线是同房间混响干净参考的ASR，不是带噪输入。请求RT不等于实测RT。四人八句重复条件，无置信区间或显著性声称。\n\n请求RT=.35时，MWF的SI-SDR更高，但两个ASR的WER均高于MVDR。这不能证明特定词语被删除，也不能将真值系统视为可部署方法或理论性能上界。全部盲系统及三种子结果仍以综合报告为准。\n')
    archive('quality_recognition_visualization',report,[png,pdf,ROOT/'scripts/g21_plot.py'])


if __name__=='__main__':main()
