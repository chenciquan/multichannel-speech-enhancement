"""Plot the matched-room difference for all frozen MVDR frontend groups."""
import os
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
from g22_common import ROOT,OUT,sha,metadata,archive


def main():
    rec=OUT/'recognition';done=metadata(rec/'completion_audit.json')
    assert metadata(OUT/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    assert sha(rec/'summary.json')==done['summary_sha256']
    summary=metadata(rec/'summary.json')
    categories=(('mcnet','#555555'),('fixed','#D55E00'),('rotated','#0072B2'),('rigid','#009E73'),('truth_covariance','#CC79A7'))
    labels={'mcnet':'McNet','fixed':'Fixed training','rotated':'Counter training','rigid':'Rigid training','truth_covariance':'True covariance (diagnostic)'}
    fig,axes=plt.subplots(1,3,figsize=(12.2,3.9),layout='constrained')
    metric_labels=(('si_sdr_db','SI-SDR change (dB)'),('Whisper_WER_pp','Whisper WER change (pp)'),('CTC_WER_pp','CTC WER change (pp)'))
    systems=summary['by_system'];rt_values=(0,.15,.35)
    for ax,(metric,ylab) in zip(axes,metric_labels):
        for group,color in categories:
            rows=sorted([r for r in summary['new_minus_reference'] if r['group']==group and r['method']=='mvdr' and not r['clean']],key=lambda r:r['rt'])
            assert len(rows)==3
            if group in ('fixed','rotated','rigid'):
                frontend_names=sorted({r['system'] for r in systems if r['system'].startswith(group+'_without_clean_identity_') and r['system'].endswith('_mask_mvdr')})
                assert len(frontend_names)==3
                for name in frontend_names:
                    indexed={(r['room'],r['rt']):r for r in systems if r['system']==name and not r['clean']}
                    def difference(rt):
                        a,b=indexed['reference',rt],indexed['new',rt]
                        if metric=='si_sdr_db':return b[metric]-a[metric]
                        engine=metric.split('_')[0];return 100*(b[engine]['wer']-a[engine]['wer'])
                    ax.plot(range(3),[difference(rt) for rt in rt_values],color=color,alpha=.3,linewidth=.8)
            ax.plot(range(3),[r[metric] for r in rows],color=color,marker='o',linewidth=1.8,label=labels[group])
        ax.axhline(0,color='black',linestyle='--',linewidth=.8)
        ax.set_xticks(range(3),['0','0.15','0.35']);ax.set_xlabel('Requested RT (s)');ax.set_ylabel(ylab);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('G22: new room minus matched reference room — MVDR outputs\nThin lines: individual adapter seeds; bold lines: complete group summaries',fontsize=11)
    png=rec/'matched_room_MVDR_deltas.png';pdf=rec/'matched_room_MVDR_deltas.pdf'
    assert not png.exists() and not pdf.exists();fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig)
    report=ROOT/'docs/G22_房间配对差值图说明.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G22 房间配对差值图\n\n![MVDR房间差值](../artifacts/g22_matched_rooms_v1/recognition/matched_room_MVDR_deltas.png)\n\n本图限定MVDR输出，展示新房间减匹配原房间；全部直接输出和MWF结果仍见综合报告。三个适配组保留全部种子，细线显示单种子，粗线为完整组差值。真值协方差曲线仅为诊断，不是部署方法或理论上界。\n\nSI-SDR差为正、WER差为负时，分别表示该指标在新房间更好。WER是总错误数/总参考词数，单位百分点。请求RT不是实测T20；房间改变同时涉及反射统计，不能分离为某个尺寸的独立作用。四人四句及重复条件不当大样本，本图无置信区间、显著性或广泛泛化声称。PDF供后续排版。\n')
    archive('room_delta_visualization',report,[png,pdf,ROOT/'scripts/g22_plot.py'])


if __name__=='__main__':main()
