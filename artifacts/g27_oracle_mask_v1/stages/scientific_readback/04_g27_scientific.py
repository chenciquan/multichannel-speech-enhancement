"""Bounded scientific readback, with no parameter selection from diagnostics."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from g27_signal import ROOT,OUT,sha,dump,now,meta,archive


def main():
    assert meta(OUT/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    data=meta(OUT/'recognition/summary.json')
    by={(r['room'],r['rt'],r['method'],r['group']):r for r in data['systems']}
    pairs=[]
    for room in ('reference','new'):
        for rt in (0,.15,.35):
            for method in ('mvdr','mwf'):
                mask=by[room,rt,method,'truth_ref_power_mask'];rigid=by[room,rt,method,'rigid'];cov=by[room,rt,method,'truth_covariance']
                row={'room':room,'rt':rt,'method':method,'mask_minus_rigid_SI':mask['si_sdr_db']-rigid['si_sdr_db'],
                     'mask_minus_cov_SI':mask['si_sdr_db']-cov['si_sdr_db']}
                for e in ('Whisper','CTC'):
                    for regime in ('raw','level_control'):
                        row[f'{e}_{regime}_mask_minus_rigid_WER_pp']=100*(mask[e][regime]['wer']-rigid[e][regime]['wer'])
                        row[f'{e}_{regime}_mask_minus_cov_WER_pp']=100*(mask[e][regime]['wer']-cov[e][regime]['wer'])
                pairs.append(row)
    joint=sum(r['mask_minus_rigid_SI']>0 and all(r[f'{e}_{regime}_mask_minus_rigid_WER_pp']<0 for e in ('Whisper','CTC') for regime in ('raw','level_control')) for r in pairs)
    contradictions=[r for r in pairs if r['mask_minus_cov_SI']>0 and all(r[f'{e}_{regime}_mask_minus_cov_WER_pp']>0 for e in ('Whisper','CTC') for regime in ('raw','level_control'))]
    dump(OUT/'recognition/scientific_readback.json',{'utc':now(),'comparisons':pairs,'joint_better_than_rigid_cells':joint,
         'higher_SI_but_all_WER_worse_than_covariance':contradictions,'summary_sha256':sha(OUT/'recognition/summary.json')},True)
    fig,axes=plt.subplots(3,2,figsize=(11,8.5),sharex=True)
    labels=['Ref\n0','Ref\n.15','Ref\n.35','New\n0','New\n.15','New\n.35']
    grid=[(room,rt) for room in ('reference','new') for rt in (0,.15,.35)]
    colors={'mcnet':'#888888','rigid':'#265f8d','truth_ref_power_mask':'#bd7823','truth_covariance':'#39734d'}
    for j,method in enumerate(('mvdr','mwf')):
        for group,color in colors.items():
            values=[by[room,rt,method,group] for room,rt in grid]
            name={'mcnet':'McNet mask','rigid':'Rigid LoRA mask (3 seeds)','truth_ref_power_mask':'True power mask *','truth_covariance':'True covariance *'}[group]
            axes[0,j].plot(range(6),[r['si_sdr_db'] for r in values],marker='o',color=color,label=name)
            for i,e in enumerate(('Whisper','CTC'),1):axes[i,j].plot(range(6),[100*r[e]['level_control']['wer'] for r in values],marker='o',color=color)
        axes[0,j].set_title(method.upper())
        for i in range(3):
            axes[i,j].grid(alpha=.2);axes[i,j].set_xticks(range(6),labels)
        axes[2,j].set_xlabel('Room / requested RT (s), not measured T60')
    for i,label in enumerate(('SI-SDR (dB)','Whisper WER (%)\nfixed level-control','CTC WER (%)\nfixed level-control')):axes[i,0].set_ylabel(label)
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,fontsize=9)
    fig.suptitle('Oracle-mask diagnostic: four reused utterances, dependent conditions\n* Uses true speech; unavailable at deployment; no upper-bound claim',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.875))
    for ext in ('png','pdf'):fig.savefig(OUT/f'recognition/oracle_mask_diagnostic.{ext}',dpi=180)
    plt.close(fig)
    report=ROOT/'docs/G27_科学结论与训练可行性门槛.md'
    lines=['# G27 科学结论与训练可行性门槛','',
        f'相对正确刚性组三种子的平均结果，真值功率比掩码在12个房间×RT×滤波条件中，有{joint}个同时提高SI-SDR，并降低两个识别器raw与电平控制的WER。这些是同四句的依赖条件，不能当12个独立重复；真值系统不可部署，不能当作所设计模型的效果。', '',
        '例如新房间请求RT=.35的MVDR：刚性组SI-SDR=1.04 dB、Whisper/CTC电平控制WER=80.37%/79.19%；真值功率比掩码为6.70 dB、46.89%/43.11%。说明该具体掩码构造仍有可利用空间，但即便使用真值，强反射识别仍不理想。', '',
        f'相对真值协方差，有{len(contradictions)}个条件出现SI-SDR较高、而两个识别器两种电平WER全部较差，条件明细已保存。新房间请求RT=.35的MVDR就是一例：真值协方差SI-SDR=3.83 dB，电平控制WER=41.33%/38.22%，比真值掩码的识别结果更好。不能以SI-SDR或协方差形状误差代替内容保真评价，也不能把真值协方差称为各指标严格上界。', '',
        '## 下一项最小实验', '',
        '先做可微路径和显存预检，不更新任何模型参数：仅从既有G15训练池按场型与角度固定选择六个训练记录，对冻结主干/同位置LoRA比较原压缩cIRM损失、真值功率比掩码MSE、真值掩码所产生协方差的形状损失。各损失分别反向，记录LoRA梯度是否有限、梯度方向关系、实际显存与耗时；核对主干及适配器参数完全未变。检查不读G22/G27测试音频。', '',
        '此处第三项的标签是“真值功率比掩码作用于混合信号所得到的统计”，不是六通道真实语音分量协方差，两者必须区分。其空间统计监督思想已有先例，不认定原创。预检也不能证明优化有效或泛化。', '',
        '路径可用后，再登记独立开发条件下的有限训练比较：相同训练混响增强、相同LoRA预算、相同更新次数和种子，保留原cIRM目标与简单掩码MSE作为成熟强对照，检验统计目标是否带来额外收益。不能仅与未适配McNet比较。损失系数与停止规则必须在开发集登记；G22/G27旧四句不再用于选系数，之后的新确认必须排除已有训练/验证/已反复查看的说话人或明确降低泛化声明。', '',
        '目前最大的风险仍是：标签改善未必能由3584个适配参数实现；信号损失改善未必降低WER；训练/开发语音规模有限且模拟域与真实阵列存在差距。保持失败分支与完整对照，不预设必胜方法或论文发表。', '',
        '![信号与电平控制识别对照](../artifacts/g27_oracle_mask_v1/recognition/oracle_mask_diagnostic.png)', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('scientific_readback',report,[OUT/'recognition/scientific_readback.json',OUT/'recognition/oracle_mask_diagnostic.png',OUT/'recognition/oracle_mask_diagnostic.pdf',ROOT/'scripts/g27_scientific.py'])
    print({'joint_cells':joint,'SI_WER_contradiction_cells':len(contradictions)})


if __name__=='__main__':main()
