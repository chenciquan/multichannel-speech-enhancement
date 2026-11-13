"""Read back the failed primary gate without changing it or selecting a new winner."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from g31_common import ROOT,OUT,sha,read,dump,now,meta,archive


def main():
    assert meta(OUT/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    decision=meta(OUT/'primary_decision.json');summary=meta(OUT/'summary.json')
    assert decision['extension_gate_passed'] is False
    by={(r['source_system'],r['method'],r['requested_rt60'],r['clean']):r for r in summary['by_rt']}
    overall={(r['source_system'],r['method'],r['clean']):r for r in summary['overall']}
    fig,axes=plt.subplots(2,2,figsize=(10,7));labels=['0','.16','.28','All'];x=np.arange(4)
    panels=[('si_sdr_db','SI-SDR delta (dB)',False),('pesq_wb','PESQ delta',False),
            ('stoi','STOI delta',False),('reference_normalized_squared_error','Clean NMSE increase',True)]
    plotted=[]
    for ax,(metric,title,clean) in zip(axes.flat,panels):
        for ci,comparator in enumerate(('cirm','cirm_mask')):
            values=[]
            for rt in (0,.16,.28,None):
                a=overall['cirm_shape','mvdr',clean] if rt is None else by['cirm_shape','mvdr',rt,clean]
                b=overall[comparator,'mvdr',clean] if rt is None else by[comparator,'mvdr',rt,clean]
                value=a['clean_levels'][metric]-b['clean_levels'][metric] if clean else a[metric]-b[metric]
                values.append(value);plotted.append({'metric':metric,'comparator':comparator,'rt':rt,'delta':value})
            ax.bar(x+(-.17 if ci==0 else .17),values,width=.32,label='Shape minus '+('cIRM' if ci==0 else 'simple mask'),color=('#2b6b91','#c27a31')[ci])
        ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(x,labels);ax.set_xlabel('Requested RT (s) / all conditions');ax.set_title(title);ax.grid(axis='y',alpha=.2)
        if metric=='si_sdr_db':ax.scatter([3],[.25],marker='_',s=200,color='black',zorder=6);ax.annotate('Overall gate .25',xy=(3,.25),xytext=(1.65,.27),fontsize=8)
        if clean:ax.scatter([3],[.02],marker='_',s=200,color='black',zorder=6);ax.annotate('Overall cap .02',xy=(3,.02),xytext=(1.4,.019),fontsize=8)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Primary MVDR development comparison: extension gate NOT passed\n12 reused utterances / 2 speakers / 1 seed; thresholds apply to All only',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9))
    for ext in ('png','pdf'):fig.savefig(OUT/f'primary_gate_deltas.{ext}',dpi=180)
    plt.close(fig)
    dump(OUT/'plotted_primary_deltas.json',plotted,True)
    sources=read(ROOT/'data/g3_v1/sources/speech_manifest.jsonl');selected=[]
    for speaker in sorted({r['speaker_id'] for r in sources if r['split']=='validation'}):
        eligible=[r for r in sources if r['split']=='validation' and r['speaker_id']==speaker and r['samples']<=56320]
        assert len(eligible)>=2;selected.extend(eligible[:2])
    assert len(selected)==4
    dump(OUT/'next_complete_development_sources.json',selected,True)
    report=ROOT/'docs/G31_科学读回与完整开发语句补充诊断决策.md'
    lines=['# G31 科学读回与完整开发语句补充诊断决策','','G31技术执行与验收通过，预登记的统计辅助扩展门槛未通过。两者要区分：不是程序失败，也不能把小幅正变化称作已证实总体更优。', '',
        '统计辅助MVDR相对原重建组：SI-SDR +0.53333 dB、PESQ +0.00067、STOI +0.01442，两位开发说话人的SI变化均为正。相对简单掩码强对照：SI-SDR +0.23827 dB（未达事前.25门槛）、PESQ −0.00901、STOI +0.00434。两项比较的干净NMSE约束均通过。', '',
        '不能因为SI差距接近门槛就临时降低门槛；也不能把这些差值当成稳定或显著的总体效应。只有一个种子、两个开发说话人，PESQ小差距及方向差异都需要不确定性解释。统计目标并未证明优于已有简单监督，因此暂不扩三种子或加大统计辅助训练。', '',
        '另一个观察是，干净输入经MVDR后平均RMS增益约.53，NMSE约.28；MWF则RMS较接近1、NMSE较小。这是保存波形的幅度/失真现象，不等于吞字或人耳不可懂，也不能由SI-SDR直接解释。原有识别器对输入电平敏感的证据仍有效。', '',
        '## 下一项限定为完整语句的补充诊断', '',
        '当前开发裁剪没有有效整句WER标签。为获得有意义的识别对照，可从同一旧开发角色每位说话人按原清单顺序选前两条长度≤56320样本的完整语句，共四条；加7680样本RIR尾后仍≤4秒。来源已按长度和角色冻结，尚未生成其新完整混响波形或读取其新输出评分。该选择不是新的独立说话人测试。', '',
        '|完整开发语句|说话人|原始样本数|','|---|---|---:|']
    for r in selected:lines.append(f"|{r['source_id']}|{r['speaker_id']}|{r['samples']}|")
    lines+=['','拟用同一G29开发房间的三RT、点源/八源/干净，共36条件；冻结四个神经系统（原McNet及三个1500步适配组）的直接/MVDR/MWF，加原带噪输入，共468输出。先数据和信号归档，再按已有两个固定ASR的raw与固定电平控制分别评分；每后端936行，不新增模型、训练或语料下载。全句转写仅用于计分，不输入识别器或增强器。', '',
        '该诊断回答完整语句下信号/识别及干净电平现象是否一致，不覆盖或改判G31主门槛。新波形与旧裁剪的噪声区间、SNR标定、尺度和长度均可能不同，不能把两轮均值之差单独归因于上下文长度。保留原重建与简单掩码强对照，不挑有利系统或更低的ASR电平成绩。', '',
        '只有开发补诊断；声学吞字、人耳可懂度、算法创新、真实阵列迁移以及未见说话人泛化仍未证明。完整归档本阶段后再冻结下一阶段的执行协议。', '',
        '![MVDR预登记主比较](../artifacts/g31_development_signals_v1/primary_gate_deltas.png)', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('scientific_readback',report,[OUT/'plotted_primary_deltas.json',OUT/'primary_gate_deltas.png',OUT/'primary_gate_deltas.pdf',OUT/'next_complete_development_sources.json',ROOT/'scripts/g31_scientific.py'])
    print({'gate':False,'next_source_ids':[r['source_id'] for r in selected]})


if __name__=='__main__':main()
