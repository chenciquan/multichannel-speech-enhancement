"""Read accepted paired results, preserving raw anchors and evaluation regimes."""
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from g26_prepare import ROOT,OUT,SOURCE,sha,read,dump,now,archive
from g26_asr import meta


def main():
    acceptance=meta(OUT/'FINAL_ACCEPTANCE_AUDIT.json');assert acceptance['status']=='ok'
    anchor=meta(SOURCE/'FINAL_ACCEPTANCE_AUDIT.json')
    anchor_checks={}
    for e in ('Whisper','CTC'):
        actual=sha(SOURCE/'recognition'/e/'asr_scores.jsonl')
        assert actual==anchor['ASR'][e]['score_sha256']
        anchor_checks[e]=actual
    summary=meta(OUT/'recognition/summary.json')
    by={(r['regime'],r['room'],r['rt'],r['group'],r['method'],r['clean']):r for r in summary['by_rt']}
    flips={e:[r for r in summary['method_comparisons'] if r['raw'][e+'_MWF_minus_MVDR_pp']*r['level_control'][e+'_MWF_minus_MVDR_pp']<0] for e in ('Whisper','CTC')}
    changes=meta(OUT/'recognition/raw_to_level_changes.json')
    counts={e:sum(r['normalized_hypothesis_changed'] for r in changes if r['engine']==e) for e in flips}
    signs={}
    for e in flips:
        signs[e]={regime:dict(Counter('MWF_better' if r[regime][e+'_MWF_minus_MVDR_pp']<0 else 'MVDR_better' if r[regime][e+'_MWF_minus_MVDR_pp']>0 else 'tie' for r in summary['method_comparisons'])) for regime in ('raw','level_control')}
    selected=[]
    for room in ('reference','new'):
        for rt in (0,.15,.35):
            for method in ('mvdr','mwf'):
                a,b=(by[regime,room,rt,'rigid',method,False] for regime in ('raw','level_control'))
                selected.append({'room':room,'rt':rt,'method':method,'rows':a['rows'],
                    **{e:{regime:by[regime,room,rt,'rigid',method,False][e] for regime in ('raw','level_control')} for e in flips}})
    out={'utc':now(),'raw_accepted_anchor_checks':anchor_checks,'normalized_hypothesis_changes':counts,
         'method_sign_flips':flips,'method_preferences':signs,'rigid_selected':selected,
         'scope':'Four reused speakers, one utterance each, two noises; dependent cells, not independent replications',
         'accepted_sha256':sha(OUT/'FINAL_ACCEPTANCE_AUDIT.json'),'summary_sha256':sha(OUT/'recognition/summary.json')}
    dump(OUT/'recognition/scientific_readback.json',out,True)
    fig,axes=plt.subplots(2,2,figsize=(10.2,6.8),sharex=True)
    for i,e in enumerate(('Whisper','CTC')):
        for j,room in enumerate(('reference','new')):
            ax=axes[i,j]
            for method,color in (('mvdr','#146c94'),('mwf','#a43e2a')):
                for regime,style in (('raw','--'),('level_control','-')):
                    ax.plot([0,.15,.35],[100*by[regime,room,rt,'rigid',method,False][e]['wer'] for rt in (0,.15,.35)],
                            style,marker='o',color=color,label=method.upper()+' / '+regime)
            ax.set_title(e+' / '+room+' room');ax.set_ylabel('WER (%)');ax.set_xticks([0,.15,.35]);ax.grid(alpha=.2)
            if i==1:ax.set_xlabel('Requested RT (s), not measured T60')
    axes[0,0].legend(fontsize=8);fig.suptitle('Fixed level-control sensitivity: rigid adapters, all three seeds\nFour reused utterances; curves are dependent condition aggregates',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91))
    for ext in ('png','pdf'):fig.savefig(OUT/f'recognition/raw_level_wer.{ext}',dpi=180)
    plt.close(fig)
    lines=['# G26 科学读回与后续瓶颈定位','','G26 已完整验收；另将两个原始 ASR 文件逐一核对到 G22 最终验收时的哈希，均一致。未替换原始评分。',
           '',f"16488 条记录中，规范化识别文本发生变化：Whisper {counts['Whisper']}，CTC {counts['CTC']}。同一句的不同系统、条件和两房间记录重复出现，该数量不是独立语句数。",
           '', '|正确刚性组三种子|请求RT|滤波|记录|Whisper raw / 电平|CTC raw / 电平|','|---|---:|---|---:|---:|---:|']
    for r in selected:
        lines.append(f"|{r['room']}|{r['rt']}|{r['method']}|{r['rows']}|{r['Whisper']['raw']['wer']:.2%} / {r['Whisper']['level_control']['wer']:.2%}|{r['CTC']['raw']['wer']:.2%} / {r['CTC']['level_control']['wer']:.2%}|")
    lines+=['','30 个房间×RT×前端类别比较中，严格改变 MWF/MVDR 排序的条件如下（差值单位：WER 百分点；负数表示 MWF 较低；等分另存，不算严格翻转）：','',
            '|识别器|房间|请求RT|前端|raw差值|电平差值|','|---|---|---:|---|---:|---:|']
    for e,entries in flips.items():
        for r in entries:
            key=e+'_MWF_minus_MVDR_pp';lines.append(f"|{e}|{r['room']}|{r['rt']}|{r['group']}|{r['raw'][key]:.3f}|{r['level_control'][key]:.3f}|")
        if not entries:lines.append(f'|{e}|无严格翻转|||||')
    lines+=['','完整偏好计数见 `scientific_readback.json`。均值排序变化不证明统计显著、声音吞字或新系统质量改善；.05/.95 为前阶段冻结的辅助评价规则，不据本表改参数。1015 个输出记录受峰值上限约束，不能声称所有输入 RMS 完全相等。',
        '', '## 下一步：先检查统计表示，再决定训练', '',
        '真值协方差系统与预测掩码系统的差距，尚不能归因于“只要学准掩码就能补齐”。可在同一 432 个带噪条件中，使用已知参考语音构造冻结功率比掩码，仅替换产生统计的掩码，其余 STFT、协方差归一化、MVDR/MWF 求解器和正则保持原设置。该系统使用不可部署真值，属于诊断，不是改进成果或理论上界。',
        '', '先固定少量覆盖两房间及全部请求 RT 的预检，核对真值参考恢复、与现成滤波函数一致的波形、固定滤波分量可加性，再决定完整运行。比较 McNet 掩码、真值功率比掩码及已有真值协方差的信号/分量差异；不在旧四句上挑参数、选最优掩码或宣称泛化。若信号诊断有意义，再按固定原始/电平控制的双 ASR 流程评分。',
        '', '空间协方差监督已有 Interspeech 2019 的直接先例，神经波束形成在线适配也已有研究，详见同日补查。后续即便训练，也应保留同数据、同参数预算的成熟目标对照；创新性暂未成立。', '',
        '![正确刚性组双房间 WER](../artifacts/g26_room_ASR_level_v1/recognition/raw_level_wer.png)', '', now()]
    report=ROOT/'docs/G26_科学读回与掩码瓶颈定位决策.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('scientific_readback',report,[OUT/'recognition/scientific_readback.json',OUT/'recognition/raw_level_wer.png',OUT/'recognition/raw_level_wer.pdf',ROOT/'scripts/g26_readback.py',ROOT/'docs/研究补查_掩码协方差监督与在线适配先例_20261113.md'])
    print({'changed':counts,'strict_flips':{e:len(v) for e,v in flips.items()},'preferences':signs})


if __name__=='__main__':main()
