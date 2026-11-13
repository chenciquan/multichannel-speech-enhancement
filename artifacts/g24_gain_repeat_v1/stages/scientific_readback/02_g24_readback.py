"""Accepted reproducibility findings and the next evaluation-control decision."""
import json
from g24_prepare import ROOT,OUT,sha,dump,now,archive


def main():
    assert json.loads((OUT/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    summary=json.loads((OUT/'recognition/summary.json').read_text());rows=[]
    for e in ('Whisper','CTC'):
        for observation in ('half','quarter'):
            selected=[r for r in summary['gain_changes'] if r['engine']==e and r['observation']==observation]
            rows.append({'engine':e,'observation':observation,'pairs':sum(r['pairs'] for r in selected),
                         'normalized_changes':sum(r['normalized_changes'] for r in selected),
                         **{k:sum(r[k] for r in selected) for k in ('S_delta','D_delta','I_delta','errors_delta','N')}})
    dump(OUT/'recognition/readback_counts.json',{'repeatability':summary['repeatability'],'gain_counts':rows},True)
    report=ROOT/'docs/G24_科学读回与识别电平控制决策.md'
    lines=['# G24 科学读回与识别电平控制决策','','两个后端各1920次独立推理已验收，全部480个基准的跨次与同次规范化文本变化均为0。历史缓存没有被新观察覆盖。','',
           '|识别器|精确幅度系数|配对数|规范化文本变化|Δ错误数|','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['engine']}|{r['observation']}|{r['pairs']}|{r['normalized_changes']}|{r['errors_delta']}|")
    lines+=['','缩放文件逐样本精确反向还原，因此这些结果排除了输入缩放文件自身的舍入损失；识别器内部的有限精度处理仍属于评估链的一部分。这里的变化次数包含重复场景与多个训练种子，不作为独立语句的大样本发生率。','',
            '本组结果支持该评估链对全局输入幅度敏感。这并不说明所有模型差异或房间差异都由幅度造成；背景噪声、混响、目标失真和识别器能力仍可能起作用。WER变化、删除计数和TSOS均不能单独证明吞字。','',
            '下一阶段预先固定一个仅用于ASR输入的电平标准化：由当前单通道波形自身计算RMS和峰值，使用 min(0.05/RMS, 0.95/峰值) 作为全局正标量，不削波。目标0.05及峰值上限0.95是本次控制的固定工程约定，没有按识别成绩选择；不宣称最优或提出新算法。零信号保持零。','',
            '先在G24全部480个基准及其二进制增益上核对标准化后的float32输入是否逐样本完全相同，再对每个规范化基准做两遍独立识别。原增强音频、原质量指标、RMS/NMSE和raw-ASR全部保留。标准化副本只用于评估，不能用其ASR成绩声称增强模型本身改善。','',
            '这样只能消除所验证的全局电平因素，不能修复目标内容、改善实际信噪比或证明人耳可懂度不变。暂不扩大模型训练，也不加入未经完整复现的TSOS。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('scientific_readback',report,[OUT/'recognition/readback_counts.json',ROOT/'scripts/g24_readback.py',OUT/'FINAL_ACCEPTANCE_AUDIT.json'])
    print(rows)


if __name__=='__main__':main()
