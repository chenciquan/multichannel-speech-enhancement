"""Accepted gain-control observations and limits on causal interpretation."""
from g23_prepare import ROOT,OUT,sha,dump,now,archive
from g23_asr import meta


def main():
    assert meta(OUT/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    changes=meta(OUT/'recognition/summary.json')['gain_changes'];result=[]
    for engine in ('Whisper','CTC'):
        for db in (-6,-12):
            selected=[r for r in changes if r['engine']==engine and r['gain_db']==db]
            assert len(selected)==6
            result.append({'engine':engine,'gain_db':db,'pairs':sum(r['pairs'] for r in selected),
                           'normalized_changes':sum(r['normalized_hypothesis_changes'] for r in selected),
                           **{k:sum(r[k] for r in selected) for k in ('S_delta','D_delta','I_delta','errors_delta','N')}})
    dump(OUT/'recognition/readback_counts.json',result,True)
    report=ROOT/'docs/G23_识别变化读回与重复性验证决策.md'
    lines=['# G23 识别变化读回与重复性验证决策','','两后端各1440评分已验收；0dB的480个基准均按原G22缓存匹配。非退化SI-SDR变化最大约4×10⁻⁸ dB，缩放往返相对误差约2.65×10⁻⁸。','',
           '|识别器|增益dB|配对数|规范化文本变化数|ΔS|ΔD|ΔI|Δ总错误|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result:lines.append(f"|{r['engine']}|{r['gain_db']}|{r['pairs']}|{r['normalized_changes']}|{r['S_delta']}|{r['D_delta']}|{r['I_delta']}|{r['errors_delta']}|")
    lines+=['','以上跨输出类别合并计数只显示控制实验发生了多少变化，不用来排名算法或把480对当480个独立句子。完整分类、种子、方向及说话人结果保留。','',
            '这一结果尚不能排除跨次运行差异：0dB来自旧缓存，而非零增益在本阶段新解码。虽已固定模型、设备和解码配置，仍应显式重新推理0dB、核对跨次与同次重复性。另应把接近二进制增益的−6/−12dB控制，补充为精确×1/2、×1/4控制，并逐样本验证可精确反向还原，以排除这类舍入影响。','',
            '下一阶段保留相同48条件和10种输出，不重新选择样本。每个基准按顺序真实解码：1倍第一次、1/2倍、1/4倍、1倍第二次；合计1920观察/后端。历史缓存只用于对照，不用于跳过这些独立解码或覆盖旧文本。若基准重复本身不一致，应先报告不稳定性，不能只把差异归因于增益。','',
            '精确二进制增益是数值控制而非最佳参数搜索，不提出新增强算法，不保证人耳可听度不变，也不实施TSOS。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('scientific_readback',report,[OUT/'recognition/readback_counts.json',ROOT/'scripts/g23_readback.py',OUT/'FINAL_ACCEPTANCE_AUDIT.json'])
    print(result)


if __name__=='__main__':main()
