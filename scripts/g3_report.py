#!/usr/bin/env python3
"""Generate a factual Chinese report from completed evidence, even unattended."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g3_overnight_v1';EVAL=OUT/'evaluation'
def mean_rows(summary,names,split='test'):
    data=[summary[split][name] for name in names]
    return {**{k:float(np.mean([r['metrics'][k] for r in data])) for k in ('si_sdr_db','pesq_wb','stoi')},
            'wer':float(np.mean([r['asr']['wer'] for r in data]))}
def main():
    summary=json.loads((EVAL/'summary.json').read_text());audit=json.loads((OUT/'completion_audit.json').read_text())
    assert audit['status']=='ok';seeds=summary['coverage']['paired_seeds'];selected=json.loads((OUT/'selected_checkpoints_extended.json').read_text())
    results=[json.loads((OUT/'train'/r['system']/'result.json').read_text()) for r in selected['systems']]
    comparison=next(r for r in summary['comparisons'] if r['label'].startswith('配对种子'))['source_paired_bootstrap_2000']
    dsi=comparison['si_sdr_db']['delta'];dw=comparison['wer']['delta'];dstoi=comparison['stoi']['delta']
    if dsi>0 and dw<=0 and dstoi>=0:verdict='完整方案在本轮平均指标上存在组合收益，但真实场景泛化、机制和新颖性仍未证明。'
    elif dsi<=0 and dw>=0:verdict='完整方案未同时优于普通适配，当前不能把这套组合认定为已验证的改进或创新。'
    else:verdict='完整方案与普通适配呈现指标权衡，当前不能认定整体更优，更不能仅凭SI-SDR宣布人声内容保护成功。'
    lines=['# G3 受控适配与组件实验验收报告','','本文件对应原五小时时间上限计划中的G3大阶段。用户后来授权连续运行约4小时；本阶段结束后先归档，再推进后续阶段。文中均为实际结果，未完成的后续工作不算结论。','',
      '## 1. 本轮判断','',f'**{verdict}**',f'完整方案相对普通适配，三个配对种子的平均差异为：SI-SDR **{dsi:+.3f} dB**、STOI **{dstoi:+.4f}**、WER **{dw*100:+.2f}个百分点**。这些是同一保留测试集上的配对比较，不是公开基准榜单成绩。','',
      'G1没有证实稳定的人声误删，所以本轮没有做声音回填、音素恢复或语义补全。研究对象是G2发现的限定空间噪声条件下的质量退化，以及小参数适配与干净语音损伤之间的关系。','',
      '## 2. 实际完成规模','',
      '- 训练：4位说话人、212句来源、约24.14分钟；固定2,048个一秒带噪训练片段，以及128个干净多通道片段。',
      '- 验证：另2位说话人、12句、48种混合条件。测试：另4位说话人、30句、360种混合条件；另有30句干净六通道输入对照。',
      '- 训练、验证、测试说话人隔离；此前G1/G2的32句从所有新划分排除，测试说话人也排除旧诊断说话人。重复条件不是独立语句。',
      '- 测试来源分布并不均衡：'+', '.join(f'{sp}号{n}句' for sp,n in summary['coverage']['test_source_counts_per_speaker'].items())+'。其中一位仅1句，不能把四位说话人当成均衡或充分的人群覆盖。',
      f'- 完成{len(results)}次正式FP32训练，每次3,000次有效更新、3,584个可训练参数；共{len(results)*3000:,}次有效更新。三个种子为'+', '.join(map(str,seeds))+'。',
      f'- 评估{summary["coverage"]["signal_rows"]:,}个信号对及相同数量的ASR结果；每系统360个噪声测试、30个干净输入，失败情况见审计。',
      '- 噪声训练使用DWASHING、DKITCHEN、OOFFICE；测试为OOFFICE与未用于训练的PCAFETER。相同录音按时间隔离，不能称为全新独立录音。',
      '- 所有条件先对噪声做固定40Hz高通，再标定−5/0/+5dB活动语音SNR；目标语音不做该滤波。六通道为名义阵列的模拟平面波，没有真实麦克风或房间混响数据。','',
      '## 3. 方法与公平性','',
      '固定官方McNet提交与离线权重，只在`freq.fc_output_layer`、`narr.fc_output_layer`加入Microsoft loralib的标准rank-4 LoRA，原3,328,706个参数冻结。适配器参数占原模型约0.108%，不是重新实现增强网络。','',
      '|方案|带噪训练采样|额外干净重建约束|', '|---|---|---|',
      '|普通适配 ordinary|同一池均匀抽样，点声源约25%|无|',
      '|完整方案 complete|点声源50%，七个点声源方向均匀；其余八源叠加|每5步一次，权重0.1|',
      '|去掉分层 without_stratification|均匀抽样|有|',
      '|去掉干净约束 without_clean_identity|分层抽样|无|','',
      '干净约束是增强输出与参考第5麦之间的归一化波形均方误差。四组共享数据池、位置、秩、Adam学习率0.0003、更新数和六次检查点选择机会；带约束组额外执行600次干净输入计算，所以不是严格等算力或等总输入次数。原监督损失为压缩cIRM均方误差；没有ASR反传、教师伪标签或大模型参与增强。','',
      '权重只按验证集选择：在STOI不低于初始McNet减0.005的检查点中，选平均SI-SDR最高者；无候选达标时显式标记失败。该门限不等于人声内容保证。最终测试在所有权重选择和训练报告保存后才打开，不用测试WER调参。','',
      '原主对照与第三配对种子按最初计划执行。首种子验证出现组件差异后，两个消融分别增加后两种子，扩展在测试前单独登记。因此主对照与验证驱动的探索性复验必须区别说明，不能把所有比较都称为最初预登记。','',
      '## 4. 测试结果','',
      '下表先列首种子，便于理解具体系统；完整三种子均值见后表。SI-SDR/PESQ/STOI越高越好，WER越低越好。WER按编辑总次数除参考词数计算，不平均单句WER。','',
      '|系统|SI-SDR dB|PESQ|STOI|WER|', '|---|---:|---:|---:|---:|']
    labels={'noisy':'带噪第5麦','hpf40':'混合输入40Hz高通','mcnet':'原始McNet','ordinary_s20261112':'普通适配','complete_s20261112':'完整方案','without_stratification_s20261112':'去掉分层','without_clean_identity_s20261112':'去掉干净约束'}
    for name,label in labels.items():
        r=summary['test'][name];m=r['metrics'];lines.append(f'|{label}|{m["si_sdr_db"]:.3f}|{m["pesq_wb"]:.3f}|{m["stoi"]:.4f}|{r["asr"]["wer"]:.2%}|')
    lines+=['','|三种子平均|SI-SDR dB|PESQ|STOI|WER|','|---|---:|---:|---:|---:|']
    variants=['ordinary','complete','without_stratification','without_clean_identity']
    for variant in variants:
        m=mean_rows(summary,[f'{variant}_s{s}' for s in seeds]);lines.append(f'|{variant}|{m["si_sdr_db"]:.3f}|{m["pesq_wb"]:.3f}|{m["stoi"]:.4f}|{m["wer"]:.2%}|')
    lines+=['','### 完整方案减普通适配的探索性区间','','先在每句内合并条件与配对种子，再以30句源语音为簇重采样2,000次。只有4位测试说话人和2段噪声录音，不能把该区间当成广泛人群泛化或多重比较校正后的显著性证明。','','|指标|平均差异|探索性95%区间|','|---|---:|---|']
    for key,r in comparison.items():
        scale=100 if key=='wer' else 1;unit='个百分点' if key=='wer' else ''
        lines.append(f'|{key}|{r["delta"]*scale:+.5f}{unit}|[{r["ci95"][0]*scale:+.5f}, {r["ci95"][1]*scale:+.5f}]|')
    macro=next(r for r in summary['comparisons'] if r['label'].startswith('配对种子'))['speaker_equal_weight_sensitivity']
    lines+=['',f'按四位说话人等权的敏感性差异：SI-SDR {macro["si_sdr_db"]:+.3f}dB，STOI {macro["stoi"]:+.4f}，WER {macro["wer"]*100:+.2f}个百分点。只有四簇、且其中一簇只有一句，不作稳定显著性推断。']
    lines+=['','![验证曲线](../artifacts/g3_overnight_v1/evaluation/figures/validation_curves.png)','','![配对测试差异](../artifacts/g3_overnight_v1/evaluation/figures/test_paired_deltas.png)','',
      '### 干净输入对照','','它检查本来没有噪声时模型是否引入误差；仍不能代替逐词听测。原始干净输入的WER为'+f'{summary["clean_test"]["noisy"]["asr"]["wer"]:.2%}，这也提示识别器存在固有错误。',
      '','|系统|干净输入SI-SDR dB|PESQ|STOI|WER|','|---|---:|---:|---:|---:|']
    for variant in ['mcnet']+variants:
        names=['mcnet'] if variant=='mcnet' else [f'{variant}_s{s}' for s in seeds];m=mean_rows(summary,names,'clean_test')
        lines.append(f'|{variant}|{m["si_sdr_db"]:.3f}|{m["pesq_wb"]:.3f}|{m["stoi"]:.4f}|{m["wer"]:.2%}|')
    lines+=['','## 5. 怎样解释及不能推出什么','',
      'SI-SDR反映波形误差，内部允许尺度投影；PESQ估计感知质量，STOI估计可懂度。它们都不能独立证明词尾、辅音或音素被保留。单一Whisper tiny.en的WER是识别任务代理指标，也不能把删除错误直接当成真实吞字。所有参考原文只用于评分，没有作为识别提示。','',
      '按噪声、结构、SNR、说话人及角度的完整分组表位于`evaluation/summary.json`；逐例分数与替换/删除/插入计数保留，不只展示平均数。试听页同时含改善候选、退步候选和各测试说话人的固定顺序抽查，没有赋予新的人工听感结论。试听副本若有越界峰值，仅对同例全部角色施加共同衰减，系数保留，不参与指标计算。','',
      '重要限制：目标方向始终固定20°方位/10°仰角；训练和测试使用同一名义阵列。点声源测试中+45°只在0dB出现，−60°只在±5dB出现，角度与SNR耦合，不能独立归因。八源为不同时间段噪声的方向叠加，不是实测扩散场。短英文句、少量说话人、无混响及无真实阵列验证都限制泛化。Whisper预训练语料重叠无法完全排除。','',
      'LoRA、采样均衡、多条件重放和干净重建均为已有思路。胜过原McNet只证明适配有用；需要进一步胜过普通适配和合理数据增强等强对照，才有资格讨论有边界的研究增量。本轮不保证发表论文，也不声称已有方法学创新。','',
      '## 6. 资源、异常和审计','',
      f'- 正式训练合计耗时{sum(r["elapsed_seconds"] for r in results)/60:.1f}分钟；这是各训练程序实测耗时之和，不含数据准备、失败尝试和后续评估。',
      f'- 正式运行最大Torch实际分配{max(r["peak_allocated_bytes"] for r in results)/2**30:.3f}GiB、最大保留缓存{max(r["peak_reserved_bytes"] for r in results)/2**30:.3f}GiB；系统与CUDA上下文另占显存。只验证本轮1秒、batch1的适配方案，不推断完整模型训练都可在8GB完成。',
      '- 初始AMP探测通过五次反向传播，但首次长跑在最后记录1,750步后出现非有限梯度并停止；失败日志与1,500步检查点保留，不纳入正式比较。随后所有组以FP32从原权重重启。',
      '- 单一原环境仅增加loralib==0.1.2，93包快照保存；未修改官方核查代码和权重、驱动、CUDA或系统代理，没有迁移或删除旧项目。',
      f'- 55项测试通过，最终审计状态 **{audit["status"]}**；覆盖{audit["counts"]["systems"]}个系统、{audit["counts"]["signal_pairs"]}个信号对与相同ASR覆盖。审计核对基线冻结、文件哈希、划分隔离、选权重先于测试、统一测试清单与固定高通对照。','',
      '## 7. 下一阶段与交付入口','',
      '本阶段报告和证据保存后，优先冻结新的泛化压力测试：把噪声角度与SNR完全交叉，改变目标方向，检查通道增益及固定同步偏差。首先使用冻结权重诊断，不借本阶段最终测试返回调参；若后续训练新方法，需另行登记验证和确认数据。真实房间与人声细节的人工核查仍需后续完成。','',
      '- [训练阶段报告](G3A_训练阶段报告.md)、[信号指标阶段报告](G3B_信号指标阶段报告.md)、[执行说明](G3_夜间实验执行说明.md)。',
      '- [完整机器可读汇总](../artifacts/g3_overnight_v1/evaluation/summary.json)、[指标速览](../artifacts/g3_overnight_v1/evaluation/metrics_overview.md)、[对照试听页](../artifacts/g3_overnight_v1/review/index.html)。',
      '- [审计](../artifacts/g3_overnight_v1/completion_audit.json)、[各阶段快照](../artifacts/g3_overnight_v1/stages/)、[选定权重清单](../artifacts/g3_overnight_v1/selected_checkpoints_extended.json)。','',
      '来源：[McNet官方仓库](https://github.com/Audio-WestlakeU/McNet)、[Microsoft loralib](https://github.com/microsoft/LoRA)、[LibriSpeech](https://www.openslr.org/12)、[DEMAND](https://zenodo.org/records/1227121)、[Whisper](https://github.com/openai/whisper)。这些链接说明组件和数据来源，不代表来源作者认可本项目的结果或创新。','',
      '生成时间（UTC）：'+datetime.now(timezone.utc).isoformat()]
    report=ROOT/'docs/G3_五小时实验验收报告.md'
    if report.exists():raise RuntimeError('Refuse overwriting completed G3 report')
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    terminal=f'G3已完成：{len(results)}次训练，{len(results)*3000}次有效更新，{audit["counts"]["systems"]}个测试系统。\n{verdict}\n完整减普通：SI-SDR {dsi:+.3f} dB；WER {dw*100:+.2f}个百分点。\n三个最大风险：1.固定目标方向和模拟阵列的泛化；2.单ASR及短句不能证明真实人声细节；3.组件均为既有方法，创新性未确定。\n报告：docs/G3_五小时实验验收报告.md\n'
    (OUT/'terminal_summary.txt').write_text(terminal,encoding='utf-8');print(terminal)
if __name__=='__main__':main()
