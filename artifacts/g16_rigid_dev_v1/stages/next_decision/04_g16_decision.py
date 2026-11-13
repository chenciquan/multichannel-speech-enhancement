#!/usr/bin/env python3
import json
from g16_common import ROOT,OUT,now,sha,dump,archive


def main():
    assert json.loads((OUT/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok';r=json.loads((OUT/'evaluation/readback.json').read_text())
    effects={str(t):{k:r['cells']['rigid'][str(t)][k]-r['cells']['counter'][str(t)][k] for k in ('si_sdr_db','Whisper_WER','CTC_WER')} for t in (-40,20,70)}
    report=ROOT/'docs/G16_开发负结果与新语音确认决策.md';report.write_text('# G16 开发负结果与新语音确认决策\n\nG16联合报告及验收保存后读回：这8句/4人开发条件上，正确刚性训练并未显示整体优于旧反向训练。合并两噪声和三种子，直接输出rigid−counter的SI差约−0.18/−0.30/−0.14dB（−40/20/70°），两个WER有小幅好坏互见；MVDR后亦没有明显均值优势。详细每种子/说话人/场型及误差计数全部保留。这是开发集描述，不作显著性或普遍无效结论。\n\n不更换步数、丢种子、修改损失或追加恢复模块救成绩；正确训练的三种子仍全部保留。旧反向增强可是一种有效联合方向分布变化，数学解释更正不等于它作为训练策略必须较差。\n\n下一阶段G17在所有21系统冻结后，按已登记的硬件时长约束与归档顺序取得4位项目从未使用的说话人、每人2句完整英语，排除全部已用speaker/sourceID/源SHA。全部固定/反向/刚性三种子及对应成熟MVDR同时比较；不按开发排名只保留赢家，预定主要对照仍rigid−counter，次要rigid−fixed。新说话人仅相对项目，不保证基础识别器训练未见公开内容。\n\n只保存8个FLAC和文本，压缩流上限192MiB、不保存整346MB归档，也不称整包MD5核验。OOFFICE/NPARK是旧噪声录音，0dB、三目标角/三空间场，共168条件×21=3528信号及各ASR评分。语料/数据/直接/MVDR/Whisper/CTC逐阶段存档。后续若需任何调参，新确认集即成为已查看数据，不能继续当未经触碰的最终测试。\n\n当前仅HEAD核实公开test-clean端点，没有下载新语音。来源：[OpenSLR12](https://www.openslr.org/12)，官方标记CC-BY4.0；选句依完整时长1—3.51秒，不截字，以便后续0.48秒有限房间响应全卷积仍小于4秒。混响仍待单独受控压力测试，不据自由场成绩推断真实录音。\n\nUTC：'+now()+'\n',encoding='utf-8')
    dump(OUT/'NEXT_CONFIRMATION_DECISION.json',{'status':'ready_for_preregistered_G17fresh_sources','utc':now(),'development_rigid_minus_counter':effects,
        'G16_acceptance_sha256':sha(OUT/'FINAL_ACCEPTANCE_AUDIT.json'),'G17_candidate_plan_sha256':sha(ROOT/'configs/g17_fresh_confirmation_plan.json')},True)
    archive('next_decision',report,[OUT/'NEXT_CONFIRMATION_DECISION.json',ROOT/'configs/g17_fresh_confirmation_plan.json',ROOT/'artifacts/g17_source_endpoint_v1.json',ROOT/'scripts/g16_decision.py']);print('G16negative results and unchanged21systemconfirmation decision archived')


if __name__=='__main__':main()
