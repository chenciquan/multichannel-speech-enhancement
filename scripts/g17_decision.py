#!/usr/bin/env python3
import json
from g17_common import ROOT,OUT,now,sha,dump,archive


def main():
    assert json.loads((OUT/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok';r=json.loads((OUT/'evaluation/readback.json').read_text());effects={}
    for suffix in ('','_mvdr'):
        effects[suffix or 'direct']={str(t):{k:r['cells']['rigid'+suffix][str(t)][k]-r['cells']['counter'+suffix][str(t)][k] for k in ('si_sdr_db','Whisper_WER','CTC_WER')} for t in (-40,20,70)}
    report=ROOT/'docs/G17_确认结果读回与混响阶段决策.md';report.write_text('# G17 确认结果读回与混响阶段决策\n\n全部21系统、3528条本轮信号与各识别器3528条评分及源排除审计通过，4组36音频辅助试听页保存；没有新的人工反馈或逐词标签。新语音4人8句在权重/方法冻结后取得，旧噪声片段与LoRA训练区间不重叠，但噪声录音仍是旧来源。\n\n没有支持“正确刚性训练必然优于反向训练”：直接rigid−counter平均SI约−0.22/−0.27/+0.02dB（−40/20/70°），WER好坏互见且种子不一致；MVDR后差异也较小。不能把小均值差称等效性或显著性证据。原反向与新刚性均保留，末步/种子不改。\n\n方向权衡仍可见：相对固定训练，变化方向分布的训练改善−40°，20°有代价；成熟MVDR大幅改变质量/识别权衡，但不是新结构创新或真实人声细节保留的充分证据。这些范围只含英语短朗读、两旧录音、0dB及名义阵列自由场。\n\n下一阶段G18固定全部21系统，复用本轮8句做单间6×5×3m的模拟反射压力，三请求RT=0/0.15/0.35秒，0.48秒有限响应全卷积，含持续背景噪声预卷积；参考为混响语音，仅做降噪。声源星座在固定房间内转动，墙不转，反射场本来会变化；RT0为同距离/同星座的无反射对照。它不同于旧平面波，而且八源场为可计算的固定全局星座，不能把与G17的全部差值都归因混响。\n\n先保存生成器模块哈希/物理检查/最大长度容量报告，再生成504条件/系统（432带噪+72干净），21系统10584信号及各ASR评分。每个源位置记录实际T20外推和有限尾端能量，不能把请求Sabine参数直接当实测T60。无新依赖、下载、权重或训练；原G12中止数据和更正记录保留，不恢复错误旋转实现。\n\n此阶段是已查看说话人的新条件压力，非新增独立说话人或真实录音。若表现退化，照实报告域失配，不追加模块或挑有利子集救成绩。阶段报告/审计保存后再决定下一项研究。\n\nUTC：'+now()+'\n',encoding='utf-8')
    dump(OUT/'NEXT_ROOM_DECISION.json',{'status':'ready_for_G18physics_preregistration','utc':now(),'rigid_minus_counter':effects,'acceptance_sha256':sha(OUT/'FINAL_ACCEPTANCE_AUDIT.json'),'candidate_plan_sha256':sha(ROOT/'configs/g18_room_candidate.json')},True)
    archive('next_decision',report,[OUT/'NEXT_ROOM_DECISION.json',ROOT/'configs/g18_room_candidate.json',ROOT/'scripts/g17_decision.py']);print('G17readback and room decision archived')


if __name__=='__main__':main()
