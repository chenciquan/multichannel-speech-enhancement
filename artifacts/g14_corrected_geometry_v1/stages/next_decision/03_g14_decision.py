#!/usr/bin/env python3
import json
from g14_common import ROOT,OUT,now,sha,dump,archive


def main():
    assert json.loads((OUT/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok';r=json.loads((OUT/'evaluation/readback.json').read_text());seeds=r['all_seed_legacy_training_contrasts']['rigid_rotation'];assert len(seeds)==3
    checks={'all3legacy_seeds_correcttest_minus_fixed_direct_minus40_SIpositive':all(s['direct']['-40']['si_sdr_db']>0 for s in seeds.values()),
       'all3legacy_seeds_correcttest_minus_fixed_direct20_SInegative':all(s['direct']['20']['si_sdr_db']<0 for s in seeds.values()),
       'all3legacy_seeds_correcttest_minus_fixed_direct20_CTCworse':all(s['direct']['20']['CTC_WER']>0 for s in seeds.values())}
    assert all(checks.values())
    report=ROOT/'docs/G14_结论读回与正确训练阶段决策.md';report.write_text('# G14 结论读回与正确训练阶段决策\n\nG14完整验收及全部三种子读回后作本决策。1,080对负控制波形和识别假设完全相同，98项测试通过。正确测试几何下，旧反向训练对固定训练：直接输出−40°的SI改善在3/3种子成立；20°SI下降和CTC退步亦3/3成立。汇总−40°SI差约+4.24dB、20°约−2.55dB。不是新独立说话人实验，也未证实吞字。Whisper部分条件超过100%受识别插入影响，不当作词语丢失率。\n\n旧反向训练并非无价值：它是改变相对几何的一种数据增强策略。错误在将其称为保持几何的刚性旋转及误标点源角，而非已经证明该训练方法无效。两者必须作为不同训练策略实证比较，不能预设正确旋转训练一定更好。\n\n下一阶段G15仅复用G5全部2048训练记录的源片段、噪声片段、目标角序列、缩放和采样，改为目标/噪声同向刚性旋转。原参考麦输入和监督目标需逐样本完全相同；保持rank4、3584参数、Adam3e-4、无干净损失、FP32/batch1、三种子各3000步。明确固定末步3000注册，不按新验证成绩择优；旧六个比较模型核实也均为3000步。\n\n验证重放原G5随机片段/八源方向并核对54个旧输入后，生成正确验证几何，仅观察训练是否正常，不用它挑新权重。原g3_train引擎的历史selected字段留存但不用于本阶段注册。训练结束必须核验原模型未变、全部有效步数、同种子采样状态/计数及实际显存，再保存报告。当前还没有正确训练结果。\n\n先在已查看的正确几何开发条件诊断，再另立未查看说话人小规模确认；混响后移。若三种子结果不利，同样全部报告，不追加模块救成绩，不声称旋转/LoRA/MVDR组合本身创新。\n\nUTC：'+now()+'\n',encoding='utf-8')
    dump(OUT/'NEXT_TRAINING_DECISION.json',{'utc':now(),'status':'ready_for_G15preregistered_preparation','checks':checks,'acceptance_sha256':sha(OUT/'FINAL_ACCEPTANCE_AUDIT.json'),'candidate_plan_sha256':sha(ROOT/'configs/g15_rigid_training_plan.json')},True)
    archive('next_decision',report,[OUT/'NEXT_TRAINING_DECISION.json',ROOT/'configs/g15_rigid_training_plan.json',ROOT/'scripts/g14_decision.py']);print('G14decision archived beforeG15data or training')


if __name__=='__main__':main()
