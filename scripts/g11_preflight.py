#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,subprocess,sys
from pathlib import Path
import torch
from g11_common import ROOT,OUT,BASE,now,read,dump,status,archive,Evaluator,selected_rows


def main():
    status(stage='preflight');assert not torch.cuda.is_initialized()
    with (OUT/'unit_tests.log').open('x') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_components.py','-v'],cwd=ROOT,
          env={**os.environ,'PYTHONPATH':str(ROOT/'src')},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    assert 'Ran 7 tests' in (OUT/'unit_tests.log').read_text() and '\nOK' in (OUT/'unit_tests.log').read_text()
    first=read(BASE/'sources/speech_manifest.jsonl')[0]['source_id'];systems={'mcnet_mask_mvdr','fixed_without_clean_identity_s20261112_mask_mvdr','rotated_without_clean_identity_s20261112_mask_mvdr'}
    probes=[r for r in selected_rows() if r['source_id']==first and r['noise_id']=='OOFFICE' and r['condition']=='point' and r['relative_angle']==-80 and r['system'] in systems];assert len(probes)==9
    evaluator=Evaluator();results=[evaluator.evaluate(r) for r in probes];assert not torch.cuda.is_initialized();dump(OUT/'preflight_probes.json',results,True)
    report=ROOT/'docs/G11A_分量诊断协议与还原验证.md';report.write_text('# G11A 分量诊断协议与还原验证\n\nG10最终验收保存后登记。采用原McNet及固定/旋转适配全部三种子，共7个已冻结MVDR前端，对G10的576带噪条件做4,032次条件分解。早期草案只取第一种子，本次在任何分量输出前扩大为全部既有种子；无额外训练、ASR、依赖或下载。16句/4人仍是已查看数据上的机制诊断。\n\n从原FLAC、相同阵列/方向/FIR和common_scale重建干净六通道，按原PCM24量化，参考第5麦必须逐样本完全一致。残余N定义为解码混合减量化干净场，因此包括量化差额。先只用带噪输入和旧神经估计重算MVDR权重，核对旧输出后固定权重，分别作用于S和N，验证输出叠加。真值只进入评估，不进入增强掩码。\n\n7项前置测试通过，覆盖已知衰减、噪声抑制、恒等、完全抵消/功率下限、非法输入、PCM24逐通道一致性和固定滤波器线性。实际首个源句OOFFICE/点噪声−80°/三方向及原始/固定/旋转首种子9个探测全部通过：参考完全一致，重算输出误差≤1e-6，叠加误差≤1e-6×(1+输出峰值)。未按结果放宽阈值。\n\n量化目标失真NMSE、目标RMS增益、残余噪声衰减及分量输出SNR；全句能量SNR不必等于活动语音标定0dB。零/过小输出功率使用预定1e-30下限并显式标记，未定义SI-SDR单列，不悄悄删除。权重范数只表示白噪声功率放大，未验证单位目标响应时不当作归一化WNG。\n\n这只能解释已经确定的线性滤波器响应，不能对非线性神经直接输出作同样因果分解，也不能证明音素或单词被删。采用经典语音失真/噪声抑制分析思路，非新增强算法。依据：[Souden等IEEE TASLP2010](https://doi.org/10.1109/TASL.2009.2025790)；这里声明的是全句经验能量指标，并非逐式复现原文全部理论公式。\n\n先保存本报告与代码/探测快照，再进行完整4,032条诊断。全程CPU，CUDA未初始化。\n\nUTC：'+now()+'\n',encoding='utf-8')
    archive('preflight',report,[OUT/'frozen_plan.json',OUT/'unit_tests.log',OUT/'preflight_probes.json',ROOT/'src/mcse/components.py',ROOT/'tests/test_components.py',ROOT/'scripts/g11_common.py',Path(__file__)])
    dump(OUT/'PREFLIGHT_COMPLETE.json',{'status':'ok','completed_utc':now(),'probes':9,'tests':7,'CUDA_initialized':torch.cuda.is_initialized()},True)


if __name__=='__main__':main()
