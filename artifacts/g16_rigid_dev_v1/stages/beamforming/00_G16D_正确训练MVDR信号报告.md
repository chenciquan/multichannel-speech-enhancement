# G16D 正确训练开发估计掩码MVDR信号报告

直接输出报告归档后，3个新对应神经前端使用冻结G9掩码与Souden MVDR，共504条。掩码仅来自带噪输入与对应神经估计；没有真值语音/方位。CPU float64/complex128，前端耗时与滤波耗时分别存档。

|系统|记录|SI-SDR dB|PESQ|STOI|
|---|---:|---:|---:|---:|
|rigid_without_clean_identity_s20261112_mask_mvdr|168|7.29|1.787|0.890|
|rigid_without_clean_identity_s20261113_mask_mvdr|168|7.36|1.797|0.894|
|rigid_without_clean_identity_s20261114_mask_mvdr|168|7.50|1.809|0.893|

上表混合干净/噪声/正确训练开发/方向，仅为覆盖核验；不能据此宣称总体优越。最终按训练策略、方向、噪声、场型和全部种子分析。完整FLOAT回读一致，指标无回填，WER另评。

UTC：2026-11-12T09:33:23.441723+00:00
