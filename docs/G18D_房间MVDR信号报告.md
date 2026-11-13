# G18D 单房间反射压力估计掩码MVDR信号报告

直接输出报告归档后，10个对应神经前端使用冻结G9掩码与Souden MVDR，共5040条。掩码仅来自带噪输入与对应神经估计；没有真值语音/方位。CPU float64/complex128，前端耗时与滤波耗时分别存档。

|系统|记录|SI-SDR dB|PESQ|STOI|
|---|---:|---:|---:|---:|
|fixed_without_clean_identity_s20261112_mask_mvdr|504|2.65|1.594|0.781|
|fixed_without_clean_identity_s20261113_mask_mvdr|504|2.99|1.586|0.791|
|fixed_without_clean_identity_s20261114_mask_mvdr|504|2.73|1.599|0.783|
|mcnet_mask_mvdr|504|-0.01|1.491|0.721|
|rigid_without_clean_identity_s20261112_mask_mvdr|504|4.50|1.663|0.822|
|rigid_without_clean_identity_s20261113_mask_mvdr|504|4.72|1.675|0.826|
|rigid_without_clean_identity_s20261114_mask_mvdr|504|4.81|1.681|0.825|
|rotated_without_clean_identity_s20261112_mask_mvdr|504|4.74|1.681|0.824|
|rotated_without_clean_identity_s20261113_mask_mvdr|504|4.60|1.682|0.822|
|rotated_without_clean_identity_s20261114_mask_mvdr|504|4.71|1.681|0.825|

上表混合干净/噪声/单房间反射压力/方向，仅为覆盖核验；不能据此宣称总体优越。最终按训练策略、方向、噪声、场型和全部种子分析。完整FLOAT回读一致，指标无回填，WER另评。

UTC：2026-11-12T11:01:01.320320+00:00
