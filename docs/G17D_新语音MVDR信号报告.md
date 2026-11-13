# G17D 新语音确认估计掩码MVDR信号报告

直接输出报告归档后，10个对应神经前端使用冻结G9掩码与Souden MVDR，共1680条。掩码仅来自带噪输入与对应神经估计；没有真值语音/方位。CPU float64/complex128，前端耗时与滤波耗时分别存档。

|系统|记录|SI-SDR dB|PESQ|STOI|
|---|---:|---:|---:|---:|
|fixed_without_clean_identity_s20261112_mask_mvdr|168|6.34|1.853|0.887|
|fixed_without_clean_identity_s20261113_mask_mvdr|168|6.34|1.812|0.897|
|fixed_without_clean_identity_s20261114_mask_mvdr|168|6.32|1.854|0.890|
|mcnet_mask_mvdr|168|1.52|1.564|0.774|
|rigid_without_clean_identity_s20261112_mask_mvdr|168|7.30|1.767|0.922|
|rigid_without_clean_identity_s20261113_mask_mvdr|168|7.47|1.781|0.926|
|rigid_without_clean_identity_s20261114_mask_mvdr|168|7.56|1.786|0.925|
|rotated_without_clean_identity_s20261112_mask_mvdr|168|7.52|1.784|0.924|
|rotated_without_clean_identity_s20261113_mask_mvdr|168|7.42|1.786|0.927|
|rotated_without_clean_identity_s20261114_mask_mvdr|168|7.53|1.789|0.926|

上表混合干净/噪声/新语音确认/方向，仅为覆盖核验；不能据此宣称总体优越。最终按训练策略、方向、噪声、场型和全部种子分析。完整FLOAT回读一致，指标无回填，WER另评。

UTC：2026-11-12T09:57:12.712841+00:00
