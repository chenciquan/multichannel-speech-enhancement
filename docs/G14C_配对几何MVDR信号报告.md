# G14D 正确几何配对自由场估计掩码MVDR信号报告

直接输出报告归档后，7个对应神经前端使用冻结G9掩码与Souden MVDR，共2352条。掩码仅来自带噪输入与对应神经估计；没有真值语音/方位。CPU float64/complex128，前端耗时与滤波耗时分别存档。

|系统|记录|SI-SDR dB|PESQ|STOI|
|---|---:|---:|---:|---:|
|fixed_without_clean_identity_s20261112_mask_mvdr|336|5.31|1.760|0.832|
|fixed_without_clean_identity_s20261113_mask_mvdr|336|5.39|1.730|0.839|
|fixed_without_clean_identity_s20261114_mask_mvdr|336|5.31|1.761|0.834|
|mcnet_mask_mvdr|336|1.99|1.557|0.714|
|rotated_without_clean_identity_s20261112_mask_mvdr|336|7.36|1.791|0.885|
|rotated_without_clean_identity_s20261113_mask_mvdr|336|7.28|1.782|0.889|
|rotated_without_clean_identity_s20261114_mask_mvdr|336|7.39|1.788|0.888|

上表混合干净/噪声/正确几何配对自由场/方向，仅为覆盖核验；不能据此宣称总体优越。最终按几何协议、方向、噪声、场型和全部种子分析。完整FLOAT回读一致，指标无回填，WER另评。

UTC：2026-11-12T08:50:49.219785+00:00
