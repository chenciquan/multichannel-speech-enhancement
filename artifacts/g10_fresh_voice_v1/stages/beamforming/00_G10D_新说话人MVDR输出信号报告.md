# G10D 新说话人MVDR输出信号报告

直接输出报告归档后，以同一G9固定滤波器处理全部七种冻结估计，无真值掩码/方位。新说话人16句/4人；7×624=4,368条MVDR输出，与4,992条直接输出合并9,360条，15系统全部保留。

|掩码前端|记录|SI-SDR dB|PESQ|STOI|
|---|---:|---:|---:|---:|
|mcnet|624|-0.20|1.366|0.670|
|fixed_without_clean_identity_s20261112|624|4.83|1.632|0.801|
|rotated_without_clean_identity_s20261112|624|7.60|1.664|0.868|
|fixed_without_clean_identity_s20261113|624|4.95|1.586|0.807|
|rotated_without_clean_identity_s20261113|624|7.50|1.655|0.872|
|fixed_without_clean_identity_s20261114|624|4.71|1.618|0.801|
|rotated_without_clean_identity_s20261114|624|7.66|1.667|0.872|

此处混合干净/噪声/方向，仅作阶段覆盖核验，不用于主结论。全部FLOAT回读一致、有限值和指标成功，无错误回填。ASR前保存组合清单SHA与脚本；下一步Whisper报告归档后再运行CTC并联合分条件分析。

UTC：2026-11-12T05:34:19.865147+00:00
