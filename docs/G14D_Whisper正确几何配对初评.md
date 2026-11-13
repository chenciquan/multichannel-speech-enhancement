# G14D Whisper正确几何配对初评

信号报告归档后才识别，沿用固定模型/选项/英语归一化，无参考提示。缓存按G3→G5→G7→G9→G10固定优先级使用，冲突全部保留，不按WER选择。下表仅完整覆盖汇总，混有干净/两种几何协议/方位，不用于宣称全条件优势。

|系统|记录|微平均WER|
|---|---:|---:|
|fixed_without_clean_identity_s20261112|336|63.2%|
|fixed_without_clean_identity_s20261112_mask_mvdr|336|37.7%|
|fixed_without_clean_identity_s20261113|336|67.0%|
|fixed_without_clean_identity_s20261113_mask_mvdr|336|36.8%|
|fixed_without_clean_identity_s20261114|336|62.0%|
|fixed_without_clean_identity_s20261114_mask_mvdr|336|44.7%|
|mcnet|336|66.0%|
|mcnet_mask_mvdr|336|97.6%|
|noisy|336|55.9%|
|rotated_without_clean_identity_s20261112|336|56.5%|
|rotated_without_clean_identity_s20261112_mask_mvdr|336|28.7%|
|rotated_without_clean_identity_s20261113|336|48.4%|
|rotated_without_clean_identity_s20261113_mask_mvdr|336|26.8%|
|rotated_without_clean_identity_s20261114|336|59.7%|
|rotated_without_clean_identity_s20261114_mask_mvdr|336|28.2%|

新增识别1564条，缓存冲突0条；保存全部逐例S/D/I，WER不截断。本轮所有系统均不使用真值掩码或方位。下一步在独立CTC后端交叉核查，然后按几何协议、方向、噪声及全部适配器种子分条件讨论。

UTC：2026-11-12T08:53:11.250345+00:00
