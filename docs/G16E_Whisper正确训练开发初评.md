# G16E Whisper正确训练开发初评

信号报告归档后才识别，沿用固定模型/选项/英语归一化，无参考提示。缓存按G3→G5→G7→G9→G10→G14固定优先级使用，冲突全部保留，不按WER选择。下表仅完整覆盖汇总，混有干净/三种训练策略/方位，不用于宣称全条件优势。

|系统|记录|微平均WER|
|---|---:|---:|
|fixed_without_clean_identity_s20261112|168|54.1%|
|fixed_without_clean_identity_s20261112_mask_mvdr|168|32.5%|
|fixed_without_clean_identity_s20261113|168|81.7%|
|fixed_without_clean_identity_s20261113_mask_mvdr|168|32.5%|
|fixed_without_clean_identity_s20261114|168|73.7%|
|fixed_without_clean_identity_s20261114_mask_mvdr|168|32.2%|
|mcnet|168|71.1%|
|mcnet_mask_mvdr|168|102.1%|
|noisy|168|55.9%|
|rigid_without_clean_identity_s20261112|168|46.8%|
|rigid_without_clean_identity_s20261112_mask_mvdr|168|28.8%|
|rigid_without_clean_identity_s20261113|168|43.9%|
|rigid_without_clean_identity_s20261113_mask_mvdr|168|26.7%|
|rigid_without_clean_identity_s20261114|168|45.4%|
|rigid_without_clean_identity_s20261114_mask_mvdr|168|27.2%|
|rotated_without_clean_identity_s20261112|168|43.3%|
|rotated_without_clean_identity_s20261112_mask_mvdr|168|27.8%|
|rotated_without_clean_identity_s20261113|168|47.1%|
|rotated_without_clean_identity_s20261113_mask_mvdr|168|24.0%|
|rotated_without_clean_identity_s20261114|168|45.7%|
|rotated_without_clean_identity_s20261114_mask_mvdr|168|27.1%|

新增识别1008条，缓存冲突0条；保存全部逐例S/D/I，WER不截断。本轮所有系统均不使用真值掩码或方位。下一步在独立CTC后端交叉核查，然后按训练策略、方向、噪声及全部适配器种子分条件讨论。

UTC：2026-11-12T09:34:55.877065+00:00
