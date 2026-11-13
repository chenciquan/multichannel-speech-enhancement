# G17E Whisper新语音确认初评

信号报告归档后才识别，沿用固定模型/选项/英语归一化，无参考提示。缓存按G3→G5→G7→G9→G10→G14→G16固定优先级使用，冲突全部保留，不按WER选择。下表仅完整覆盖汇总，混有干净/三种训练策略/方位，不用于宣称全条件优势。

|系统|记录|微平均WER|
|---|---:|---:|
|fixed_without_clean_identity_s20261112|168|67.3%|
|fixed_without_clean_identity_s20261112_mask_mvdr|168|19.4%|
|fixed_without_clean_identity_s20261113|168|39.5%|
|fixed_without_clean_identity_s20261113_mask_mvdr|168|17.7%|
|fixed_without_clean_identity_s20261114|168|42.7%|
|fixed_without_clean_identity_s20261114_mask_mvdr|168|19.0%|
|mcnet|168|63.3%|
|mcnet_mask_mvdr|168|58.5%|
|noisy|168|36.7%|
|rigid_without_clean_identity_s20261112|168|32.9%|
|rigid_without_clean_identity_s20261112_mask_mvdr|168|15.9%|
|rigid_without_clean_identity_s20261113|168|33.1%|
|rigid_without_clean_identity_s20261113_mask_mvdr|168|14.1%|
|rigid_without_clean_identity_s20261114|168|33.6%|
|rigid_without_clean_identity_s20261114_mask_mvdr|168|14.7%|
|rotated_without_clean_identity_s20261112|168|33.6%|
|rotated_without_clean_identity_s20261112_mask_mvdr|168|15.0%|
|rotated_without_clean_identity_s20261113|168|33.4%|
|rotated_without_clean_identity_s20261113_mask_mvdr|168|14.4%|
|rotated_without_clean_identity_s20261114|168|35.0%|
|rotated_without_clean_identity_s20261114_mask_mvdr|168|14.6%|

新增识别3416条，缓存冲突0条；保存全部逐例S/D/I，WER不截断。本轮所有系统均不使用真值掩码或方位。下一步在独立CTC后端交叉核查，然后按训练策略、方向、噪声及全部适配器种子分条件讨论。

UTC：2026-11-12T10:01:52.765169+00:00
