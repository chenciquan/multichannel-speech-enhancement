# G18E Whisper单房间反射压力初评

信号报告归档后才识别，沿用固定模型/选项/英语归一化，无参考提示。缓存按G3→G5→G7→G9→G10→G14→G16→G17固定优先级使用，冲突全部保留，不按WER选择。下表仅完整覆盖汇总，混有干净/三种训练策略/请求RT/方位，不用于宣称全条件优势。

|系统|记录|微平均WER|
|---|---:|---:|
|fixed_without_clean_identity_s20261112|504|68.5%|
|fixed_without_clean_identity_s20261112_mask_mvdr|504|34.4%|
|fixed_without_clean_identity_s20261113|504|54.0%|
|fixed_without_clean_identity_s20261113_mask_mvdr|504|36.7%|
|fixed_without_clean_identity_s20261114|504|56.4%|
|fixed_without_clean_identity_s20261114_mask_mvdr|504|43.0%|
|mcnet|504|68.9%|
|mcnet_mask_mvdr|504|58.0%|
|noisy|504|49.7%|
|rigid_without_clean_identity_s20261112|504|53.6%|
|rigid_without_clean_identity_s20261112_mask_mvdr|504|30.1%|
|rigid_without_clean_identity_s20261113|504|48.8%|
|rigid_without_clean_identity_s20261113_mask_mvdr|504|32.3%|
|rigid_without_clean_identity_s20261114|504|48.6%|
|rigid_without_clean_identity_s20261114_mask_mvdr|504|31.1%|
|rotated_without_clean_identity_s20261112|504|60.6%|
|rotated_without_clean_identity_s20261112_mask_mvdr|504|32.9%|
|rotated_without_clean_identity_s20261113|504|64.8%|
|rotated_without_clean_identity_s20261113_mask_mvdr|504|35.4%|
|rotated_without_clean_identity_s20261114|504|52.6%|
|rotated_without_clean_identity_s20261114_mask_mvdr|504|36.3%|

新增识别10640条，缓存冲突0条；保存全部逐例S/D/I，WER不截断。本轮所有系统均不使用真值掩码或方位。下一步在独立CTC后端交叉核查，然后按请求RT、训练策略、方向、噪声及全部适配器种子分条件讨论。

UTC：2026-11-12T11:13:32.179621+00:00
