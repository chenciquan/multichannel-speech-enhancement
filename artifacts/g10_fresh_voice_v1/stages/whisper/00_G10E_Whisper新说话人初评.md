# G10E Whisper新说话人初评

信号报告归档后才识别，沿用固定模型/选项/英语归一化，无参考提示。缓存按G3→G5→G7→G9固定优先级使用，冲突全部保留，不按WER选择。下表仅完整覆盖汇总，混有干净/不同SNR/方位，不用于宣称全条件优势。

|系统|记录|微平均WER|
|---|---:|---:|
|fixed_without_clean_identity_s20261112|624|52.4%|
|fixed_without_clean_identity_s20261112_mask_mvdr|624|33.5%|
|fixed_without_clean_identity_s20261113|624|47.3%|
|fixed_without_clean_identity_s20261113_mask_mvdr|624|35.7%|
|fixed_without_clean_identity_s20261114|624|47.7%|
|fixed_without_clean_identity_s20261114_mask_mvdr|624|43.2%|
|mcnet|624|55.8%|
|mcnet_mask_mvdr|624|62.7%|
|noisy|624|56.6%|
|rotated_without_clean_identity_s20261112|624|46.7%|
|rotated_without_clean_identity_s20261112_mask_mvdr|624|21.2%|
|rotated_without_clean_identity_s20261113|624|43.0%|
|rotated_without_clean_identity_s20261113_mask_mvdr|624|22.3%|
|rotated_without_clean_identity_s20261114|624|51.2%|
|rotated_without_clean_identity_s20261114_mask_mvdr|624|21.5%|

新增识别8961条，缓存冲突0条；保存全部逐例S/D/I，WER不截断。本轮所有系统均不使用真值掩码或方位。下一步在独立CTC后端交叉核查，然后结合旧全部适配器种子分条件讨论。

UTC：2026-11-12T05:45:08.117737+00:00
