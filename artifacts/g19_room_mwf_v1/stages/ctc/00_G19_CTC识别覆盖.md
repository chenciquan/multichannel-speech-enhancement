# G19 CTC 识别覆盖

固定识别器覆盖 10 个 MWF 系统，各 504 个 G18 条件。新增解码 5040 条，历史缓存中同一样本的识别文本冲突 0 条。参考波形逐例匹配 G18 样本哈希，参考转写复用相同识别器的缓存。G18 基线评分文件及其 SHA-256 见 provenance.json。

|系统|条件数|参考词数|替换|删除|插入|微平均 WER|
|---|---:|---:|---:|---:|---:|---:|
|fixed_without_clean_identity_s20261112_mask_mwf|504|3969|1022|684|57|44.4%|
|fixed_without_clean_identity_s20261113_mask_mwf|504|3969|1056|496|59|40.6%|
|fixed_without_clean_identity_s20261114_mask_mwf|504|3969|1040|620|49|43.1%|
|mcnet_mask_mwf|504|3969|1202|1060|68|58.7%|
|rigid_without_clean_identity_s20261112_mask_mwf|504|3969|980|389|60|36.0%|
|rigid_without_clean_identity_s20261113_mask_mwf|504|3969|985|338|66|35.0%|
|rigid_without_clean_identity_s20261114_mask_mwf|504|3969|967|358|58|34.8%|
|rotated_without_clean_identity_s20261112_mask_mwf|504|3969|992|405|43|36.3%|
|rotated_without_clean_identity_s20261113_mask_mwf|504|3969|1029|356|58|36.4%|
|rotated_without_clean_identity_s20261114_mask_mwf|504|3969|986|357|53|35.2%|

各行合并所有房间条件，仅供核对覆盖。逐例日志保留全部词对齐和未截断的 WER。

2026-11-12T13:31:21.289428+00:00
