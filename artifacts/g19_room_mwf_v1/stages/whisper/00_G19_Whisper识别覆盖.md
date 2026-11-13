# G19 Whisper 识别覆盖

固定识别器覆盖 10 个 MWF 系统，各 504 个 G18 条件。新增解码 5040 条，历史缓存中同一样本的识别文本冲突 0 条。参考波形逐例匹配 G18 样本哈希，参考转写复用相同识别器的缓存。G18 基线评分文件及其 SHA-256 见 provenance.json。

|系统|条件数|参考词数|替换|删除|插入|微平均 WER|
|---|---:|---:|---:|---:|---:|---:|
|fixed_without_clean_identity_s20261112_mask_mwf|504|3969|998|480|252|43.6%|
|fixed_without_clean_identity_s20261113_mask_mwf|504|3969|1046|282|263|40.1%|
|fixed_without_clean_identity_s20261114_mask_mwf|504|3969|1023|392|672|52.6%|
|mcnet_mask_mwf|504|3969|1244|608|1150|75.6%|
|rigid_without_clean_identity_s20261112_mask_mwf|504|3969|1006|157|774|48.8%|
|rigid_without_clean_identity_s20261113_mask_mwf|504|3969|941|177|321|36.3%|
|rigid_without_clean_identity_s20261114_mask_mwf|504|3969|947|185|218|34.0%|
|rotated_without_clean_identity_s20261112_mask_mwf|504|3969|952|204|414|39.6%|
|rotated_without_clean_identity_s20261113_mask_mwf|504|3969|973|200|324|37.7%|
|rotated_without_clean_identity_s20261114_mask_mwf|504|3969|956|178|339|37.1%|

各行合并所有房间条件，仅供核对覆盖。逐例日志保留全部词对齐和未截断的 WER。

2026-11-12T13:30:11.266835+00:00
