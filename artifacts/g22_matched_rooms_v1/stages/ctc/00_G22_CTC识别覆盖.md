# G22 CTC 识别覆盖

固定G21解码身份和选项，仅按PCM样本哈希复用缓存。新房间及共同尺度下未缓存的参考实际解码，不按转写文本或旧条件ID套用参考成绩。

|系统|记录数|参考词数|替换|删除|插入|微平均WER|
|---|---:|---:|---:|---:|---:|---:|
|fixed_without_clean_identity_s20261112|504|3150|1209|481|76|56.06%|
|fixed_without_clean_identity_s20261112_mask_mvdr|504|3150|857|336|38|39.08%|
|fixed_without_clean_identity_s20261112_mask_mwf|504|3150|938|560|68|49.71%|
|fixed_without_clean_identity_s20261113|504|3150|1147|411|66|51.56%|
|fixed_without_clean_identity_s20261113_mask_mvdr|504|3150|810|314|34|36.76%|
|fixed_without_clean_identity_s20261113_mask_mwf|504|3150|986|386|57|45.37%|
|fixed_without_clean_identity_s20261114|504|3150|1180|449|70|53.94%|
|fixed_without_clean_identity_s20261114_mask_mvdr|504|3150|843|314|40|38.00%|
|fixed_without_clean_identity_s20261114_mask_mwf|504|3150|959|503|55|48.16%|
|mcnet|504|3150|1420|623|100|68.03%|
|mcnet_mask_mvdr|504|3150|1136|611|53|57.14%|
|mcnet_mask_mwf|504|3150|1231|738|44|63.90%|
|noisy|504|3150|1212|890|8|66.98%|
|rigid_without_clean_identity_s20261112|504|3150|1317|249|112|53.27%|
|rigid_without_clean_identity_s20261112_mask_mvdr|504|3150|794|191|27|32.13%|
|rigid_without_clean_identity_s20261112_mask_mwf|504|3150|972|279|55|41.46%|
|rigid_without_clean_identity_s20261113|504|3150|1283|251|131|52.86%|
|rigid_without_clean_identity_s20261113_mask_mvdr|504|3150|774|192|27|31.52%|
|rigid_without_clean_identity_s20261113_mask_mwf|504|3150|969|235|57|40.03%|
|rigid_without_clean_identity_s20261114|504|3150|1326|248|109|53.43%|
|rigid_without_clean_identity_s20261114_mask_mvdr|504|3150|787|189|27|31.84%|
|rigid_without_clean_identity_s20261114_mask_mwf|504|3150|984|253|51|40.89%|
|rotated_without_clean_identity_s20261112|504|3150|1350|253|129|54.98%|
|rotated_without_clean_identity_s20261112_mask_mvdr|504|3150|804|184|30|32.32%|
|rotated_without_clean_identity_s20261112_mask_mwf|504|3150|979|288|53|41.90%|
|rotated_without_clean_identity_s20261113|504|3150|1316|279|108|54.06%|
|rotated_without_clean_identity_s20261113_mask_mvdr|504|3150|815|194|32|33.05%|
|rotated_without_clean_identity_s20261113_mask_mwf|504|3150|1027|243|61|42.25%|
|rotated_without_clean_identity_s20261114|504|3150|1322|279|104|54.13%|
|rotated_without_clean_identity_s20261114_mask_mvdr|504|3150|820|184|25|32.67%|
|rotated_without_clean_identity_s20261114_mask_mwf|504|3150|991|246|56|41.05%|
|truth_covariance_mvdr|432|2700|354|30|10|14.59%|
|truth_covariance_mwf|432|2700|412|40|12|17.19%|

覆盖表混合两房间、方向、RT和干净条件，不用来选择方法。真值统计只有432条带噪，其他系统各504条，不直接比较表中合并均值。

新增解码参考 220 条、输出 10143 条。
2026-11-12T19:32:11.923557+00:00
