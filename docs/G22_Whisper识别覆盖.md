# G22 Whisper 识别覆盖

固定G21解码身份和选项，仅按PCM样本哈希复用缓存。新房间及共同尺度下未缓存的参考实际解码，不按转写文本或旧条件ID套用参考成绩。

|系统|记录数|参考词数|替换|删除|插入|微平均WER|
|---|---:|---:|---:|---:|---:|---:|
|fixed_without_clean_identity_s20261112|504|3150|1394|375|461|70.79%|
|fixed_without_clean_identity_s20261112_mask_mvdr|504|3150|941|268|205|44.89%|
|fixed_without_clean_identity_s20261112_mask_mwf|504|3150|1009|391|300|53.97%|
|fixed_without_clean_identity_s20261113|504|3150|1359|293|327|62.83%|
|fixed_without_clean_identity_s20261113_mask_mvdr|504|3150|922|245|214|43.84%|
|fixed_without_clean_identity_s20261113_mask_mwf|504|3150|1064|275|620|62.19%|
|fixed_without_clean_identity_s20261114|504|3150|1408|328|301|64.67%|
|fixed_without_clean_identity_s20261114_mask_mvdr|504|3150|935|268|183|44.00%|
|fixed_without_clean_identity_s20261114_mask_mwf|504|3150|1056|343|545|61.71%|
|mcnet|504|3150|1511|504|857|91.17%|
|mcnet_mask_mvdr|504|3150|1178|487|552|70.38%|
|mcnet_mask_mwf|504|3150|1345|414|546|73.17%|
|noisy|504|3150|1380|334|283|63.40%|
|rigid_without_clean_identity_s20261112|504|3150|1448|155|361|62.35%|
|rigid_without_clean_identity_s20261112_mask_mvdr|504|3150|934|115|193|39.43%|
|rigid_without_clean_identity_s20261112_mask_mwf|504|3150|1076|93|671|58.41%|
|rigid_without_clean_identity_s20261113|504|3150|1424|141|359|61.08%|
|rigid_without_clean_identity_s20261113_mask_mvdr|504|3150|913|114|197|38.86%|
|rigid_without_clean_identity_s20261113_mask_mwf|504|3150|1043|101|600|55.37%|
|rigid_without_clean_identity_s20261114|504|3150|1440|157|343|61.59%|
|rigid_without_clean_identity_s20261114_mask_mvdr|504|3150|941|98|212|39.71%|
|rigid_without_clean_identity_s20261114_mask_mwf|504|3150|1043|108|590|55.27%|
|rotated_without_clean_identity_s20261112|504|3150|1471|145|323|61.56%|
|rotated_without_clean_identity_s20261112_mask_mvdr|504|3150|950|110|205|40.16%|
|rotated_without_clean_identity_s20261112_mask_mwf|504|3150|1055|117|476|52.32%|
|rotated_without_clean_identity_s20261113|504|3150|1453|167|336|62.10%|
|rotated_without_clean_identity_s20261113_mask_mvdr|504|3150|947|112|220|40.60%|
|rotated_without_clean_identity_s20261113_mask_mwf|504|3150|1070|118|372|49.52%|
|rotated_without_clean_identity_s20261114|504|3150|1452|154|350|62.10%|
|rotated_without_clean_identity_s20261114_mask_mvdr|504|3150|952|88|216|39.87%|
|rotated_without_clean_identity_s20261114_mask_mwf|504|3150|1058|111|394|49.62%|
|truth_covariance_mvdr|432|2700|474|4|118|22.07%|
|truth_covariance_mwf|432|2700|550|18|103|24.85%|

覆盖表混合两房间、方向、RT和干净条件，不用来选择方法。真值统计只有432条带噪，其他系统各504条，不直接比较表中合并均值。

新增解码参考 220 条、输出 10143 条。
2026-11-12T19:29:44.177590+00:00
