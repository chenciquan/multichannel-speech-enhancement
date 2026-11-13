# G7B 新噪声信号指标报告

数据与前一阶段快照保存后推理。本评估阶段没有训练或调参。每系统648例，全部使用同一清单、FP32推理，保存FLOAT WAV重读计分。

|系统|条件|有效/总数|SI-SDR dB|PESQ|STOI|
|---|---|---:|---:|---:|---:|
|noisy|multi8|216/216|-1.288|1.064|0.6540|
|noisy|point|432/432|-1.360|1.070|0.6646|
|mcnet|multi8|216/216|-3.490|1.090|0.6325|
|mcnet|point|432/432|-5.138|1.067|0.5805|
|fixed_without_clean_identity_s20261112|multi8|216/216|-0.762|1.243|0.6041|
|fixed_without_clean_identity_s20261112|point|432/432|-0.370|1.322|0.7107|
|rotated_without_clean_identity_s20261112|multi8|216/216|3.430|1.247|0.7044|
|rotated_without_clean_identity_s20261112|point|432/432|2.879|1.220|0.7259|
|fixed_complete_s20261112|multi8|216/216|0.229|1.235|0.6252|
|fixed_complete_s20261112|point|432/432|0.246|1.284|0.7064|
|rotated_complete_s20261112|multi8|216/216|2.803|1.229|0.7025|
|rotated_complete_s20261112|point|432/432|2.597|1.210|0.7195|
|fixed_without_clean_identity_s20261113|multi8|216/216|-0.660|1.214|0.6229|
|fixed_without_clean_identity_s20261113|point|432/432|-0.616|1.265|0.7125|
|rotated_without_clean_identity_s20261113|multi8|216/216|2.882|1.236|0.7046|
|rotated_without_clean_identity_s20261113|point|432/432|2.793|1.233|0.7314|
|fixed_complete_s20261113|multi8|216/216|0.499|1.240|0.6289|
|fixed_complete_s20261113|point|432/432|0.588|1.312|0.7204|
|rotated_complete_s20261113|multi8|216/216|2.852|1.230|0.7052|
|rotated_complete_s20261113|point|432/432|2.618|1.216|0.7216|
|fixed_without_clean_identity_s20261114|multi8|216/216|-0.762|1.234|0.6127|
|fixed_without_clean_identity_s20261114|point|432/432|-0.160|1.316|0.7151|
|rotated_without_clean_identity_s20261114|multi8|216/216|3.180|1.249|0.7058|
|rotated_without_clean_identity_s20261114|point|432/432|2.580|1.228|0.7229|
|fixed_complete_s20261114|multi8|216/216|-0.384|1.231|0.6286|
|fixed_complete_s20261114|point|432/432|0.099|1.287|0.7146|
|rotated_complete_s20261114|multi8|216/216|2.795|1.231|0.7067|
|rotated_complete_s20261114|point|432/432|2.684|1.217|0.7263|

对应空间反事实组的参考一致性由数据协议和最终审计核查；这些指标不能证明词语或音素完整。下面进入固定识别器阶段，不返回修改模型。
