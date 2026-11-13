# G5C 新说话人信号指标报告

数据与前一阶段快照保存后推理。本评估阶段没有训练或调参。每系统252例，全部使用同一清单、FP32推理，保存FLOAT WAV重读计分。

|系统|条件|有效/总数|SI-SDR dB|PESQ|STOI|
|---|---|---:|---:|---:|---:|
|noisy|clean|36/36|156.536|4.644|1.0000|
|noisy|multi8|72/72|-1.107|1.060|0.6552|
|noisy|point|144/144|-1.236|1.072|0.6917|
|mcnet|clean|36/36|20.824|4.311|0.9906|
|mcnet|multi8|72/72|-1.970|1.081|0.6550|
|mcnet|point|144/144|-3.960|1.069|0.6161|
|fixed_without_clean_identity_s20261112|clean|36/36|14.885|3.393|0.9414|
|fixed_without_clean_identity_s20261112|multi8|72/72|0.012|1.196|0.6107|
|fixed_without_clean_identity_s20261112|point|144/144|-0.868|1.296|0.7218|
|rotated_without_clean_identity_s20261112|clean|36/36|19.758|3.958|0.9824|
|rotated_without_clean_identity_s20261112|multi8|72/72|2.644|1.197|0.7110|
|rotated_without_clean_identity_s20261112|point|144/144|1.704|1.193|0.7472|
|fixed_complete_s20261112|clean|36/36|18.699|3.742|0.9652|
|fixed_complete_s20261112|multi8|72/72|0.592|1.188|0.6335|
|fixed_complete_s20261112|point|144/144|-0.684|1.251|0.7245|
|rotated_complete_s20261112|clean|36/36|22.361|4.053|0.9875|
|rotated_complete_s20261112|multi8|72/72|2.619|1.187|0.7097|
|rotated_complete_s20261112|point|144/144|1.330|1.178|0.7401|
|fixed_without_clean_identity_s20261113|clean|36/36|14.498|3.342|0.9375|
|fixed_without_clean_identity_s20261113|multi8|72/72|0.108|1.176|0.6379|
|fixed_without_clean_identity_s20261113|point|144/144|-1.128|1.239|0.7246|
|rotated_without_clean_identity_s20261113|clean|36/36|19.612|3.963|0.9845|
|rotated_without_clean_identity_s20261113|multi8|72/72|2.378|1.188|0.7108|
|rotated_without_clean_identity_s20261113|point|144/144|1.597|1.200|0.7507|
|fixed_complete_s20261113|clean|36/36|17.928|3.570|0.9574|
|fixed_complete_s20261113|multi8|72/72|0.755|1.184|0.6376|
|fixed_complete_s20261113|point|144/144|-0.453|1.273|0.7338|
|rotated_complete_s20261113|clean|36/36|22.397|4.073|0.9879|
|rotated_complete_s20261113|multi8|72/72|2.591|1.191|0.7118|
|rotated_complete_s20261113|point|144/144|1.441|1.185|0.7435|
|fixed_without_clean_identity_s20261114|clean|36/36|13.801|3.245|0.9314|
|fixed_without_clean_identity_s20261114|multi8|72/72|-0.153|1.189|0.6187|
|fixed_without_clean_identity_s20261114|point|144/144|-0.953|1.291|0.7264|
|rotated_without_clean_identity_s20261114|clean|36/36|19.564|3.930|0.9814|
|rotated_without_clean_identity_s20261114|multi8|72/72|2.622|1.195|0.7130|
|rotated_without_clean_identity_s20261114|point|144/144|1.576|1.199|0.7451|
|fixed_complete_s20261114|clean|36/36|17.967|3.584|0.9578|
|fixed_complete_s20261114|multi8|72/72|0.236|1.176|0.6346|
|fixed_complete_s20261114|point|144/144|-0.725|1.258|0.7313|
|rotated_complete_s20261114|clean|36/36|22.377|4.050|0.9875|
|rotated_complete_s20261114|multi8|72/72|2.558|1.186|0.7132|
|rotated_complete_s20261114|point|144/144|1.424|1.183|0.7466|

对应空间反事实组的参考一致性由数据协议和最终审计核查；这些指标不能证明词语或音素完整。下面进入固定识别器阶段，不返回修改模型。
