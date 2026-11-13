# G4B 冻结模型信号评估报告

在G4A数据协议与快照保存后推理。本阶段没有训练或调参。每系统472例，全部使用同一清单、FP32推理，保存FLOAT WAV重读计分。

|系统|条件|有效/总数|SI-SDR dB|PESQ|STOI|
|---|---|---:|---:|---:|---:|
|noisy|normal|288/288|-1.870|1.110|0.6874|
|noisy|rotation|96/96|-1.871|1.087|0.6921|
|noisy|gain|32/32|-1.871|1.087|0.6921|
|noisy|latency|32/32|-1.871|1.087|0.6921|
|noisy|clean|24/24|156.536|4.644|1.0000|
|mcnet|normal|288/288|-3.715|1.128|0.6412|
|mcnet|rotation|96/96|-3.010|1.082|0.6437|
|mcnet|gain|32/32|-2.380|1.099|0.6705|
|mcnet|latency|32/32|-2.863|1.090|0.6659|
|mcnet|clean|24/24|23.715|4.314|0.9946|
|ordinary_s20261112|normal|288/288|0.674|1.351|0.7794|
|ordinary_s20261112|rotation|96/96|1.159|1.303|0.7915|
|ordinary_s20261112|gain|32/32|4.821|1.497|0.8608|
|ordinary_s20261112|latency|32/32|4.195|1.466|0.8513|
|ordinary_s20261112|clean|24/24|16.968|3.167|0.9656|
|complete_s20261112|normal|288/288|1.405|1.397|0.7847|
|complete_s20261112|rotation|96/96|1.733|1.357|0.7994|
|complete_s20261112|gain|32/32|5.826|1.593|0.8864|
|complete_s20261112|latency|32/32|4.014|1.522|0.8557|
|complete_s20261112|clean|24/24|20.812|3.363|0.9768|
|without_stratification_s20261112|normal|288/288|0.608|1.342|0.7813|
|without_stratification_s20261112|rotation|96/96|0.936|1.291|0.7928|
|without_stratification_s20261112|gain|32/32|5.017|1.461|0.8643|
|without_stratification_s20261112|latency|32/32|3.502|1.430|0.8432|
|without_stratification_s20261112|clean|24/24|19.982|3.398|0.9783|
|without_clean_identity_s20261112|normal|288/288|2.009|1.430|0.7801|
|without_clean_identity_s20261112|rotation|96/96|2.588|1.405|0.7969|
|without_clean_identity_s20261112|gain|32/32|6.118|1.667|0.8843|
|without_clean_identity_s20261112|latency|32/32|5.796|1.635|0.8622|
|without_clean_identity_s20261112|clean|24/24|16.450|3.096|0.9581|

原始带噪参考在对应空间反事实组内完全相同，构建哈希已验证；这些指标仍不能证明词语或音素完整。下面进入固定识别器阶段，不返回修改模型。
