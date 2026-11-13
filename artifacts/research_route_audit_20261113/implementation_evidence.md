# 实现、模型注册和训练分布只读审计

审计日期：2026-11-13。仅读取源码、配置、归档、目录和哈希；未运行训练、增强推理、角度扫描或安装/下载。未找到工作区及 C:/、C:/Users、用户目录、Desktop 祖先层的 AGENTS.md。以下路径均相对于项目根目录。本文为新增审计笔记，不改历史证据。

## 1. 官方模型及双层 LoRA

- `third_party/McNet` 当前 Git HEAD 为 `9c194032659a3047a4b4c244a9f541f4d34b1682`，只读 `git status --short` 输出为空。主项目的无提交/未跟踪状态由总审计另述。
- `configs/baseline_provenance.json` 指向官方 `config/mc_net_offline.yaml` 及 `model_checkpoints/offline/epoch494_criteria18.78_sdr18.78.ckpt`，登记 SHA256 为 `24b8c556cc52990e4d7db83da4909b7e8a1dc4a0c14880b98381bb62818beecf`。这是登记事实，本笔记没有再次读取整块官方权重计算哈希。
- 官方配置选择通道0—5、参考通道4，16kHz，STFT 512/256。`third_party/McNet/src/model/mc_net.py:249` 为双向 RNN 的输出 FC；频率层双向 hidden=128，因此 FC 为256→64；窄带层双向 hidden=256，因此 FC 为512→64。
- `src/mcse/adaptation.py:5` 只注册 `freq.fc_output_layer`、`narr.fc_output_layer`；第8—21行先冻结所有参数，复制原 FC 权重/偏置，仅打开 `lora_A/lora_B` 梯度。rank=4、alpha=4、dropout=0、`merge_weights=False`。可训练数分别为4×(256+64)=1280、4×(512+64)=2304，共3584。
- `src/mcse/runtime.py:78` 注入同一结构并断言3584；80—84行按登记加载，仅允许匹配的适配键、形状和有限值。`adaptation.py:28` 检查上述条件。单通道重建沿用官方模型和原重建桥接，没有第二增强网络。
- 已有证据：`docs/G15B_正确旋转三种子训练与注册报告.md`、`artifacts/g15_rigid_training_v1/training_audit.json` 确认三次训练冻结基线摘要一致；`docs/G16A_正确训练权重注册与推理兼容验证.md` 记载三新模型加载重现开发评分、两旧输出回放与98项测试。`tests/test_adaptation.py` 还保留零初始等价、只更新低秩参数及加载异常检查。这里只读，不重复运行。
- 结论：两个适配位置已正确实现并在实际训练、注册、加载中使用。不能据此推定它们足以覆盖所有方位，或已证明容量充足/不足。

## 2. 已注册模型及新旧名称

`runtime.py:26—39` 的共享注册只读入三份归档注册表：G3 12个适配器、G5 12个适配器、G15 3个适配器，总27个，另有官方 `mcnet`。本次逐一读取27个已注册适配权重：全部存在，均54,564字节，计算 SHA256 与各注册表一致；均登记为3000步，种子均完整保留20261112/20261113/20261114。注册表分别为：

- `artifacts/g3_overnight_v1/evaluation/systems.json`：ordinary、complete、without_stratification、without_clean_identity各3种子。它们为早期固定目标方向的损失/采样消融历史，不宜将全部混为与G15严格匹配的主对照。
- `artifacts/g5_rotation_v1/selected_checkpoints.json`：fixed/rotated × complete/without_clean_identity ×3种子。
- `artifacts/g15_rigid_training_v1/registered_checkpoints.json`：rigid_without_clean_identity ×3种子。

|统一展示名称|原始完整运行时ID模板（s∈三个完整种子）|权重路径模板|位置与用途|
|---|---|---|---|
|单方位适配 LoRA（Fixed-DOA LoRA）|`g5/fixed_without_clean_identity_s{s}`|`artifacts/g5_rotation_v1/fixed/train/without_clean_identity_s{s}/adapter_03000.pt`|与G15预算、损失、采样匹配的主对照；目标+20°|
|相对方位变化增强 LoRA（Relative-Geometry-Varying LoRA）|`g5/rotated_without_clean_identity_s{s}`|`artifacts/g5_rotation_v1/rotated/train/without_clean_identity_s{s}/adapter_03000.pt`|历史counter/rotated；训练几何错误事实保留；必要补充|
|几何保持式多方位 LoRA（Geometry-Preserving LoRA）|`g15/rigid_without_clean_identity_s{s}`|`artifacts/g15_rigid_training_v1/train/without_clean_identity_s{s}/adapter_03000.pt`|正确刚性旋转主线|
|单方位适配 LoRA：含干净保真辅助项的历史版本|`g5/fixed_complete_s{s}`|`artifacts/g5_rotation_v1/fixed/train/complete_s{s}/adapter_03000.pt`|历史消融，区别于主对照|
|相对方位变化增强 LoRA：含干净保真辅助项的历史版本|`g5/rotated_complete_s{s}`|`artifacts/g5_rotation_v1/rotated/train/complete_s{s}/adapter_03000.pt`|历史消融，区别于主对照|

注册表保留完整原始 SHA256；新图表应同时附原ID。不得改名/移动权重、覆盖日志或将rotated重新叙述为几何保持。G30/G34存在另外的训练档案，不在当前共享runtime这27个历史适配器入口内；这不表示它们未训练或不存在。

## 3. 实际训练分布与采样

直接读取 `data/g15_v1/train_catalog.json`（与G5 rotated一一配对）。2048条重复裁剪混合来自212句、4位训练说话人，不是2048独立语句；`artifacts/g15_rigid_training_v1/PREPARATION_COMPLETE.json` 已记录并验收。

- 目标角：G5 fixed全为20°；counter与rigid复用同一2048个预抽取角，生成器为`Uniform(-45,75)`，实际最小−44.955422°、最大74.991644°、均值15.731359°。
- 按从−45开始的12个10°箱，池内数量依次为147、197、183、157、166、149、170、164、183、172、174、186。有限池不严格均衡；生成器的连续均匀分布不等于每个角度等量曝光。
- 场型：point512、multi8 1536。噪声录音：DKITCHEN694、DWASHING671、OOFFICE683；SNR：−5dB649、0dB713、+5dB686。噪声固定40Hz高通，训练噪声片段来自0—120秒。
- point分类字段`angle`取−90/−60/−30/0/30/60/90，池数量67/75/95/70/76/60/69。**这些是基准噪声方位分类，不是目标方位！** 目标基准20°，所以rigid点噪声相对水平角实际为−110/−80/−50/−20/+10/+40/+70°（噪声俯仰−15°，语音俯仰+10°）。multi8使用八个独立方向的平面波相加，不应称真实漫射声场。
- `scripts/g3_train.py:64—88`：without_clean_identity/complete会先以0.5概率选point，再均匀选7个基准噪声分类，再在该类内抽样；否则在multi8中均匀抽样。有放回抽取3000次。**现有分层不是按target_azimuth分层**，没有预留20°目标角曝光比例。target角在预生成池上分配，被噪声分类的重采样间接再加权。
- 已归档实际point/multi8采样次数：20261112为1459/1541，20261113为1563/1437，20261114为1518/1482。G15 `training_audit.json`核对它们与同种子G5 fixed/counter分类次数及NumPy、CPU/CUDA随机状态全部相同。当前训练日志主要记录场型/旧噪声类别次数，没有逐步目标角曝光直方图；不冒充已取得真实更新级目标角均衡证据。
- 源裁片、噪声片段、target角序列、监督参考、旧共同尺度、同种子采样、优化器及预算受控。G15只改噪声旋转符号；全部2048旧输入重放精确，新旧第5麦训练输入相同，train_y与clean_x文件字节相同。G15 FLOAT32输入有9条abs>1，最高1.129169；保留浮点不静默削波。
- 公共训练条件：rank4/alpha4，3584参数，Adam，lr3e−4，batch1，无梯度累积，3000有效更新，clip5，AMP关闭，1秒核心+128样本保护。主对照without_clean_identity的辅助保真权重为0；complete每5步另加0.1权重的干净重建项。
- 检查点：G5按54旧验证例SI-SDR与STOI门选，但所有六个主对照实际都选3000；G15事前规定3000末步注册，保留旧引擎selected仅诊断，不能改写为两者都事前固定末步。G15验证6句/2人，诊断用不择权重。

## 4. 正确几何和19角协议接口

- `src/mcse/geometry.py:5—16`：x向右、y向前、z向上，正方位向右，`azimuth=atan2(x,y)`；正确矩阵为x′=cosδ·x+sinδ·y、y′=−sinδ·x+cosδ·y。`scripts/g5_prepare.py`旧rotate符号相反，只能留作历史复现。
- `scripts/g14_build.py:45—73`、`scripts/g17_build.py:11—63`正确使用rotate_azimuth，核验目标/噪声点积保持、点源实际相对角、参考麦及目标相等、共用尺度；`tests/test_azimuth_geometry.py`已有正方向、组合逆变换及相对几何测试。
- `configs/g1_geometry.json`是名义CHiME3平板六麦近似坐标，声速343m/s，参考第5麦；自由场全向平面波。未模拟真实平板遮挡、麦频响或真实噪声/房间。目标方位扫描若保持语音俯仰+10°，必须写成固定俯仰下的水平面方位扫描，不宣称覆盖三维声场。
- 19点为−90至+90每10°；训练名义覆盖[−45,+75]内的测试点共12个（−40…+70），范围外7个（−90、−80、−70、−60、−50、+80、+90）。不能只给两个分组均值而漏原始19点，也不能将12点离散结果称连续鲁棒。
- 未来用同一源、噪声片段、基准多源方向、SNR和一次生成的输入给全部模型；对整组19角共同确定防削波尺度，禁止逐方向/模型重新归一化。参考位置相对时延为0，有利于检查19角第5麦与干净参考一致，避免电平混杂。正确旋转Δ=新目标角−20°只是沿用基准坐标，**不代表20°最优或最终锚点**。
- `runtime.py:41—55`默认6通道16kHz、512样本至4秒，拒绝隐式重采样、复制通道和静默分块。若未来用更长完整句，需在另行批准的资源/接口核查后登记长度上限；不能未经核查把默认4秒当任意长度支持。
- `runtime.py:94—107`共享增强路径直接返回单通道，未接MVDR/MWF；ASR不在增强推理里。

## 5. 执行精度及解释边界

`scripts/g3_train.py`关闭AMP，日志写FP32，但没有显式冻结TF32许可。`docs/G33B_科学读回与统一执行精度建议.md`显示cuDNN TF32许可会影响LoRA替换层与全参数图的数值差异；原前向门槛失败和cIRM小残差未被消除。不能声称旧历史训练已经达到G34后来规定的统一后端许可。`scripts/g34_common.py:20—31`提供进程内的明确设置（cuDNN启用、matmul/cuDNN TF32关闭、benchmark=False等），当前共享runtime本身没有实施该configure。

因此后续19角正式协议应预登记同一进程内后端策略、版本、dtype、重建和评价条件，并在少量既有输入上验证所有系统接口后再获准批量扫描。选择显式后端策略属于可审查协议，不能据开发/测试成绩选精度。若改变后端重评，应另存新结果而非覆盖旧分数，也不把旧训练模型改写成该后端重训。

现有证据只能确认固定/多方位训练空间分布、采样目标和方向表现有差别。尚未直接验证平均损失导致方向退化、方向梯度冲突或rank4容量不足。若仅考察一种后续候选，应保持模型/损失/预算不变，单独控制目标方位训练曝光，并完整记录每步样本索引、目标角、噪声类别和场型；具体比例和锚点须经开发依据、冻结规则及用户批准后决定。
