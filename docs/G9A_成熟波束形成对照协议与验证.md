# G9A 成熟波束形成对照协议与验证

G8最终验收归档后登记本协议。补足成熟方法对照，不把现有旋转训练或LoRA当作新颖算法。G5/G7已查看数据用于探索性比较，不冒充新的盲测集。全部900条件/系统，包含G5的36个干净对照，3个新增系统共2,700条。源句仍为同12句/3人。

主对照为McNet估计掩码Souden MVDR：仅接收六通道输入和原McNet保存的单通道估计，估计噪声为参考麦克风减该估计，平方比掩码估计两个PSD，通过官方Souden权重及共轭波束形成得到输出。主函数不接收目标参考或方位。

两个额外诊断分别使用真实参考掩码、模拟器真实DOA延时求和，均标为oracle。它们使用额外真值，不能与可部署方法当作公平竞赛，也不是数学性能上界。

固定16k、512点周期Hann、128hop、center/reflect；CPU float64/complex128；PSD normalize及eps1e-10，Souden对角加载1e-7和eps1e-8，参考索引4。保存float32 FLOAT WAV，无归一化、拟合或错误回填。公式与数值在测试前登记。

五项前置测试通过：STFT准确长度还原、相同通道保留参考、静音退化协方差有限、独立平面波渲染器验证DS延时符号且目标SI>25dB、非法形状/非有限值拒绝。协议与测试快照先保存，再运行实际900条件。

指标沿用SI-SDR/PESQ/STOI和Whisper/CTC；全部适配器种子保留，不挑最优种子。滤波CPU耗时单列，主对照还需原McNet估计，其生成耗时另列，不能只报MVDR时间作为整条流程成本。

依据：[官方2.5 Souden API](https://docs.pytorch.org/audio/2.5.0/generated/torchaudio.functional.mvdr_weights_souden.html)、[官方教程](https://docs.pytorch.org/audio/2.5.0/tutorials/mvdr_tutorial.html)。经典论文：Souden等，On Optimal Frequency-Domain Multichannel Linear Filtering for Noise Reduction，IEEE TASLP18(2),260–276，2010年2月；[DOI](https://doi.org/10.1109/TASL.2009.2025790)。官方引用的2009对应在线发表记录。我们复用现有库，不安装新依赖、不下载新语料、不训练模型。

UTC：2026-11-12T04:22:23.547819+00:00
