# G8A 独立CTC识别器来源与资源验证

G7报告和审计已归档后激活。现有torchaudio加载官方WAV2VEC2_ASR_BASE_960H，没有安装依赖或更改增强权重。

[官方2.5版文档](https://docs.pytorch.org/audio/2.5.0/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html)说明：base模型、LibriSpeech训练子集960小时预训练及ASR微调，MIT许可。安装的2.5.1实现路径、标签、16k和不归一化设置已核对并保存源码哈希。文档说明不等于对原训练数据的独立审计。

论文：[wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://proceedings.neurips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)，NeurIPS2020。本文实验使用base贪心CTC，不引用论文大模型或其他解码配置的成绩为本项目结果。

官方HTTPS权重实收377,664,473字节，低于450MiB上限；本地SHA256 `488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d`。记录Content-Length/ETag；没有核实独立公布的发布者完整SHA，因此本地哈希只用于后续一致性。仅在完整下载并登记哈希后通过weights_only=True加载。

模型参数94,393,245，设备NVIDIA GeForce RTX 4060 Laptop GPU。batch1 FP32，无提示、外部语言模型、重采样或峰值归一化。CTC仍具有学习得到的语言规律，不能称为纯声学真值。

|探测输入|样本数|计算秒|峰值分配GiB|假设文本|
|---|---:|---:|---:|---|
|one_second|16000|0.0066|0.387|THE SETTING OF THE|
|three_seconds|48000|0.0268|0.409|THE SETTING OF THE SCENE SEEMED TO HER ALL IMPORTANT|
|fixed_actual_reference|49168|0.0146|0.411|THE SETTING OF THE SCENE SEEMED TO HER ALL IMPORTANT|

四项解码测试和两项配对计分测试通过：先折叠相邻重复、再去除blank，以保留被blank隔开的重复字母；不同参考长度下仍取微平均，缺配对直接拒绝。资源探测不是性能测试，未按识别结果挑选识别器。

下一阶段对已有G5/G7保存音频共12,600对重新计WER/S/D/I，使用原英语归一化，不重新训练增强器、不重新选择权重。分别讨论同一识别器下的方法差值；两识别器结论一致也不能替代人工声学验证。

证据：`artifacts/g8_cross_asr_v1/asset_provenance.json`、`capacity_probe.json`、`ctc_unit_tests.log`。

UTC：2026-11-12T04:12:11.751140+00:00
