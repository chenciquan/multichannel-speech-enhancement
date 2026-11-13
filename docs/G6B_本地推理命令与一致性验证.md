# G6B 本地推理命令与一致性验证

G6A资源报告归档后开展。命令复用原McNet、原解码器和现有loralib，不另写增强模型或安装依赖；不启动服务或公开发布。

输入为六个真实麦克风通道、16kHz、512—64000样本；参考第5麦。不得把单麦复制六份。输出单通道FLOAT WAV，保留幅度与长度，不自动重采样、归一化或分块。只加载由G3/G5归档清单登记且SHA匹配的适配器；未知/篡改权重拒绝加载。

## 命令

在项目WSL目录中执行：

```bash
.venv/bin/python scripts/enhance_registered.py --list-systems
.venv/bin/python scripts/enhance_registered.py --input /path/to/six_channel.wav --output artifacts/my_run/enhanced.wav --report artifacts/my_run/provenance.json --system g5/rotated_without_clean_identity_s20261112 --device cuda
```

示例系统只说明调用方式，不代表全条件最佳。默认`--system mcnet`为原始基线；也可显式选择CPU。输出与报告必须是不同的新路径，已有文件会在模型加载前被拒绝。超过4秒不静默分块，较长录音尚需另立上下文/拼接验证方案。

## 实际验证

固定选择G5非干净确认集的第一条ID，没有按结果挑选。独立CLI进程输出与已有G5保存波形比较：

|系统|设备|最大绝对差|允许值|独立进程墙钟秒|
|---|---|---:|---:|---:|
|mcnet|cuda|0|1e-05|24.52|
|g5/fixed_without_clean_identity_s20261112|cuda|0|1e-05|26.02|
|g5/rotated_without_clean_identity_s20261112|cuda|0|1e-05|26.01|
|g5/fixed_complete_s20261112|cuda|0|1e-05|25.97|
|g5/rotated_complete_s20261112|cuda|0|1e-05|25.92|
|mcnet|cpu|0.000279426575|0.001|29.32|

全部通过。GPU允许最大绝对差1e-5，CPU跨后端1e-3仅作数值兼容界限；不是音质改善幅度。真实二次调用验证了拒绝覆盖且原字节不变。单测还检查错误采样率/通道/长度、非有限与溢出/复数、输出路径冲突及注册清单篡改。完整单测日志和逐次CLI日志保留。

独立进程墙钟包含Python/库导入、GPU初始化、核验、加载、IO和计算，因此明显高于G6A纯计算耗时，不能将两者混报。命令没有自动运行ASR或质量指标，也不保证真实房间效果。原模型为离线双向结构，不能因此宣称流式部署完成。

接口代码为`scripts/enhance_registered.py`、`src/mcse/runtime.py`；测试/QA和注册模型哈希见`artifacts/g6_local_runtime_v1/prototype/`。用户使用许可仍按既有本地研究范围，不推定公开再分发授权。
