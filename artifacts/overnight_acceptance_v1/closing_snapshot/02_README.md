# 多通道语音增强本科毕业设计

技术路线见 [研究报告](docs/本科毕业设计_研究路线与创新方向分析.md)。截至2026-11-12，**G3—G5已完成并归档，本轮已停止**。[简明验收汇总](docs/夜间实验_验收汇总.md)是最新入口。完成24次标准低秩训练、72,000次有效更新及12,210个信号/识别对，审计与窗口后文件回查通过。原先吞字恢复假设没有获得支持。

G2历史原因分析见 [原因分析与改进方向](docs/G2_原因分析与改进方向.md)，当前方案状态见 [规格](configs/improvement_direction.json)。原噪声的极低频能量主导全频SNR，造成首轮场景难度与指标解释偏差；修正噪声频谱后，空间结构与入射角影响模型表现，8条新语句、两个空间种子确认了当时模拟范围内的差异。两处空间FC的标准低秩适配、分层采样和保真约束随后在G3—G5实测；组件均有先前研究，不宣称创新已确立。

最新方法证据见[G5新说话人确认](docs/G5_整场景旋转与新说话人确认报告.md)：旋转增强主要改善−40°退化，但20°保持与70°识别仍有代价，尚无全条件赢家。LoRA、采样均衡、旋转与干净重建均是已有思路，不认定创新已成立。G6资源基准/独立CLI未执行，不能算已完成。

G3已保存[总报告](docs/G3_五小时实验验收报告.md)、训练与信号阶段快照：36,000次有效更新、5,850个信号对和识别对，审计通过。完整方案相对普通适配平均WER降低2.28个百分点、SI-SDR降低0.62dB，呈现权衡，不能认定整体更优或创新已成立。连续授权及归档入口见[阶段总览](docs/连续实验_阶段总览.md)。

G2使用6条旧发现语句和8条全新确认语句，272个条件混合；条件和种子不能当作独立语句。G3新增说话人隔离的训练/验证/测试，排除旧32句；55项测试通过，首组FP32训练已完成3,000步并验证冻结基线未改变。所有G3结果以最终验收报告及逐例文件为准。

原始自动诊断见 [G1首轮诊断报告](docs/G1_首轮诊断报告.md)、[G1后续核查](docs/G1_问题核查与阶段决策.md)和[48条旧样本试听](artifacts/g1_v1/review_v2/index.html)。旧结果与13条反馈保留。新128条确认音频、识别结果和核查表见[新确认集试听页](artifacts/g2_confirmation_v1/review_v2/index.html)；这些新样本尚无人工逐词判定。

## 本地开发约定

- 在 WSL2 Ubuntu 中运行，使用当前项目内唯一的 `.venv`；系统 Python、NVIDIA 驱动和系统 CUDA 配置不由项目脚本修改。
- `third_party/McNet` 是固定提交的官方稀疏检出，保留原始代码。外部接口兼容适配与算法改进分开记录。
- 数据、权重、工具和生成结果留在本机，由 `.gitignore` 排除。用户已表示取得 McNet 使用许可；具体许可范围尚未归档，不推定具有公开再分发授权。
- 首轮实际24条源语音、48个模拟六通道短混合、一个ASR、无训练。模拟定向噪声不等于真实阵列声场；人工核查与独立确认前，不声称发现稳定人声误删。

## 目录

```text
configs/                  固定诊断协议与基线来源
docs/                     研究报告、阶段记录与数据规范
scripts/                  环境准备、推理、受控适配、评估与归档
src/mcse/                 本项目的诊断基础工具，不包含重写的增强模型
tests/                    数据约束与泄漏检查的标准库测试
data/sources/g1_v1/        小规模公开语音/噪声及逐文件来源清单
data/diagnostic/g1_v1/     冻结的48条发现样本及协议
artifacts/g1_v1/           增强、ASR、完整指标、核查材料及交付审计
data/g3_v1/               G3说话人隔离来源、训练池和保留测试
artifacts/g3_overnight_v1/ G3训练、选权重、逐例评估与阶段快照
third_party/McNet/        本地官方源码与离线权重（不纳入项目分发）
.venv/                    唯一WSL虚拟环境
artifacts/environment/    实测环境、哈希、运行报告与依赖快照
```

## WSL 中的命令

先进入项目目录：

```bash
cd /mnt/c/Users/chenciquan/Desktop/multichannel_speech_enhancement
```

首次准备及安装：

```bash
bash scripts/bootstrap_wsl.sh
bash scripts/install_inference.sh
bash scripts/fetch_mcnet.sh
```

`bootstrap_wsl.sh` 下载固定版本uv并校验SHA256，用现有Python3.12建立`.venv`。原Python3.10候选因下载504暂未采用。PyTorch CUDA运行库由官方wheel提供，不需要另装Linux NVIDIA驱动或完整CUDA Toolkit。

安装状态及真实命令结果以 [阶段记录](docs/开发阶段记录.md) 为准。脚本存在不等于安装/模型验证已经通过。

WSL中的代理仅按进程传入`HTTP_PROXY`/`HTTPS_PROXY`，不修改系统配置。安装脚本默认使用WSL原生临时缓存`/tmp/mcse-uv-cache-01a1abd4`以避免已发生的NTFS重命名问题，可用`MCSE_UV_CACHE_DIR`覆盖。缓存可在WSL重启/清理后消失，环境与记录仍在项目内。

本次用户提供的代理可用于单次下载命令，例如：

```bash
HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
ALL_PROXY=http://127.0.0.1:7897 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
bash scripts/install_inference.sh
```

`requirements-inference.lock.txt`记录G0时的77个包；`requirements-diagnostics.lock.txt`记录G1加入ASR、指标和绘图后的92个包。诊断安装脚本以G0锁文件约束已有依赖。G1为编译PESQ安装了gcc、Python3.12开发头及依赖，详见阶段记录。原模型推理通过不代表原Trainer训练/测试CLI已验证。

G3在同一环境增加`loralib==0.1.2`，93包快照为`requirements-adaptation.lock.txt`。G3使用独立、受控的小参数训练入口，不声称已经复现官方完整训练流程。首次AMP长跑数值失败保留，正式对照统一FP32。

运行环境与合成输入检查：

```bash
.venv/bin/python scripts/check_runtime.py --require-cuda --report artifacts/environment/runtime.json
.venv/bin/python scripts/mcnet_infer.py --smoke-seconds 1 --device cpu --report artifacts/environment/mcnet_cpu_smoke.json
.venv/bin/python scripts/mcnet_infer.py --smoke-seconds 1 --device cuda --report artifacts/environment/mcnet_cuda_smoke.json
```

合成随机信号仅用于检查权重加载、形状、有限值及显存，**不是人声保真实验**。默认拒绝覆盖已有输出；复测时选新的报告路径。模型推理报告包含真实配置和文件哈希、冷启动耗时、显存与未计算质量指标的说明。

真实输入示例（需要自行提供符合协议的音频，不能把单麦复制六份）：

```bash
.venv/bin/python scripts/mcnet_infer.py \
  --input data/diagnostic/noisy/example.wav \
  --output artifacts/inference/example.wav \
  --device cuda --report artifacts/inference/example.json
```

输入必须是六通道16kHz，参考索引4即第5麦。输出为单通道FLOAT WAV，不逐句峰值归一化、重采样或静默分块；默认拒绝超过已探测的4秒长度，扩展`--max-seconds`前应另测显存。单条`mcnet_infer.py`不计算质量/ASR；G1批量与补评脚本见阶段报告。PESQ补评以`artifacts/g1_v1/signal_scores/`为准，旧`mcnet/`中的缺后端记录保留原样。

清单与测试：

```bash
.venv/bin/python scripts/validate_manifest.py data/diagnostic/g1_v1/manifest.jsonl --check-audio --json
.venv/bin/python -B -m unittest discover -s tests -v
```

数据字段与阶段隔离见 [诊断数据规范](docs/诊断数据规范.md)。数据、权重和生成结果只在本机；G3已开始轻量适配训练，旧G1/G2诊断结果保留。旧诊断正式核查使用`review_v2/`，其中带噪播放器只播放第5麦；旧`review/`的六通道播放链接保留用于追溯，不用于研究者判断。

