# G3 指标速览

所有指标均为本次实际计算；WER按编辑次数/参考词数汇总。

|系统|SI-SDR dB|PESQ|STOI|WER|成功/总数|
|---|---:|---:|---:|---:|---:|
|noisy|-1.439|1.121|0.6739|61.17%|360/360|
|hpf40|-0.469|1.124|0.6739|55.53%|360/360|
|mcnet|-1.609|1.162|0.6761|67.15%|360/360|
|ordinary_s20261112|4.670|1.475|0.8638|31.58%|360/360|
|complete_s20261112|4.566|1.511|0.8651|31.65%|360/360|
|without_stratification_s20261112|4.098|1.443|0.8580|33.67%|360/360|
|without_clean_identity_s20261112|6.314|1.568|0.8699|36.77%|360/360|
|ordinary_s20261113|5.623|1.486|0.8622|33.00%|360/360|
|complete_s20261113|4.848|1.515|0.8679|28.18%|360/360|
|ordinary_s20261114|5.027|1.486|0.8639|31.61%|360/360|
|complete_s20261114|4.047|1.480|0.8653|29.52%|360/360|
|without_stratification_s20261113|5.063|1.471|0.8608|31.84%|360/360|
|without_clean_identity_s20261113|6.143|1.549|0.8698|27.47%|360/360|
|without_stratification_s20261114|5.077|1.501|0.8632|31.39%|360/360|
|without_clean_identity_s20261114|5.338|1.536|0.8693|28.21%|360/360|

## 完整方案减普通适配（配对种子平均）

- si_sdr_db: -0.619661，探索性95%区间 [-0.672470, -0.566897]
- pesq_wb: 0.019708，探索性95%区间 [0.013118, 0.026838]
- stoi: 0.002779，探索性95%区间 [0.001905, 0.003660]
- wer: -0.022795，探索性95%区间 [-0.039751, -0.005781]

## 限制

- Short 1–3.8s English clean-source utterances, four held-out speakers, two test noise recordings.
- No room reverberation or real microphone recordings; nominal far-field plane-wave simulation only.
- Point-source test angle is coupled with SNR: +45 degrees only at 0 dB, -60 degrees at +/-5 dB; angle-specific causal conclusions are invalid.
- Multi8 is an unconditioned eight independent-segment sum, not a measured diffuse field.
- Single ASR is recognizer-specific and cannot prove acoustic word deletion.
- Test source counts per speaker are 10,10,9,1; the fourth speaker has only one utterance. Speaker-macro sensitivity is secondary and highly variable.
- LoRA, balancing, replay, and clean identity constraints are known ideas. Novelty is not established by beating original McNet.
- Source bootstrap intervals are exploratory given only four speakers; ablation seeds 2 and 3 were registered after first-seed validation, before any final-test evaluation.
