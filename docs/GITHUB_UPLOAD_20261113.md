# GitHub完整科研项目同步交付记录

日期：2026-11-13。任务依据：用户明确要求将整个科研项目保存至本人GitHub，并在以后每个获批大任务完成后更新Git。本次只做版本保存和数据完整性核查，没有训练、推理、ASR、下载数据或安装依赖，也没有恢复G34。

## 1. 已完成的保存

- 私有仓库：[chenciquan/multichannel-speech-enhancement](https://github.com/chenciquan/multichannel-speech-enhancement)，主分支`main`。保持原有可见性，没有使用Git LFS或付费存储。
- 代码、配置、全部中文报告、项目记忆、登记清单、可读指标/日志、图表和适配权重已直接提交，便于按路径读取。
- 完整科研文件：[research-snapshot-g35-20261113正式Release](https://github.com/chenciquan/multichannel-speech-enhancement/releases/tag/research-snapshot-g35-20261113)，已发布，非草稿。
- `data/`、`artifacts/`、`third_party/`共**106409个文件，29620353853字节**，打包为29个独立ZIP，ZIP总计**25862710012字节**。其中包含原音频、数组、全部历史结果和模型权重，以及官方McNet源码、检查点和内部Git记录。
- 远端实际有**90个附件**：9个完整ZIP、78个不超过256MiB的传输分片，以及3个清单文件。附件总计25894241516字节。剩余20个ZIP由分片无损拼接恢复。

没有将虚拟环境、便携工具、包缓存、上传缓存、字节码和根Git内部文件作为科研数据上传。依赖锁定文件与环境/来源报告已保存。官方源码的网页入口为[固定上游提交](https://github.com/Audio-WestlakeU/McNet/tree/9c194032659a3047a4b4c244a9f541f4d34b1682)，本地原件仍在完整归档中；不推定公开再分发授权。

## 2. 版本与历史保护

保留远端原始提交`ba289f73e90982d7315bf7b7dc443484afbf3932`，随后创建当前科研快照`4731e3f3a1b41b5f6e632c34b2af393aa4bd4a0e`，没有伪造G0—G35的逐阶段Git历史。Release标签对应`766265669f348b9bb5efed11be96a702b026837b`，该版本包含分片恢复工具；后续提交主要补充上传工具和管理说明。

旧README、AGENTS、项目记忆及Git忽略/属性文件的原版保存在`docs/history_before_github_20261113/`。历史研究报告、几何错误记录、负结果、训练种子、模型和评分保持原状。`.gitattributes`禁用自动换行转换，避免冻结文件入Git后改变字节哈希；曾逐项确认当时3870个暂存文件的Git对象与原始字节一致。

## 3. 实际核验结果

|核验|实际结果|证据|
|---|---|---|
|全部原文件与ZIP|106409个成员全部CRC、逐文件SHA256及29卷SHA256通过|[archive_validation.json](backup_manifests/20261113/archive_validation.json)|
|分片恢复|实际拼接第29卷，3片恢复551705935字节ZIP，SHA256与原卷一致，原卷未改|[restore_test.json](backup_manifests/20261113/restore_test.json)|
|项目文件覆盖|收尾说明新增前107020个项目文件，全部属于Git或科研归档清单，无漏项|[snapshot_coverage.json](backup_manifests/20261113/snapshot_coverage.json)|
|远端独立检查|90个附件均为uploaded，大小和GitHub提供的SHA256全部匹配；私有状态、正式发布状态、恢复工具及标签祖先关系通过|[remote_assets_verified.json](backup_manifests/20261113/remote_assets_verified.json)|

远端独立核验时间为2026-11-13 20:10:53北京时间，当时本地与远端`main`同为`41e33690e5edde15bda88a799f2ce6a223bcabf5`。本交付记录及最终同步说明随后通过正常提交加入`main`；最终提交SHA以仓库提交记录为准。

本地全量ZIP核验耗时163.125秒；完整逐文件清单见[research-file-manifest.json](backup_manifests/20261113/research-file-manifest.json)，分片对应关系见[release-transport-manifest.json](backup_manifests/20261113/release-transport-manifest.json)。远端核验采用GitHub返回的文件摘要，与本地完整内容校验相对应。

上传曾遇到TLS连接中断和Windows进度文件共享锁，保留已完成附件并断点续传；大请求改为有界分片，进度写入增加短暂共享锁重试，最终使用3路进程内直连完成。未修改系统代理或放宽TLS/哈希校验；只清理了本次草稿中未完成的上传临时资产，没有替换历史或已完成附件。本地归档及诊断缓存保留在`.cache/github_upload_20261113/`，未做删除清理。

## 4. 读取与恢复

ChatGPT/Codex按顺序读取`AGENTS.md`、`docs/CODEX_PROJECT_MEMORY.md`，再按需读取G35交付及对应阶段证据。GitHub连接必须具有该私有仓库权限；文件不会因为上传而自动全部进入对话。

完整恢复时，将仓库克隆到**新目录**，把Release全部附件下载到另一个备份目录，然后运行该仓库已有工具：

```text
python scripts/release_transport.py --restore 备份目录
```

工具检查每个分片及拼接后的ZIP哈希，拒绝覆盖不符的已有文件。随后将29个ZIP解压到新克隆的项目根目录，保持相对路径。不要解压覆盖现有研究工作区。详细说明见[GitHub同步约定](GITHUB_SYNC.md)。

## 5. 后续规则与停止边界

每个经批准的大任务结束后，保存完整结果、更新项目记忆、检查差异及敏感信息、提交并推送本任务相关成果；新大文件采用新的只追加归档，不覆盖本次Release，不强制推送。持续授权已记录于`AGENTS.md`和`docs/GITHUB_SYNC.md`。

此授权仅用于版本保存，不扩大实验范围。G35 A/B/C与三项曲线保持既有完成状态；没有新增ASR或候选训练，G34保持停止。此次GitHub保存完成后停止，等待用户下一项指令。
