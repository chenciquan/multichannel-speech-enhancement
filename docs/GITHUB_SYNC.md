# GitHub 项目入口与同步约定

仓库：[chenciquan/multichannel-speech-enhancement](https://github.com/chenciquan/multichannel-speech-enhancement)，保持私有。

## ChatGPT / Codex 快速读取

先读根 `AGENTS.md` 与 `docs/CODEX_PROJECT_MEMORY.md`，再按需读 `docs/G35_准入及19角信号扫描交付报告.md`、相关阶段报告及 `artifacts` 中的可读记录。仓库访问仍需已连接的 GitHub 账号具有该私有仓库权限；不会因为上传而自动读入全部文件。

代码、配置、全部历史中文报告、研究清单/JSON/JSONL/日志、CSV、图表及本地适配权重直接入库，便于按路径核查。既有远端初始提交保留，本地旧 README 与项目记忆规则留在 `docs/history_before_github_20261113/`，不伪造逐阶段历史提交。

## 完整科研文件备份

`data/`、`artifacts/`、`third_party/` 的完整科研文件以 ZIP 分卷保存至本仓库 Release：`research-snapshot-g35-20261113`。初次同步期间 Release 可能仍为草稿，以远端正式 Release 和最终同步记录为完成依据。

归档共有29个独立ZIP。为避免大请求长连接中断，已上传的完整ZIP保留，剩余ZIP采用256MiB传输分片。`release-transport-manifest.json`记录各ZIP对应的完整资产或有序分片，分别附SHA256；`research-file-manifest.json`记录每个原文件的路径、大小、SHA256及所在ZIP，`SHA256SUMS.txt`用于拼接后的29个原ZIP校验。

恢复流程：先将对应Git提交克隆到新目录；下载Release的全部附件到另一个备份目录，运行`python scripts/release_transport.py --restore 备份目录`。工具核验分片并无损拼接出29个ZIP，拒绝覆盖或接受不符哈希的已有文件。随后将全部ZIP解压到新克隆的项目根目录，保持`data/`、`artifacts/`、`third_party/`相对路径。不要解压覆盖已有研究工作区；与Git重叠的可读文件应保持相同字节。

本次归档使用已有原生Python标准库（只读现有科研文件），上传使用`scripts/github_release_parallel.py`的三个有界流式连接；已完成资产按GitHub提供的SHA256校验复用。身份验证只在内存中读取Git凭据，不写入脚本、命令行或仓库。未完成的本次草稿上传临时资产可重传，绝不替换已完成或历史Release资产。

`.venv/`、工具缓存、下载缓存、Python字节码和根Git内部文件不是科研备份对象，不上传；依赖锁定、环境报告、官方来源/提交记录保留。官方 McNet 子仓库及权重属于私有科研备份，不据此推定公开再分发授权。没有启用 Git LFS 或付费存储。

## 此后每个大任务结束

用户已持续授权任务结束后更新本仓库：

1. 保存完整结果、负结果、验收和来源；更新项目摘要。
2. 检查 Git 差异、敏感文件、文件体积和相关验证结果，提交本任务的代码、文档及可读产物，推送到已确认的远端；不捎带无关用户修改，不强制推送。
   `data/`和`artifacts/`默认仍被忽略，新增可读成果需按本任务明确路径使用`git add -f -- <路径>`加入；不得直接强制加入整个数据目录。已跟踪文件的正常修改可用普通`git add`。`.gitattributes`禁用换行转换，保持冻结文件的原始字节SHA。
3. 需要保存的新大文件以新的只追加 Release/分卷归档处理；记录清单及远端位置，不覆盖旧卷，不擅自使用付费服务或改变仓库可见性。
4. 核对远端提交与关键文件，报告提交SHA/链接后停止。网络或权限失败时说明实际状态，不把本地提交说成已上传，也不因此恢复任何实验队列。

Git更新授权只用于保存已经获批任务的成果，不授权新的训练、推理或实验分支。
