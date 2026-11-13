#!/usr/bin/env python3
"""Preserve pre-install tool mismatch evidence before revising G12 execution."""
import json
from pathlib import Path
from g12_common import ROOT,OUT,now,dump,archive
from g12_asset import installed


def main():
    before=json.loads((OUT/'assets/distributions_before.json').read_text());unchanged=installed()==before
    audit={'status':'documented_preinstall_failure','package_versions_unchanged':unchanged,'wheel_files':len(list((OUT/'assets').glob('*.whl'))),
      'room_lock_created':(ROOT/'requirements-room.lock.txt').exists(),'reason':'Existing uv-managed venv has no pip module; script stopped at freeze_before,prior to HTTP wheel fetch or install.',
      'existing_package_manager':'.tools/uv-x86_64-unknown-linux-gnu/uv','closed_utc':now()}
    assert unchanged and audit['wheel_files']==0 and not audit['room_lock_created'];dump(OUT/'PREINSTALL_FAILURE_AUDIT.json',audit,True)
    report=ROOT/'docs/G12A_前置包管理检查失败记录.md';report.write_text('# G12A 前置包管理检查失败记录\n\n首版安装脚本错误假设venv内有pip。实际项目此前使用自带uv，venv没有pip是既有状态，不能解释为环境损坏。程序在freeze_before阶段报No module named pip，尚未请求wheel或安装；全部发行包版本与调用前逐项相同，无wheel文件、无新锁文件。\n\n保留失败日志、冻结协议、调用前包版本、实际脚本及本审计。不删除或覆盖首版目录。后续修订仅沿用原有.tools/uv-x86_64-unknown-linux-gnu/uv，使用明确--python、离线--no-index/--no-deps/--no-build及禁止Python下载，不补装pip/编译器/构建工具。使用新的运行版本目录，声学条件与模型协议不变。\n\n此失败不是数值或效果失败，没有产生RIR/模型成绩。\n\nUTC：'+now()+'\n',encoding='utf-8')
    archive('failure',report,[OUT/'frozen_plan.json',OUT/'PREINSTALL_FAILURE_AUDIT.json',OUT/'asset.log',OUT/'assets/distributions_before.json',ROOT/'scripts/g12_asset.py',ROOT/'scripts/g12_common.py',ROOT/'scripts/g12_start_asset.py',Path(__file__)])
    print(json.dumps(audit))


if __name__=='__main__':main()
