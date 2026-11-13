#!/usr/bin/env python3
"""Freeze and execute bounded sequential matched-room signal phases."""
import os
import signal
import subprocess
import sys
import traceback
from g22_common import ROOT,OUT,BASE,sha,read,dump,now,archive,metadata,status


def main():
    assert metadata(OUT/'DATA_ACCEPTANCE_AUDIT.json')['status']=='ok'
    assert (OUT/'stages/next_signal_decision/archive_manifest.json').exists()
    names=['scripts/g22_common.py','scripts/g22_signal.py','scripts/g22_filter_worker.py','scripts/g22_signal_queue.py','scripts/g22_signal_finish.py',
           'scripts/g22_prepare.py','scripts/g4_run.py','src/mcse/runtime.py','src/mcse/adaptation.py','src/mcse/beamforming.py','src/mcse/wiener.py',
           'src/mcse/components.py','src/mcse/reverberation.py','src/mcse/evaluation.py',
           'artifacts/g18_room_stress_v1/systems.json']
    original=ROOT/'artifacts/g18_room_stress_v1/systems.json';registry=metadata(original)
    dump(OUT/'signal_systems.json',registry,True)
    ready={'frozen_utc':now(),'hashes':{n:sha(ROOT/n) for n in names},'max_phase_seconds':7200,
           'registry_sha256':sha(OUT/'signal_systems.json'),'registry_source_sha256':sha(original),
           'manifest_sha256':sha(BASE/'manifest.jsonl'),'RT0_output_max_abs':1e-5,'expected_total_rows':16488,
           'data_acceptance_sha256':sha(OUT/'DATA_ACCEPTANCE_AUDIT.json')}
    dump(OUT/'SIGNAL_PIPELINE_READY.json',ready,True)
    protocol=ROOT/'docs/G22C_完整系统信号评估冻结协议.md'
    with protocol.open('x',encoding='utf-8') as f:
        f.write('# G22C 完整系统信号评估冻结协议\n\n固定原有模型注册：11直接、20估计掩码滤波，各504条件；2真值统计滤波仅432带噪，共16488信号。所有9个适配器保持既定种子与末步3000。FP32/batch1，逐模型加载释放，两个单线程CPU指标/滤波工作进程；无新训练、参数选择或依赖。\n\n每阶段设置7200秒上限，完成报告和哈希归档后才进入下一阶段。无反射配对要求输出最大差≤1e−5，并保存精确相同比例。直接、盲滤波、真值统计相互分阶段；不在盲滤波中读取真实分量。真值诊断不作部署或理论上界。\n')
    archive('signal_protocol',protocol,[OUT/'SIGNAL_PIPELINE_READY.json',OUT/'signal_systems.json']+[ROOT/n for n in names])
    def run(args,limit,label):
        with (OUT/(label+'.log')).open('x') as log:
            process=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=limit)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        if code:raise RuntimeError(label+' failed; evidence preserved, no automatic retry')
    for phase in ('direct','blind','truth'):
        run(['scripts/g22_signal.py',phase],7200,'signal_'+phase)
        assert (OUT/'stages'/('signal_'+phase)/'archive_manifest.json').exists()
    run(['scripts/g22_signal_finish.py'],600,'signal_finish')


if __name__=='__main__':
    try:main()
    except Exception as e:status(stage='signal_failed',error=repr(e));traceback.print_exc();raise
    finally:(OUT/'signal_keep_awake_stop').write_text('matched-room signal queue ended\n')
