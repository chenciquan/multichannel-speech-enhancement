#!/usr/bin/env python3
"""Sequential bounded calibration/training, archive before development evaluation."""
import subprocess
import sys
import time
import traceback
from g30_common import ROOT,OUT,sha,dump,now,meta,guard,status,archive


def main():
    plan=guard();start=time.monotonic()
    jobs=([('calibration',['scripts/g30_calibrate.py'],plan['max_calibration_seconds'])]+
        [(variant,['scripts/g30_train.py',variant],plan['max_run_seconds']) for variant in plan['variants']])
    for label,args,bound in jobs:
        assert time.monotonic()-start<plan['max_queue_seconds'];status(stage='running',job=label)
        with (OUT/(label+'.log')).open('x',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                timeout=min(bound+30,plan['max_queue_seconds']-(time.monotonic()-start)),check=False)
        assert result.returncode==0,f'{label} failed, see preserved log'
        if label=='calibration':assert meta(OUT/'CALIBRATION_COMPLETE.json')['status']=='ok'
        else:
            marker=meta(OUT/'train'/label/'result.json');assert marker['status']=='ok' and marker['optimizer_updates']==1500
            report=ROOT/'docs'/f'G30_{label}_训练覆盖与资源.md'
            with report.open('x',encoding='utf-8') as f:f.write(f"# G30 {label} 训练覆盖\n\n单种子固定1500次更新已完成，保存500/1000/1500步，登记1500步。原主干未改变，与校准初始化及共享抽样一致，尚未访问开发或测试音频。Torch已分配峰值{marker['peak_allocated_bytes']/2**30:.3f}GiB，实际耗时{marker['elapsed_seconds']:.1f}秒。仅训练完成，不作方法改善结论。\n")
            archive('train_'+label,report,[OUT/'train'/label/'result.json',OUT/'train'/label/'config.json',ROOT/'scripts/g30_train.py'])
    guard();dump(OUT/'TRAINING_COMPLETE.json',{'status':'ok','utc':now(),'variants':plan['variants'],
        'updates':4500,'elapsed_seconds':time.monotonic()-start,'development_evaluated':False,
        'result_hashes':{v:sha(OUT/'train'/v/'result.json') for v in plan['variants']}},True)
    status(stage='training_archived_pending_acceptance')


if __name__=='__main__':
    try:main()
    except Exception as e:dump(OUT/'queue_failure.json',{'utc':now(),'error':repr(e)});traceback.print_exc();raise
    finally:(OUT/'keep_awake_stop').write_text('G30 calibration/training queue ended\n')
    for _ in range(30):
        if (OUT/'power_helper.json').exists() and meta(OUT/'power_helper.json')['status']=='released':break
        time.sleep(2)
    from g30_accept import main as accept
    accept();print('G30 calibrated objective training accepted',flush=True)
