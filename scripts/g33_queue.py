import subprocess
import sys
import time
import traceback
from g33_common import ROOT,OUT,sha,dump,now,meta,archive,guard,status


def main():
    plan=guard();start=time.monotonic()
    for mode in plan['modes']:
        assert time.monotonic()-start<plan['max_queue_seconds'];status(stage='running',mode=mode)
        with (OUT/(mode+'.log')).open('x',encoding='utf-8') as f:
            r=subprocess.run([sys.executable,'scripts/g33_probe.py',mode],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                timeout=min(plan['max_mode_seconds']+30,plan['max_queue_seconds']-(time.monotonic()-start)),check=False)
        assert r.returncode==0,f'{mode} probe failed; inspect preserved log'
        marker=meta(OUT/mode/'COMPLETE.json');assert marker['status']=='ok' and marker['optimizer_updates']==0
        report=ROOT/'docs'/f'G33_{mode}_梯度资源覆盖.md'
        with report.open('x',encoding='utf-8') as f:f.write(f"# G33 {mode} 梯度资源覆盖\n\n六条固定训练输入的原cIRM目标梯度均有限非零，{marker['trainable_parameters']}个开放参数均连通，模型参数摘要不变。无优化器或参数更新、开发/测试访问。Torch已分配峰值{marker['peak_allocated_bytes']/2**30:.3f}GiB；完整比较与验收尚待另一模式完成。\n")
        archive(mode,report,[OUT/mode/'COMPLETE.json',OUT/mode/'records.json',OUT/mode/'initial_forward_outputs.npz'])
    guard();dump(OUT/'PROBE_COMPLETE.json',{'status':'ok','utc':now(),'modes':plan['modes'],'optimizer_updates':0,
        'elapsed_seconds':time.monotonic()-start,'result_hashes':{m:sha(OUT/m/'COMPLETE.json') for m in plan['modes']}},True)


if __name__=='__main__':
    try:main()
    except Exception as e:dump(OUT/'queue_failure.json',{'utc':now(),'error':repr(e)});traceback.print_exc();raise
    finally:(OUT/'keep_awake_stop').write_text('G33 resource comparison ended\n')
    for _ in range(30):
        if (OUT/'power_helper.json').exists() and meta(OUT/'power_helper.json')['status']=='released':break
        time.sleep(2)
    from g33_accept import main as accept
    accept();print('G33 full/LoRA no-update resource comparison accepted',flush=True)
