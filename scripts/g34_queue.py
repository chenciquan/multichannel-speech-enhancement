"""One bounded sequential training queue; archive each run before advancing."""
import subprocess
import sys
import time
import traceback
from g34_common import ROOT,OUT,sha,meta,dump,now,guard,archive,status


def main():
    plan=guard();started=time.monotonic();results={}
    jobs=[('initial',['scripts/g34_initial.py'],plan['max_initial_seconds'])]+[
        (r['id'],['scripts/g34_train.py',r['id']],plan['max_run_seconds']) for r in plan['runs']]
    for label,args,bound in jobs:
        remaining=plan['max_queue_seconds']-(time.monotonic()-started);assert remaining>0
        status(stage='running',job=label,completed_training_runs=len(results))
        with (OUT/f'{label}.log').open('x',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                timeout=min(bound,remaining),check=False)
        assert result.returncode==0,f'{label} failed, inspect preserved log'
        if label=='initial':
            report=ROOT/'docs/G34A_训练前六条波形接口核查.md';r=meta(OUT/'INITIAL_COMPLETE.json')
            with report.open('x',encoding='utf-8') as f:f.write(f"# G34A 训练前六条波形接口核查\n\n同六条训练输入、统一精度设置，重建波形通过原容差{r['wave_passes']}/6，压缩cIRM通过{r['cirm_passes']}/6；基础参数摘要相同，LoRA零B、参数未更新。只通过本次波形接口条件，不改判旧G33失败，也不保证所有训练输入逐位一致。实际配置、逐例差异及原始预测/波形均保存。\n")
            archive('initial',report,[OUT/'INITIAL_COMPLETE.json',OUT/'initial_records.json'])
        else:
            r=meta(OUT/'train'/label/'result.json');assert r['status']=='ok' and r['optimizer_updates']==1500
            results[label]=sha(OUT/'train'/label/'result.json');report=ROOT/'docs'/f'G34_{label}_固定预算训练记录.md'
            with report.open('x',encoding='utf-8') as f:f.write(f"# G34 {label} 固定预算训练记录\n\n固定1500次原cIRM更新已完成，保存500/1000/1500步，只登记1500步，无开发/测试评分。开放参数{r['run']['parameters']}；实测含Adam状态的Torch峰值已分配{r['peak_allocated_bytes']/2**30:.3f}GiB、保留{r['peak_reserved_bytes']/2**30:.3f}GiB；同步逐步计时中位数{r['warm_step_median_seconds']:.6f}秒，总墙时{r['elapsed_seconds']:.1f}秒（含导入后权重加载、校验和I/O），两者不可混称GPU纯计算时间。有限性、最终参数摘要、全部日志和checkpoint保存，最终独立验收尚待完成；不作语音效果结论。\n")
            archive('train_'+label,report,[OUT/'train'/label/'result.json',OUT/'train'/label/'config.json',ROOT/'scripts/g34_train.py'])
    guard();dump(OUT/'TRAINING_COMPLETE.json',{'utc':now(),'status':'ok','runs':4,'updates':6000,
        'result_hashes':results,'elapsed_seconds':time.monotonic()-started,'no_development_or_test_audio':True},True)
    status(stage='training_complete_pending_acceptance')


if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(OUT/'queue_failure.json',{'utc':now(),'error':repr(e)})
        status(stage='failed_preserved_no_automatic_retry');traceback.print_exc();raise
    finally:(OUT/'keep_awake_stop').write_text('G34 bounded queue ended\n')
    for _ in range(30):
        if (OUT/'power_helper.json').exists() and meta(OUT/'power_helper.json')['status']=='released':break
        time.sleep(2)
    try:
        from g34_accept import main as accept
        accept()
    except Exception as e:
        dump(OUT/'acceptance_failure.json',{'utc':now(),'error':repr(e)})
        status(stage='acceptance_failed_preserved');traceback.print_exc();raise
    print('G34 fixed training accepted; development not started',flush=True)
