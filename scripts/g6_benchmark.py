#!/usr/bin/env python3
"""Fixed systems and lengths, actual synchronized compute/memory measurements."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import gc,json,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.runtime import load_system,enhance_tensor,sha256,read_input
OUT=ROOT/'artifacts/g6_local_runtime_v1';PLAN=json.loads((ROOT/'configs/g6_runtime_plan.json').read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    assert json.loads((ROOT/'artifacts/g5_rotation_v1/completion_audit.json').read_text())['status']=='ok'
    OUT.mkdir(parents=True,exist_ok=False);folder=OUT/'benchmark';folder.mkdir();start=time.perf_counter();deadline=start+PLAN['runtime_limit_seconds']
    rows=[json.loads(s) for s in (ROOT/'data/g5_v1/confirmation/manifest.jsonl').read_text().splitlines()]
    options=[r for r in rows if r['condition']=='point' and r['noise_id']=='OOFFICE' and r['target_azimuth']==20 and r['relative_angle']==-80]
    chosen=sorted(options,key=lambda r:(-sf.info(r['noisy_path']).frames,r['sample_id']))[0];audio=read_input(chosen['noisy_path']);torch.set_num_threads(2);torch.manual_seed(20261112)
    raw=[];metadata=[]
    for system in PLAN['systems']:
        if time.perf_counter()>deadline:raise TimeoutError('Benchmark time cap')
        torch.cuda.empty_cache();runtime=load_system(system);metadata.append(runtime.metadata)
        for seconds in (1,3):
            count=seconds*16000;clip=np.zeros((count,6),dtype='float32');clip[:min(count,len(audio))]=audio[:count]
            x=torch.from_numpy(np.ascontiguousarray(clip.T))[None].cuda()
            for _ in range(PLAN['warmup_per_system_duration']):y=enhance_tensor(runtime,x)
            assert torch.isfinite(y).all();del y;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(PLAN['timed_calls_per_system_duration']):
                torch.cuda.synchronize();began=time.perf_counter();y=enhance_tensor(runtime,x);torch.cuda.synchronize();times.append(time.perf_counter()-began)
            assert torch.isfinite(y).all()
            record={'system':system,'seconds':seconds,'calls_seconds':times,'mean_ms':1000*float(np.mean(times)),'p50_ms':1000*float(np.quantile(times,.5)),
              'p95_ms':1000*float(np.quantile(times,.95)),'compute_rtf':float(np.mean(times))/seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'input_zero_padded':len(audio)<count,'input_sample_sha256':__import__('hashlib').sha256(clip.tobytes()).hexdigest()}
            raw.append(record);dump(folder/'status.json',{'stage':'timing','system':system,'seconds':seconds});print(json.dumps({k:record[k] for k in ('system','seconds','mean_ms','p95_ms','compute_rtf')}),flush=True)
            del x,y
        del runtime;gc.collect();torch.cuda.empty_cache()
    result={'status':'ok','completed_utc':datetime.now(timezone.utc).isoformat(),'protocol':PLAN,'systems':metadata,'input_case':chosen['sample_id'],
      'input_path':chosen['noisy_path'],'input_file_sha256':sha256(chosen['noisy_path']),'measurements':raw,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
      'gpu':torch.cuda.get_device_name(0),'elapsed_seconds':time.perf_counter()-start,'no_quality_or_streaming_claim':True}
    dump(folder/'raw_timings.json',result)
    report=ROOT/'docs/G6A_本地推理资源实测.md';lines=['# G6A 本地推理资源实测','','G5报告和审计保存、用户审查认可后开展。固定原McNet与G5首种子的四个训练单元，未挑选最优测试种子，没有新增训练或依赖。','',
      '同一条已有六通道音频裁为1/3秒；若不足则只为容量探测补零，原数据不改。每配置预热5次，CUDA同步计时20次。计时包括GPU上的前向与原始掩码解码，不含文件IO、传输或模型加载。FP32、batch1、LoRA不合并。','',
      '|系统|音频秒|均值ms|P50 ms|P95 ms|计算RTF|峰值分配GiB|峰值缓存GiB|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in raw:lines.append(f'|{r["system"]}|{r["seconds"]}|{r["mean_ms"]:.2f}|{r["p50_ms"]:.2f}|{r["p95_ms"]:.2f}|{r["compute_rtf"]:.4f}|{r["peak_allocated_bytes"]/2**30:.3f}|{r["peak_reserved_bytes"]/2**30:.3f}|')
    lines+=['','RTF小于1只表示本机计算吞吐快于音频时长。离线双向McNet需要整段上下文，不能由此声称流式低延迟。20次P95是本机短测，后台负载与GPU功耗可能影响；Torch分配/缓存不等于整卡占用。',
      '', '加载时间另存原始记录；首个加载包含部分库导入和GPU初始化，后续共享进程缓存，不能直接比较为独立冷启动速度。模型架构相同，参数数目不能代替实测耗时。',
      '', '原始逐次计时、输入/权重/注册清单哈希见`artifacts/g6_local_runtime_v1/benchmark/raw_timings.json`。本阶段无语音质量或创新结论。下一阶段在本报告归档后开发本地推理命令，并与G5保存音频核对。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8');archive=OUT/'stages/benchmark';archive.mkdir(parents=True);entries=[]
    for i,p in enumerate([report,folder/'raw_timings.json',ROOT/'configs/g6_runtime_plan.json',ROOT/'src/mcse/runtime.py',Path(__file__)]):
        dst=archive/f'{i:02d}_{p.name}';shutil.copy2(p,dst);entries.append({'source':str(p.relative_to(ROOT)),'snapshot':dst.name,'sha256':sha256(p)})
    dump(archive/'archive_manifest.json',{'saved_utc':datetime.now(timezone.utc).isoformat(),'files':entries});dump(OUT/'BENCHMARK_COMPLETE.json',{'report':str(report.relative_to(ROOT)),'configurations':len(raw),'calls':len(raw)*20})
if __name__=='__main__':main()
