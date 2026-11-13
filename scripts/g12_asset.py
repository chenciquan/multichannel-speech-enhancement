#!/usr/bin/env python3
"""One pinned official wheel in existing venv, physics and compatibility QA."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import gc,json,subprocess,sys,time
from importlib.metadata import distributions,version
from pathlib import Path
import numpy as np
import requests
import soundfile as sf
from g12_common import ROOT,OUT,now,sha,read,dump,status,archive


def installed():return {d.metadata['Name'].lower().replace('_','-'):d.version for d in distributions()}


def main():
    plan=json.loads((OUT/'frozen_plan.json').read_text());assets=OUT/'assets';assets.mkdir(exist_ok=False);status(stage='asset')
    before=installed();dump(assets/'distributions_before.json',before,True);uv=ROOT/'.tools/uv-x86_64-unknown-linux-gnu/uv';assert uv.is_file()
    check_command=[str(uv),'pip','check','--python',sys.executable,'--offline','--no-python-downloads']
    freeze_command=[str(uv),'pip','freeze','--python',sys.executable,'--offline','--no-python-downloads']
    with (assets/'dependency_check_before.log').open('x') as f:check_before=subprocess.run(check_command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=60)
    if check_before.returncode:raise RuntimeError('Existing dependency check failed before installation')
    with (assets/'freeze_before.txt').open('x') as f:subprocess.run(freeze_command,stdout=f,check=True,timeout=60)
    metadata=requests.get(plan['package_metadata'],timeout=(15,30));metadata.raise_for_status();meta=metadata.json()
    chosen=next(f for f in meta['urls'] if f['filename']==plan['wheel_filename'])
    assert chosen['digests']['sha256']==plan['wheel_sha256'] and chosen['size']==plan['wheel_bytes']
    dump(assets/'publisher_metadata.json',{'source':plan['package_metadata'],'version':meta['info']['version'],'requires_python':meta['info']['requires_python'],
      'requires_dist':meta['info']['requires_dist'],'license':meta['info']['license'],'selected_wheel':chosen,'retrieved_utc':now()},True)
    wheel=assets/plan['wheel_filename'];received=0
    with requests.get(plan['wheel_url'],stream=True,timeout=(15,30)) as response:
        response.raise_for_status()
        with wheel.open('xb') as f:
            for chunk in response.iter_content(65536):
                received+=len(chunk)
                if received>plan['download_cap_bytes']:raise ValueError('Wheel byte cap')
                f.write(chunk)
    assert received==plan['wheel_bytes'] and sha(wheel)==plan['wheel_sha256']
    if 'rir-generator' in before and before['rir-generator']!='0.3.0':raise ValueError('Existing different RIR version; refuse overwrite')
    with (assets/'install.log').open('x') as f:
        if 'rir-generator' not in before:subprocess.run([str(uv),'pip','install','--python',sys.executable,'--offline','--no-index','--no-deps','--no-build','--no-python-downloads','--link-mode','copy',str(wheel)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=180,check=True)
        else:f.write('Existing matching version retained,no installation performed.\n')
    after=installed();assert all(after.get(k)==v for k,v in before.items()) and set(after)-set(before)<={'rir-generator'} and after['rir-generator']=='0.3.0'
    dump(assets/'distributions_after.json',after,True)
    with (assets/'dependency_check_after.log').open('x') as f:subprocess.run(check_command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=60,check=True)
    with (ROOT/'requirements-room.lock.txt').open('x') as f:subprocess.run(freeze_command,stdout=f,check=True,timeout=60)
    import rir_generator as rir
    assert version('rir-generator')=='0.3.0'
    geometry=json.loads((ROOT/'configs/g1_geometry.json').read_text());positions=np.asarray(geometry['microphone_positions_m']);ref=np.array(plan['reference_mic4_position_m']);receivers=ref+(positions-positions[4]);room=np.array(plan['room_dimensions_m'])
    from mcse.spatial_noise import direction
    source=ref+plan['source_distance_m']*direction(20,10);assert ((receivers>0)&(receivers<room)).all() and ((source>0)&(source<room)).all()
    probes=[];kernels={};gain=4*np.pi*plan['source_distance_m']
    for rt in plan['requested_RT60_seconds']:
        kwargs={'beta':np.zeros(6)} if rt==0 else {'reverberation_time':rt}
        h=rir.generate(c=343,fs=16000,r=receivers,s=source,L=room,nsample=plan['RIR_samples'],mtype=rir.mtype.omnidirectional,order=-1,dim=3,hp_filter=False,**kwargs)*gain
        assert h.shape==(7680,6) and np.isfinite(h).all() and np.any(h)
        repeated=rir.generate(c=343,fs=16000,r=receivers,s=source,L=room,nsample=plan['RIR_samples'],mtype=rir.mtype.omnidirectional,order=-1,dim=3,hp_filter=False,**kwargs)*gain
        assert np.array_equal(h,repeated);kernels['rt_'+str(rt)]=h
        direct_samples=np.linalg.norm(receivers-source,axis=1)/343*16000;peaks=np.argmax(np.abs(h),axis=0);tail=float(np.sum(h[320:,4]**2)/np.sum(h[:,4]**2));measured=None
        if rt==0:assert np.max(np.abs(peaks-direct_samples))<=2 and tail<1e-8
        else:
            assert tail>1e-8;edc=np.cumsum(h[::-1,4]**2)[::-1];db=10*np.log10(np.maximum(edc/edc[0],1e-30));selected=(db<=-5)&(db>=-25)
            assert selected.sum()>20;slope,_=np.polyfit(np.arange(len(h))[selected]/16000,db[selected],1);assert slope<0;measured=float(-60/slope)
        probes.append({'requested_RT60':rt,'Schroeder_T20_extrapolated_RT60':measured,'tail_energy_fraction_after20ms':tail,'direct_expected_samples':direct_samples.tolist(),'largest_peak_samples':peaks.tolist(),'deterministic_repeat_exact':True})
    assert probes[2]['Schroeder_T20_extrapolated_RT60']>probes[1]['Schroeder_T20_extrapolated_RT60']>0
    np.savez_compressed(assets/'RIR_smoke.npz',**kernels);dump(assets/'RIR_smoke.json',probes,True)
    source_rows=read(ROOT/'data/g10_v1/sources/speech_manifest.jsonl');speakers=list(dict.fromkeys(r['speaker_id'] for r in source_rows));selected=[]
    for sp in speakers:
        eligible=[r for r in source_rows if r['speaker_id']==sp and r['duration_seconds']<=3.51];assert len(eligible)>=2;selected+=eligible[:2]
    assert len(selected)==8;dump(OUT/'selected_sources.json',selected,True);longest=max(r['samples'] for r in selected)+7680-1;assert longest==63759 and longest<64000
    import torch
    from mcse.runtime import load_system,enhance_array
    torch.set_num_threads(2);torch.manual_seed(20261119);rng=np.random.default_rng(20261119);synthetic=(rng.standard_normal((longest,6))*.03).astype(np.float32)
    case=read(ROOT/'data/g10_v1/manifest.jsonl')[0];x,sr=sf.read(case['noisy_path'],dtype='float32',always_2d=True);assert sr==16000
    old=read(ROOT/'artifacts/g10_fresh_voice_v1/evaluation/combined_signals.jsonl');capacity=[]
    for system in ('mcnet','g5/rotated_without_clean_identity_s20261112'):
        runtime=load_system(system,'cuda');name=system.split('/')[-1];prior=next(r for r in old if r['system']==name and r['sample_id']==case['sample_id']);assert sha(prior['output_path'])==prior['output_sha256']
        torch.cuda.reset_peak_memory_stats();y,_=enhance_array(runtime,x);expected=sf.read(prior['output_path'],dtype='float32')[0];error=float(np.max(np.abs(y-expected)));assert error<=plan['asset_GPU_parity_tolerance']
        result,seconds=enhance_array(runtime,synthetic);assert result.shape==(longest,) and np.isfinite(result).all()
        capacity.append({'system':system,'existing_G10_max_abs_difference':error,'frames':longest,'capacity_seconds':seconds,
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'metadata':runtime.metadata})
        del runtime;gc.collect();torch.cuda.empty_cache()
    dump(assets/'capacity_and_parity.json',capacity,True)
    with (OUT/'full86_unit_tests.log').open('x') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
      env={**os.environ,'PYTHONPATH':str(ROOT/'src')},stdout=f,stderr=subprocess.STDOUT,timeout=300,check=True)
    assert 'Ran 86 tests' in (OUT/'full86_unit_tests.log').read_text() and '\nOK' in (OUT/'full86_unit_tests.log').read_text()
    installed_modules={str(p):sha(p) for p in Path(rir.__file__).parent.rglob('*') if p.is_file() and p.suffix in ('.py','.so')}
    completion={'status':'ok','completed_utc':now(),'wheel_bytes':received,'wheel_sha256':sha(wheel),'added_distributions':sorted(set(after)-set(before)),
      'all_previous_distribution_versions_unchanged':True,'uv_sha256':sha(uv),'dependency_check_before_returncode':check_before.returncode,'dependency_check_after_ok':True,'unit_tests':86,'installed_module_files_sha256':installed_modules}
    dump(OUT/'ASSET_COMPLETE.json',completion,True)
    lines=['# G12A 房间响应生成器来源、兼容性与资源验证','','G11验收保存后登记本阶段。仅在原venv加入rir-generator0.3.0；官方Python3.12/Linux预编译wheel实收36,773字节，PyPI公布SHA256与本地核对一致。使用项目原有uv、明确原Python，离线--no-index/--no-deps/--no-build及禁止Python下载，未编译、未补装pip或构建工具，全部旧发行包版本保持不变，uv pip check通过。新环境快照另存requirements-room.lock.txt，旧锁文件不覆盖。首版pip调用在安装前失败，完整记录另存v1及失败报告，本次v2不改声学/模型/指标条件。',
       '', '来源：[PyPI0.3.0](https://pypi.org/project/rir-generator/0.3.0/)、[维护者仓库](https://github.com/audiolabs/rir-generator)、[MIT许可](https://github.com/audiolabs/rir-generator/blob/master/LICENSE)、[官方接口](https://rir-generator.readthedocs.io/en/latest/module.html)。这是Habets生成器的官方Python端口，用现有NumPy/SciPy/CFFI，不是新编写房间声学算法。',
       '', '六麦单房间6×5×3m、源距1.5m，16k/7680点RIR、全反射阶、全向、禁用内部高通。无混响用零反射系数；两正请求参数用官方T60接口。直接传播最大峰位置误差≤2样本，无混响20ms后能量近零；正混响有尾部且Schroeder拟合衰减顺序正确，同参重复逐样本一致。请求的Sabine参数不直接当实测RT60。','','|请求RT60秒|Schroeder T20外推秒|20ms后能量比例|','|---:|---:|---:|']
    for r in probes:lines.append(f'|{r["requested_RT60"]}|{r["Schroeder_T20_extrapolated_RT60"]}|{r["tail_energy_fraction_after20ms"]:.6f}|')
    lines+=['','按完整语句长度≤3.51秒，各说话人前两条，共8句/4人；加全部0.48秒响应尾部，最长63,759样本(<4秒)，未截断词或尾部。选句依长度与原清单顺序，不按效果。','','|模型|旧G10最大波形差|最长容量探测秒|峰值Torch分配GiB|峰值保留GiB|','|---|---:|---:|---:|---:|']
    for r in capacity:lines.append(f'|{r["system"]}|{r["existing_G10_max_abs_difference"]}|{r["capacity_seconds"]:.4f}|{r["peak_allocated_bytes"]/1024**3:.3f}|{r["peak_reserved_bytes"]/1024**3:.3f}|')
    lines+=['','容量探测使用合成随机输入，不是质量实验或模型选择。两模型的旧G10实际波形一致性在预定1e-5内。86项原有完整测试通过。该探测不替代整个变长批次的峰值，后续每系统会记录峰值并释放缓存。',
       '', '本阶段只完成依赖/物理基本检查和容量验证，没有生成房间评估集或运行其成绩。下一步数据目标是参考麦克风的混响语音，用于降噪，不混称去混响；加入无反射对照、保持全部语句与尾部、连续噪声预卷积。仅单间模拟房，不能代替真实多通道录音。','', 'UTC：'+now()]
    report=ROOT/'docs/G12A_房间响应生成器与资源验证.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('asset',report,[OUT/'frozen_plan.json',OUT/'ASSET_COMPLETE.json',OUT/'selected_sources.json',assets/'publisher_metadata.json',assets/'distributions_before.json',assets/'distributions_after.json',assets/'RIR_smoke.json',assets/'capacity_and_parity.json',OUT/'full86_unit_tests.log',ROOT/'requirements-room.lock.txt',Path(__file__)])


if __name__=='__main__':main()
