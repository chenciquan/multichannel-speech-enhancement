#!/usr/bin/env python3
"""CPU-only full fixed-filter component diagnostics, no model training/ASR."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time
from pathlib import Path
import torch
from g11_common import ROOT,OUT,now,sha,dump,status,archive,Evaluator,selected_rows


def main():
    assert (OUT/'stages/preflight/archive_manifest.json').exists();plan=json.loads((OUT/'frozen_plan.json').read_text())
    for name,h in plan['source_evidence_sha256'].items():assert sha(ROOT/name)==h
    for name,h in json.loads((OUT/'ANALYZER_READY.json').read_text())['files'].items():assert sha(ROOT/name)==h
    assert not torch.cuda.is_initialized();start=time.monotonic();rows=selected_rows();evaluator=Evaluator();folder=OUT/'analysis';folder.mkdir(exist_ok=False)
    from importlib.metadata import version
    dump(folder/'provenance.json',{'started_utc':now(),'plan_sha256':sha(OUT/'frozen_plan.json'),'preflight_archive_sha256':sha(OUT/'stages/preflight/archive_manifest.json'),
      'CUDA_initialized':torch.cuda.is_initialized(),'CPU_threads':2,'precision':'float64/complex128; wavecomponents float32 like saved output',
      'packages':{p:version(p) for p in ('torch','torchaudio','numpy','scipy','soundfile')},'no_training_or_ASR':True,
      'seeds':[20261112,20261113,20261114],'component_module_sha256':sha(ROOT/'src/mcse/components.py')},True)
    complete=[]
    with (folder/'components.jsonl').open('x') as log:
        for i,r in enumerate(rows):
            if time.monotonic()-start>plan['max_analysis_seconds']-180:raise TimeoutError('Component diagnostic reserve')
            result=evaluator.evaluate(r);complete.append(result);log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush()
            if (i+1)%100==0:status(stage='components',done=i+1,total=4032,system=r['system']);print(json.dumps({'components':i+1}),flush=True)
    assert len(complete)==4032 and not torch.cuda.is_initialized()
    dump(folder/'verified_source_files.json',evaluator.verified,True)
    dump(folder/'COMPONENTS_COMPLETE.json',{'status':'ok','completed_utc':now(),'rows':4032,'seconds':time.monotonic()-start,'CUDA_initialized':torch.cuda.is_initialized(),
      'verified_source_files':len(evaluator.verified),'clean_field_cache':evaluator.clean.cache_info()._asdict()},True)
    report=ROOT/'docs/G11B_完整分量诊断核对记录.md';report.write_text('# G11B 完整分量诊断核对记录\n\nG11A预检报告归档后，完成全部4,032次固定滤波器分解：原始前端576条，固定/旋转前端各三种子1,728条。所有参考精确还原、旧输出复现、线性叠加与SNR能量核算检查通过，真值仅用于诊断。\n\n逐例分量能量、下限标记、目标SI-SDR未定义状态和旧双ASR计分已保存；没有重新运行识别或训练模型。全程CPU，CUDA未初始化。数据/原输出及权重不修改。\n\n该记录与全部计算代码先保存，下一步才作汇总、按说话人探索性bootstrap及机制解释。\n\nUTC：'+now()+'\n',encoding='utf-8')
    archive('components',report,[folder/'provenance.json',folder/'COMPONENTS_COMPLETE.json',OUT/'analysis_unit_tests.log',ROOT/'tests/test_g11_analysis.py',ROOT/'src/mcse/components.py',ROOT/'scripts/g11_run.py',ROOT/'scripts/g11_finish.py'])
    from g11_finish import finish
    finish(complete)


if __name__=='__main__':main()
