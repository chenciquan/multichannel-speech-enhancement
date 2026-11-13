#!/usr/bin/env python3
"""Bounded official weight acquisition and capacity probe; no dependency changes."""
import os
os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,subprocess,sys,time
from urllib.parse import urlparse
import numpy as np
import requests
import soundfile as sf
from g8_common import ROOT,OUT,ASSET,FILENAME,now,sha,read,dump,status,archive,model_load
from mcse.ctc import greedy_ctc_tokens


def main():
    assert json.loads((ROOT/'artifacts/g7_unseen_noise_v1/completion_audit.json').read_text())['status']=='ok'
    assert (ROOT/'artifacts/g7_unseen_noise_v1/stages/completion/archive_manifest.json').is_file()
    plan=json.loads((OUT/'frozen_plan.json').read_text());ASSET.mkdir(parents=True,exist_ok=False)
    target=ASSET/FILENAME;start=time.monotonic();received=0;url=plan['weight_url']
    with requests.get(url,stream=True,timeout=(20,45),headers={'Accept-Encoding':'identity'}) as response:
        response.raise_for_status()
        assert urlparse(response.url).scheme=='https' and urlparse(response.url).hostname=='download.pytorch.org'
        length=int(response.headers['Content-Length']);assert 0<length<=plan['download_cap_bytes']
        h=hashlib.sha256()
        with target.open('xb') as f:
            for chunk in response.iter_content(1048576):
                if time.monotonic()-start>plan['download_timeout_seconds']:raise TimeoutError('Weight download deadline')
                received+=len(chunk)
                if received>length or received>plan['download_cap_bytes']:raise ValueError('Weight byte cap')
                f.write(chunk);h.update(chunk)
                status(stage='asset_download',received_bytes=received,expected_bytes=length)
        if received!=length:raise ValueError('Incomplete model download; partial bytes preserved, not loadable')
        meta={'started_utc':plan['frozen_utc'],'download_completed_utc':now(),'official_url':url,'final_url':response.url,
          'bytes':received,'content_length':length,'ETag':response.headers.get('ETag'),'local_sha256':h.hexdigest(),
          'publisher_SHA256_verified':False,'license':'MIT per official bundle documentation','download_seconds':time.monotonic()-start,
          'integrity_notice':'Official HTTPS transport and local pin; no independently published full publisher SHA256 verified.',
          'dependency_installations':False,'reference_text_used_as_prompt':False}
    import torch,torchaudio
    torch.set_num_threads(2);torch.manual_seed(20261112)
    from torchaudio.pipelines._wav2vec2 import impl,utils
    meta.update(torch_version=torch.__version__,torchaudio_version=torchaudio.__version__,installed_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'requirements-adaptation.lock.txt',__import__('pathlib').Path(impl.__file__),__import__('pathlib').Path(utils.__file__))})
    dump(OUT/'asset_provenance.json',meta,True);status(stage='capacity_probe')
    model,labels=model_load();assert len(labels)==29
    records=read(ROOT/'artifacts/g5_rotation_v1/evaluation/signals.jsonl');reference=min(r['reference_path'] for r in records if r['condition']!='clean')
    a,sr=sf.read(reference,dtype='float32');assert sr==16000 and a.ndim==1 and np.isfinite(a).all()
    probes=[]
    with torch.inference_mode():
        model(torch.from_numpy(a[:16000])[None].cuda());torch.cuda.synchronize()
        for name,audio in [('one_second',a[:16000]),('three_seconds',a[:48000]),('fixed_actual_reference',a)]:
            x=torch.from_numpy(np.ascontiguousarray(audio))[None].cuda();torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
            emission,_=model(x);torch.cuda.synchronize();elapsed=time.perf_counter()-began
            assert emission.ndim==3 and emission.shape[0]==1 and emission.shape[-1]==29 and torch.isfinite(emission).all()
            text=greedy_ctc_tokens(emission[0].argmax(-1).cpu().tolist(),labels)
            probes.append({'name':name,'samples':len(audio),'seconds':elapsed,'emission_shape':list(emission.shape),
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'hypothesis':text})
    probe={'status':'ok','completed_utc':now(),'device':torch.cuda.get_device_name(),'model_parameters':sum(p.numel() for p in model.parameters()),
      'labels':labels,'reference_path':reference,'reference_sha256':sha(reference),'normalize_waveform':False,'no_resampling':True,
      'no_external_LM_or_prompt':True,'FP32_batch1':True,'probes':probes,'notice':'Capacity and finite-output probe only; no recognizer selection by measured WER.'}
    dump(OUT/'capacity_probe.json',probe,True)
    with (OUT/'ctc_unit_tests.log').open('x') as log:
        env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
        for pattern in ('test_ctc.py','test_g8_analysis.py'):
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    report=ROOT/'docs/G8A_CTC识别器来源与资源验证.md'
    report.write_text('\n'.join(['# G8A 独立CTC识别器来源与资源验证','','G7报告和审计已归档后激活。现有torchaudio加载官方WAV2VEC2_ASR_BASE_960H，没有安装依赖或更改增强权重。',
       '', '[官方2.5版文档](https://docs.pytorch.org/audio/2.5.0/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html)说明：base模型、LibriSpeech训练子集960小时预训练及ASR微调，MIT许可。安装的2.5.1实现路径、标签、16k和不归一化设置已核对并保存源码哈希。文档说明不等于对原训练数据的独立审计。',
       '', '论文：[wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://proceedings.neurips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)，NeurIPS2020。本文实验使用base贪心CTC，不引用论文大模型或其他解码配置的成绩为本项目结果。',
       '', f'官方HTTPS权重实收{received:,}字节，低于450MiB上限；本地SHA256 `{meta["local_sha256"]}`。记录Content-Length/ETag；没有核实独立公布的发布者完整SHA，因此本地哈希只用于后续一致性。仅在完整下载并登记哈希后通过weights_only=True加载。',
       '', f'模型参数{probe["model_parameters"]:,}，设备{probe["device"]}。batch1 FP32，无提示、外部语言模型、重采样或峰值归一化。CTC仍具有学习得到的语言规律，不能称为纯声学真值。',
       '', '|探测输入|样本数|计算秒|峰值分配GiB|假设文本|','|---|---:|---:|---:|---|']+
       [f'|{r["name"]}|{r["samples"]}|{r["seconds"]:.4f}|{r["peak_allocated_bytes"]/1024**3:.3f}|{r["hypothesis"]}|' for r in probes]+
       ['', '四项解码测试和两项配对计分测试通过：先折叠相邻重复、再去除blank，以保留被blank隔开的重复字母；不同参考长度下仍取微平均，缺配对直接拒绝。资源探测不是性能测试，未按识别结果挑选识别器。',
        '', '下一阶段对已有G5/G7保存音频共12,600对重新计WER/S/D/I，使用原英语归一化，不重新训练增强器、不重新选择权重。分别讨论同一识别器下的方法差值；两识别器结论一致也不能替代人工声学验证。',
        '', '证据：`artifacts/g8_cross_asr_v1/asset_provenance.json`、`capacity_probe.json`、`ctc_unit_tests.log`。','', 'UTC：'+now()])+'\n',encoding='utf-8')
    archive('asset',report,[OUT/'frozen_plan.json',OUT/'asset_provenance.json',OUT/'capacity_probe.json',OUT/'ctc_unit_tests.log',ROOT/'src/mcse/ctc.py',ROOT/'tests/test_ctc.py',ROOT/'tests/test_g8_analysis.py']+[ROOT/'scripts'/name for name in ('g8_common.py','g8_evaluate.py','g8_finish.py','g8_queue.py')]+[__import__('pathlib').Path(__file__)])
    dump(OUT/'ASSET_STAGE_COMPLETE.json',{'completed_utc':now(),'status':'ok','report':str(report.relative_to(ROOT))},True)


if __name__=='__main__':main()
