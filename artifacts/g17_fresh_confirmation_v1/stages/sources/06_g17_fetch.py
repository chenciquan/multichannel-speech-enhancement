#!/usr/bin/env python3
"""Bounded small fresh-speaker subset, fixed selection before model output."""
import hashlib,io,json,re,runpy,tarfile,time
from pathlib import Path,PurePosixPath
import requests
import soundfile as sf
import numpy as np
from g17_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive


def main():
    plan=json.loads((OUT/'frozen_plan.json').read_text());folder=BASE/'sources';folder.mkdir(parents=True,exist_ok=False);(folder/'speech').mkdir();old=[]
    manifests=['data/sources/g1_v1/source_manifest.jsonl','data/sources/g2_confirmation_v1/source_manifest.jsonl','data/g3_v1/sources/speech_manifest.jsonl','data/g5_v1/sources/speech_manifest.jsonl','data/g10_v1/sources/speech_manifest.jsonl']
    for relative in manifests:old+=read(ROOT/relative)
    excluded={k:{r[k] for r in old} for k in ('speaker_id','source_id','sha256')};helper=runpy.run_path(str(ROOT/'scripts/fetch_diagnostic_sources.py'),run_name='g17_bounded_stream')
    start=time.monotonic();candidates={};transcripts={};ready=[];selection={'status':'running','started_utc':now(),'excluded':{k:sorted(v) for k,v in excluded.items()},'prior_manifests_sha256':{p:sha(ROOT/p) for p in manifests}}
    dump(folder/'status.json',selection,True);status(stage='fresh_source_fetch')
    try:
        with requests.get(plan['source_url'],stream=True,timeout=(20,60)) as response:
            response.raise_for_status();stream=helper['BoundedStream'](response.raw,plan['compressed_stream_cap_bytes'])
            with tarfile.open(fileobj=stream,mode='r|gz') as tar:
                for member in tar:
                    if time.monotonic()-start>plan['fetch_timeout_seconds']:raise TimeoutError('Fresh-source time cap')
                    if not member.isfile():continue
                    leaf=PurePosixPath(member.name).name
                    if leaf.endswith('.trans.txt'):
                        if member.size>1048576:raise ValueError('Transcript member cap')
                        for line in tar.extractfile(member).read().decode('utf-8').splitlines():
                            sid,text=line.split(' ',1);transcripts[sid]=text
                    elif re.fullmatch(r'\d+-\d+-\d+\.flac',leaf):
                        sid=leaf[:-5];speaker=sid.split('-')[0]
                        if speaker in excluded['speaker_id'] or sid in excluded['source_id'] or len(candidates.get(speaker,[]))>=2 or member.size>8*1024**2:continue
                        data=tar.extractfile(member).read();info=sf.info(io.BytesIO(data));h=hashlib.sha256(data).hexdigest()
                        if 1<=info.duration<=3.51 and info.channels==1 and info.samplerate==16000 and h not in excluded['sha256']:
                            candidates.setdefault(speaker,[]).append((sid,data,info,h))
                    for speaker,items in candidates.items():
                        if speaker not in ready and len(items)==2 and all(r[0] in transcripts for r in items):ready.append(speaker)
                    if len(ready)>=4:break
            if len(ready)<4:raise RuntimeError('Not enough fresh balanced speakers within bound; no fallback')
            selected=[]
            for speaker in ready[:4]:
                for sid,data,info,h in candidates[speaker]:
                    p=folder/'speech'/(sid+'.flac')
                    with p.open('xb') as f:f.write(data)
                    wave,rate=sf.read(io.BytesIO(data),dtype='float32');assert rate==16000 and wave.ndim==1 and len(wave)==info.frames and np.isfinite(wave).all()
                    selected.append({'source_id':sid,'speaker_id':speaker,'split':'fresh_voice_confirmation','path':'speech/'+p.name,'samples':info.frames,
                      'duration_seconds':info.duration,'sha256':h,'license':plan['source_license'],'source_url':plan['source_url'],'transcript':transcripts[sid]})
            assert len(selected)==8 and len({r['speaker_id'] for r in selected})==4
            with (folder/'speech_manifest.jsonl').open('x') as f:
                for r in selected:f.write(json.dumps(r)+'\n')
            selection.update(status='ok',completed_utc=now(),compressed_bytes_streamed=stream.count,selected_speakers=ready[:4],sources=8,
              full_archive_saved=False,full_archive_MD5_verified=False,elapsed_seconds=time.monotonic()-start,integrity='Official HTTPS stream,FLAC readable,per-file SHA; no full-archive checksum claim')
    except Exception as e:selection.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:dump(folder/'status.json',selection)
    report=ROOT/'docs/G17A_新确认语音选择与排除记录.md';report.write_text('# G17A 新确认语音选择与排除记录\n\nG16开发负结果、全部21方法与新确认协议先冻结。官方test-clean归档顺序，前4位符合排除规则的新说话人，各取前2句完整1—3.51秒16k单声道及文本，共8句。没有裁剪词尾或按增强/识别成绩选句。所有G1/G2/G3/G5/G10的说话人、语句ID和源SHA排除；本项目后续阶段此前均复用这些源。\n\n实际说话人：'+', '.join(ready[:4])+'。实际有界流读取'+str(selection['compressed_bytes_streamed'])+'字节，上限192MiB；只保存8个FLAC和文本元数据，不保存整个346,663,984字节归档，不称整包MD5核验。所有选中FLAC完整解码、16k/单声道/帧数/有限值及本地SHA核验。完整排除表和原清单SHA留存。\n\n来源：[OpenSLR12官方记录](https://www.openslr.org/12)，CC-BY4.0；HEAD端点记录在取数前保存。新说话人仅相对本项目增强训练/调参/旧实验，不保证Whisper基础训练从未见过公开内容。8句/4人不是数千个独立样本。\n\n本阶段没有对新语音运行增强或ASR。下一步只复用OOFFICE/NPARK旧噪声，正确刚性几何、0dB、三方向/三空间场及干净控制，冻结21系统全部三种子。新语音确认不代表真实六麦或新噪声录音泛化。\n\nUTC：'+now()+'\n',encoding='utf-8')
    archive('sources',report,[OUT/'frozen_plan.json',folder/'status.json',folder/'speech_manifest.jsonl',ROOT/'configs/g17_fresh_confirmation_plan.json',ROOT/'artifacts/g17_source_endpoint_v1.json',Path(__file__)])
    print(json.dumps({'stage':'sources','status':'ok','bytes':selection['compressed_bytes_streamed']}),flush=True)


if __name__=='__main__':main()
