#!/usr/bin/env python3
"""Bounded small fresh-speaker subset, fixed selection before model output."""
import hashlib,io,json,re,runpy,tarfile,time
from pathlib import Path,PurePosixPath
import requests
import soundfile as sf
from g10_common import ROOT,OUT,BASE,now,sha,read,dump,status,archive


def main():
    plan=json.loads((OUT/'frozen_plan.json').read_text());folder=BASE/'sources';folder.mkdir(parents=True,exist_ok=False);(folder/'speech').mkdir();old=[]
    manifests=['data/sources/g1_v1/source_manifest.jsonl','data/sources/g2_confirmation_v1/source_manifest.jsonl','data/g3_v1/sources/speech_manifest.jsonl','data/g5_v1/sources/speech_manifest.jsonl']
    for relative in manifests:old+=read(ROOT/relative)
    excluded={k:{r[k] for r in old} for k in ('speaker_id','source_id','sha256')};helper=runpy.run_path(str(ROOT/'scripts/fetch_diagnostic_sources.py'),run_name='g10_bounded_stream')
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
                        if speaker in excluded['speaker_id'] or sid in excluded['source_id'] or len(candidates.get(speaker,[]))>=4 or member.size>8*1024**2:continue
                        data=tar.extractfile(member).read();info=sf.info(io.BytesIO(data));h=hashlib.sha256(data).hexdigest()
                        if 1<=info.duration<=3.8 and info.channels==1 and info.samplerate==16000 and h not in excluded['sha256']:
                            candidates.setdefault(speaker,[]).append((sid,data,info,h))
                    for speaker,items in candidates.items():
                        if speaker not in ready and len(items)==4 and all(r[0] in transcripts for r in items):ready.append(speaker)
                    if len(ready)>=4:break
            if len(ready)<4:raise RuntimeError('Not enough fresh balanced speakers within bound; no fallback')
            selected=[]
            for speaker in ready[:4]:
                for sid,data,info,h in candidates[speaker]:
                    p=folder/'speech'/(sid+'.flac')
                    with p.open('xb') as f:f.write(data)
                    selected.append({'source_id':sid,'speaker_id':speaker,'split':'fresh_voice_confirmation','path':'speech/'+p.name,'samples':info.frames,
                      'duration_seconds':info.duration,'sha256':h,'license':plan['source_license'],'source_url':plan['source_url'],'transcript':transcripts[sid]})
            assert len(selected)==16 and len({r['speaker_id'] for r in selected})==4
            with (folder/'speech_manifest.jsonl').open('x') as f:
                for r in selected:f.write(json.dumps(r)+'\n')
            selection.update(status='ok',completed_utc=now(),compressed_bytes_streamed=stream.count,selected_speakers=ready[:4],sources=16,
              full_archive_saved=False,full_archive_MD5_verified=False,elapsed_seconds=time.monotonic()-start,integrity='Official HTTPS stream,FLAC readable,per-file SHA; no full-archive checksum claim')
    except Exception as e:selection.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:dump(folder/'status.json',selection)
    report=ROOT/'docs/G10A_新说话人语料选择记录.md';report.write_text('# G10A 新说话人语料选择记录\n\nG9最终验收保存后登记协议，按官方归档流顺序选择4位新说话人、每人4条完整1—3.8秒16k单通道语音及完整文本，共16句。全部G1/G2/G3/G5说话人、语句ID和源文件SHA排除；没有根据识别或增强效果挑句。\n\n实际说话人：'+', '.join(ready[:4])+'。实际有界读取压缩字节：'+str(selection['compressed_bytes_streamed'])+'，上限224MiB；仅保存选中FLAC和文本元数据，不保存整个337M归档，也没有核实整包MD5。本地逐文件SHA/FLAC可读性验证，完整排除清单留档。\n\n来源：[OpenSLR12官方记录](https://www.openslr.org/12)，CC-BY4.0。官方为16k英语朗读语料；这不是中文真实场景或实测多通道语音。新说话人仅相对本项目增强实验与训练，不保证Whisper训练语料从未包含这些公开内容；CTC训练来源沿用官方960h训练子集说明，不作独立审计。\n\n本阶段尚未运行这批源句的增强或识别。下一步复用既有四类噪声，生成固定0dB、三方向/三空间条件及干净对照，再按冻结15系统评估。\n\nUTC：'+now()+'\n',encoding='utf-8')
    archive('sources',report,[OUT/'frozen_plan.json',folder/'status.json',folder/'speech_manifest.jsonl',ROOT/'configs/g10_fresh_speaker_plan.json',Path(__file__)])
    print(json.dumps({'stage':'sources','status':'ok','bytes':selection['compressed_bytes_streamed']}),flush=True)


if __name__=='__main__':main()
