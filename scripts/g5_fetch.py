#!/usr/bin/env python3
"""Bounded fresh, speaker-balanced confirmation sources. No old-test fallback."""
import hashlib,io,json,re,runpy,tarfile,time,traceback
from datetime import datetime
from pathlib import Path,PurePosixPath
import requests
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/g5_v1/sources';PLAN=json.loads((ROOT/'configs/g5_rotation_plan.json').read_text())
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    assert (ROOT/'artifacts/g4_counterfactual_v1/MAJOR_STAGE_COMPLETE.json').exists();OUT.mkdir(parents=True,exist_ok=False);(OUT/'speech').mkdir()
    old=[]
    for relative in ('data/sources/g1_v1/source_manifest.jsonl','data/sources/g2_confirmation_v1/source_manifest.jsonl','data/g3_v1/sources/speech_manifest.jsonl'):
        old += [json.loads(s) for s in (ROOT/relative).read_text().splitlines()]
    excluded_speakers={r['speaker_id'] for r in old};excluded_hashes={r['sha256'] for r in old};helper=runpy.run_path(str(ROOT/'scripts/fetch_diagnostic_sources.py'),run_name='g5_fetch_helper')
    deadline=datetime.fromisoformat(PLAN['data_cutoff_utc'].replace('Z','+00:00')).timestamp();candidates={};transcripts={};ready=[];start=time.time()
    result={'status':'running','excluded_speakers':sorted(excluded_speakers),'plan':'configs/g5_rotation_plan.json'};dump(OUT/'status.json',result)
    try:
        with requests.Session() as session,session.get(PLAN['source_url'],stream=True,timeout=(30,90)) as response:
            response.raise_for_status();stream=helper['BoundedStream'](response.raw,PLAN['compressed_stream_cap_bytes'])
            with tarfile.open(fileobj=stream,mode='r|gz') as archive:
                for member in archive:
                    if time.time()>deadline:raise TimeoutError('Fresh-data time cap reached')
                    if not member.isfile():continue
                    leaf=PurePosixPath(member.name).name
                    if leaf.endswith('.trans.txt'):
                        for line in archive.extractfile(member).read().decode().splitlines():
                            sid,text=line.split(' ',1);transcripts[sid]=text
                    elif re.fullmatch(r'\d+-\d+-\d+\.flac',leaf):
                        sid=leaf[:-5];speaker=sid.split('-')[0]
                        if speaker in excluded_speakers or len(candidates.get(speaker,[]))>=4 or member.size>8*1024**2:continue
                        data=archive.extractfile(member).read();info=sf.info(io.BytesIO(data));h=hashlib.sha256(data).hexdigest()
                        if 1<=info.duration<=3.8 and info.channels==1 and info.samplerate==16000 and h not in excluded_hashes:
                            candidates.setdefault(speaker,[]).append((sid,data,info,h))
                    for speaker,items in candidates.items():
                        if speaker not in ready and len(items)==4 and all(r[0] in transcripts for r in items):ready.append(speaker)
                    if len(ready)>=3:break
            if len(ready)<3:raise RuntimeError('Fewer than three eligible balanced fresh speakers within cap')
            selected=[]
            for speaker in ready[:3]:
                for sid,data,info,h in candidates[speaker]:
                    path=OUT/'speech'/(sid+'.flac');path.write_bytes(data);selected.append({'source_id':sid,'speaker_id':speaker,'split':'confirmation',
                      'path':'speech/'+path.name,'samples':info.frames,'duration_seconds':info.duration,'sha256':h,'license':'CC-BY-4.0','source_url':PLAN['source_url'],'transcript':transcripts[sid]})
            with (OUT/'speech_manifest.jsonl').open('x') as f:
                for r in selected:f.write(json.dumps(r)+'\n')
            result.update(status='ok',selected_speakers=ready[:3],sources=len(selected),compressed_bytes_streamed=stream.count,full_archive_saved=False,elapsed_seconds=time.time()-start)
    except Exception as e:result.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:dump(OUT/'status.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
