#!/usr/bin/env python3
"""Deadline-bounded public source selection with speaker-isolated final test."""
import hashlib,io,json,re,runpy,tarfile,time,zipfile
from datetime import datetime
from pathlib import Path,PurePosixPath
import requests
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/g3_v1/sources'
PLAN=json.loads((ROOT/'configs/g3_overnight_plan.json').read_text())

def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    if OUT.exists():raise RuntimeError('Source directory exists; inspect instead of overwriting')
    OUT.mkdir(parents=True);(OUT/'speech').mkdir();(OUT/'noise').mkdir()
    dump(OUT/'frozen_plan.json',PLAN)
    old=[]
    for relative in ['data/sources/g1_v1/source_manifest.jsonl','data/sources/g2_confirmation_v1/source_manifest.jsonl']:
        old.extend(json.loads(t) for t in (ROOT/relative).read_text().splitlines())
    excluded={r['source_id'] for r in old};old_speakers={r['speaker_id'] for r in old}
    modules=runpy.run_path(str(ROOT/'scripts/fetch_diagnostic_sources.py'),run_name='g3_source_helpers')
    report={'status':'running','start':time.time(),'speech':{},'noise':[]};dump(OUT/'status.json',report)
    roles={};selected=[];counts={};seconds={};transcripts={}
    deadline=datetime.fromisoformat(PLAN['data_cutoff_utc'].replace('Z','+00:00')).timestamp()
    try:
        with requests.Session() as session:
            with session.get(PLAN['source'],stream=True,timeout=(30,90)) as response:
                response.raise_for_status();bounded=modules['BoundedStream'](response.raw,PLAN['max_stream_bytes'])
                with tarfile.open(fileobj=bounded,mode='r|gz') as archive:
                    for member in archive:
                        if time.time()>deadline:raise TimeoutError('Data time cap reached; partial files preserved')
                        if not member.isfile():continue
                        leaf=PurePosixPath(member.name).name
                        if leaf.endswith('.trans.txt'):
                            for line in archive.extractfile(member).read().decode().splitlines():
                                key,value=line.split(' ',1);transcripts[key]=value
                        elif re.fullmatch(r'\d+-\d+-\d+\.flac',leaf):
                            sid=leaf[:-5];speaker=sid.split('-')[0]
                            if speaker not in roles:
                                roles[speaker]='train' if sum(v=='train' for v in roles.values())<4 else 'validation' if sum(v=='validation' for v in roles.values())<2 else 'test'
                            role=roles[speaker]
                            if sid in excluded:continue
                            if role=='train' and seconds.get(speaker,0)>=PLAN['train_seconds_per_speaker']:continue
                            if role=='validation' and counts.get(speaker,0)>=6:continue
                            if role=='test' and (speaker in old_speakers or counts.get(speaker,0)>=10 or sum(r['split']=='test' for r in selected)>=30):continue
                            if member.size>8*1024**2:continue
                            data=archive.extractfile(member).read();info=sf.info(io.BytesIO(data))
                            eligible=(1<=info.duration<=30) if role=='train' else (1<=info.duration<=3.8)
                            if not eligible or info.channels!=1 or info.samplerate!=16000:continue
                            (OUT/'speech'/leaf).write_bytes(data)
                            selected.append({'source_id':sid,'speaker_id':speaker,'split':role,'path':'speech/'+leaf,
                                'samples':info.frames,'duration_seconds':info.duration,'sha256':sha(data),'license':'CC-BY-4.0','source_url':PLAN['source']})
                            counts[speaker]=counts.get(speaker,0)+1;seconds[speaker]=seconds.get(speaker,0)+info.duration
                        train=sum(r['duration_seconds'] for r in selected if r['split']=='train')
                        val=sum(r['split']=='validation' for r in selected);test=sum(r['split']=='test' for r in selected)
                        if train>=1200 and val>=10 and test==30 and len({r['speaker_id'] for r in selected if r['split']=='test'})>=3 and all(r['source_id'] in transcripts for r in selected):break
                report['speech']={'compressed_bytes_streamed':bounded.count,'roles':roles,'train_seconds':train,'validation_sources':val,'test_sources':test,'full_archive_saved':False}
            if train<1200 or val<10 or test<30:raise RuntimeError('Frozen source targets not satisfied; partial source data retained')
            for r in selected:r['transcript']=transcripts[r['source_id']]
            with (OUT/'speech_manifest.jsonl').open('x',encoding='utf-8') as f:
                for r in selected:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            dump(OUT/'speech_complete.json',report['speech'])
            noises=[]
            for name in ['DWASHING','DKITCHEN','OOFFICE','PCAFETER']:
                existing=ROOT/'data/sources/g1_v1/noise'/(name+'_ch01.wav')
                if existing.exists():
                    path=existing;url='https://zenodo.org/records/1227121';received=0
                else:
                    url=f'https://zenodo.org/records/1227121/files/{name}_16k.zip?download=1'
                    remote=modules['RangeReader'](session,url,max_bytes=20*1024**2)
                    with zipfile.ZipFile(remote) as z:
                        members=[m for m in z.namelist() if PurePosixPath(m).name.lower()=='ch01.wav']
                        if len(members)!=1:raise ValueError('Unexpected noise ZIP')
                        payload=z.read(members[0])
                    path=OUT/'noise'/(name+'_ch01.wav');path.write_bytes(payload);received=remote.received
                info=sf.info(path)
                if info.samplerate!=16000 or info.channels!=1:raise ValueError('Noise format mismatch')
                noises.append({'noise_id':name,'path':str(path.resolve()),'sha256':sha(path.read_bytes()),'duration_seconds':info.duration,'license':'CC-BY-SA-3.0','source_url':url})
                report['noise'].append({'name':name,'bytes_downloaded':received});dump(OUT/'status.json',report)
            dump(OUT/'noise_manifest.json',noises);report['status']='ok'
    except Exception as exc:
        report.update(status='error',error_type=type(exc).__name__,error=str(exc));raise
    finally:dump(OUT/'status.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
