#!/usr/bin/env python3
"""Fetch eight preselected-policy new short utterances from a bounded official stream."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import re
import runpy
import tarfile
import requests
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Refusing to overwrite any prior source acquisition')
    planpath=ROOT/'configs/g2_confirmation_plan.json';planbytes=planpath.read_bytes();plan=json.loads(planbytes)
    old=[json.loads(t) for t in (ROOT/'data/sources/g1_v1/source_manifest.jsonl').read_text().splitlines()]
    excluded={r['source_id'] for r in old};excluded_hashes={r['sha256'] for r in old}
    stream_class=runpy.run_path(str(ROOT/'scripts/fetch_diagnostic_sources.py'),run_name='bounded_stream_reader')['BoundedStream']
    args.output_dir.mkdir(parents=True);(args.output_dir/'speech').mkdir()
    (args.output_dir/'source_plan_frozen.json').write_bytes(planbytes)
    choices={speaker:[] for speaker in plan['speaker_quotas']};transcripts={};complete=False
    report={'status':'running','plan_sha256':sha(planbytes),'training':False,'whole_archive_saved':False,
            'whole_archive_hash_verified':False,'original_source_ids_excluded':sorted(excluded)}
    try:
        with requests.get(plan['speech_archive_url'],stream=True,timeout=90) as response:
            response.raise_for_status();bounded=stream_class(response.raw,plan['max_stream_bytes'])
            with tarfile.open(fileobj=bounded,mode='r|gz') as archive:
                for member in archive:
                    if not member.isfile():continue
                    leaf=PurePosixPath(member.name).name
                    if leaf.endswith('.trans.txt'):
                        if member.size>1024**2:raise ValueError('Unexpected oversized transcript')
                        for line in archive.extractfile(member).read().decode('utf-8').splitlines():
                            key,text=line.split(' ',1);transcripts[key]=text
                    elif re.fullmatch(r'\d+-\d+-\d+\.flac',leaf):
                        sid=leaf[:-5];speaker=sid.split('-')[0]
                        if speaker not in choices or sid in excluded or len(choices[speaker])>=plan['speaker_quotas'][speaker]:continue
                        if member.size>3*1024**2:continue
                        data=archive.extractfile(member).read();info=sf.info(io.BytesIO(data));digest=sha(data)
                        if info.channels==1 and info.samplerate==16000 and 1<=info.duration<=3.8 and digest not in excluded_hashes:
                            (args.output_dir/'speech'/leaf).write_bytes(data)
                            choices[speaker].append({'source_id':sid,'speaker_id':speaker,'path':'speech/'+leaf,'sha256':digest,
                                'sample_rate':16000,'samples':info.frames,'duration_seconds':info.duration,
                                'source_url':plan['speech_archive_url'],'archive_member':member.name,
                                'dataset':'LibriSpeech dev-clean','license':'CC-BY-4.0'})
                            print(json.dumps({'new_source':sid,'speaker_count':len(choices[speaker])}),flush=True)
                    selected=[r for speaker in choices for r in choices[speaker]]
                    if all(len(choices[s])==quota for s,quota in plan['speaker_quotas'].items()) and all(r['source_id'] in transcripts for r in selected):
                        complete=True;break
            report['compressed_bytes_streamed']=bounded.count
        if not complete:raise RuntimeError('Could not satisfy frozen source quotas within archive/cap')
        selected=[dict(r,transcript=transcripts[r['source_id']]) for speaker in choices for r in choices[speaker]]
        if len(selected)!=8 or len({r['source_id'] for r in selected})!=8:raise ValueError('Unexpected denominator')
        with (args.output_dir/'source_manifest.jsonl').open('x',encoding='utf-8') as stream:
            for r in selected:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
        report.update(status='ok',sources=8,speakers=6,all_source_ids_and_bytes_new=True,
                      selected_ids=[r['source_id'] for r in selected])
    except Exception as error:
        report.update(status='error',error_type=type(error).__name__,error=str(error));raise
    finally:
        (args.output_dir/'fetch_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
