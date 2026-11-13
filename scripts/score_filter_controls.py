#!/usr/bin/env python3
"""Score saved DSP FLOAT WAVs; preserve the original pre-serialization records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mcse.evaluation import MetricEvaluator


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():p.error('Refusing to overwrite scores')
    original=[json.loads(t) for t in args.cases.read_text().splitlines() if t.strip()]
    if len({(r['sample_id'],r['method']) for r in original})!=len(original):raise ValueError('Duplicate control key')
    args.output_dir.mkdir(parents=True);evaluator=MetricEvaluator();results=[];differences=[]
    with (args.output_dir/'scores.jsonl').open('x') as stream:
        for r in original:
            if sha(r['output_path'])!=r['output_sha256']:raise ValueError('Control audio changed')
            s,sr=sf.read(r['reference_path'],dtype='float64');y,ysr=sf.read(r['output_path'],dtype='float64')
            if sr!=16000 or ysr!=sr or y.shape!=s.shape:raise ValueError('Changed alignment/format')
            item={k:v for k,v in r.items() if k!='metrics'}
            item['metrics']=evaluator.evaluate(s,y,sr);item['metrics_on_serialized_audio']=True
            differences.append(abs(item['metrics']['si_sdr_db']-r['metrics']['si_sdr_db']))
            results.append(item);stream.write(json.dumps(item,allow_nan=False)+'\n')
    grouped={}
    for spectrum in sorted({r['noise_spectrum'] for r in results}):
        grouped[spectrum]={}
        for method in sorted({r['method'] for r in results}):
            g=[r for r in results if r['noise_spectrum']==spectrum and r['method']==method]
            grouped[spectrum][method]={'n':len(g),**{m:float(np.mean([r['metrics'][m] for r in g])) for m in ('si_sdr_db','pesq_wb','stoi')}}
    summary={'status':'ok','evaluations':len(results),'grouped':grouped,'source_cases_sha256':sha(args.cases),
             'maximum_si_sdr_difference_from_prewrite_scoring_db':max(differences),
             'metrics_on_serialized_audio':True,'audio_altered':False,'model_rerun':False,'training':False,
             'reason':'Original controls evaluated float64 filter output before FLOAT WAV float32 quantization; rescore actual stored bytes for exact evidence consistency'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({'status':'ok','evaluations':len(results),'max_si_difference_db':max(differences)}))


if __name__=='__main__':main()
