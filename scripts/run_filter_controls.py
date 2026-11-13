#!/usr/bin/env python3
"""Fixed40Hz DSP controls against the original, unfiltered clean target; no training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mcse.spatial_noise import highpass40
from mcse.evaluation import MetricEvaluator


def rows(path):return [json.loads(t) for t in Path(path).read_text().splitlines() if t.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed-replica',type=int,help='Fixed subset for deterministic point controls; no source selection')
    args=p.parse_args()
    if args.output_dir.exists():p.error('Refusing to overwrite prior controls')
    manifest={r['sample_id']:r for r in rows(args.manifest)}
    cases=[r for r in rows(args.cases) if manifest[r['sample_id']]['noise_field']=='original_point'
           and (args.seed_replica is None or manifest[r['sample_id']].get('seed_replica',0)==args.seed_replica)]
    args.output_dir.mkdir(parents=True);(args.output_dir/'audio').mkdir()
    frozen={'cases_sha256':sha(args.cases),'manifest_sha256':sha(args.manifest),
            'case_ids':[r['sample_id'] for r in cases],'filter':'4th-order40Hz Butterworth forward-backward; fixed0.25s padding',
            'target':'same unfiltered clean reference','model_training':False,'selected_seed_replica':args.seed_replica}
    (args.output_dir/'frozen_inputs.json').write_text(json.dumps(frozen,indent=2))
    adapter=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='filter_control_adapter')
    repo=ROOT/'third_party/McNet';checkpoint=repo/adapter['OFFLINE_CHECKPOINT']
    upstream=adapter['verify_upstream'](repo,checkpoint)
    model,icirm,model_info=adapter['load_model'](repo,checkpoint,torch.device('cuda'))
    evaluator=MetricEvaluator();results=[]
    with torch.inference_mode(),(args.output_dir/'cases.jsonl').open('x') as stream:
        for index,c in enumerate(cases):
            m=manifest[c['sample_id']]
            s,sr=sf.read(c['reference_path'],dtype='float64')
            x,_=sf.read(c['noisy_path'],dtype='float64',always_2d=True)
            y,_=sf.read(c['enhanced_path'],dtype='float64')
            if sr!=16000 or sha(c['enhanced_path'])!=c['enhanced_file']['sha256']:raise ValueError('Original audio changed')
            filtered=highpass40(x).astype(np.float32)
            tensor=torch.from_numpy(np.ascontiguousarray(filtered.T)).unsqueeze(0).cuda()
            prediction,data=model(tensor)
            pre=adapter['reconstruct_waveform'](model,icirm,tensor,prediction,data['original_time_len'])[0].cpu().numpy()
            outputs={'noisy':x[:,4],'mcnet':y,'highpass_noisy':highpass40(x[:,4]),
                     'mcnet_then_highpass':highpass40(y),'highpass_then_mcnet':pre,'highpass_clean_sanity':highpass40(s)}
            for name,audio in outputs.items():
                path=args.output_dir/'audio'/f'{index:03d}_{name}.wav'
                sf.write(path,audio,16000,subtype='FLOAT')
                saved_audio,_=sf.read(path,dtype='float64')
                item={'sample_id':c['sample_id'],'source_id':c['source_id'],'speaker_id':c['speaker_id'],
                      'noise_id':c['noise_id'],'noise_spectrum':m.get('noise_spectrum','raw'),'method':name,
                      'reference_path':c['reference_path'],'noisy_path':c['noisy_path'],
                      'output_path':str(path.resolve()),'output_sha256':sha(path),'metrics':evaluator.evaluate(s,saved_audio,16000),
                      'metrics_on_serialized_audio':True,
                      'training':False,'target_filtered':False}
                results.append(item);stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush()
            print(json.dumps({'case':index+1,'total':len(cases)}),flush=True)
    grouped={}
    for spectral in sorted({r['noise_spectrum'] for r in results}):
        grouped[spectral]={}
        for method in outputs:
            group=[r for r in results if r['noise_spectrum']==spectral and r['method']==method]
            grouped[spectral][method]={'n':len(group),**{metric:float(np.mean([r['metrics'][metric] for r in group]))
                                                      for metric in ('si_sdr_db','pesq_wb','stoi')}}
    summary={'status':'ok','input_cases':len(cases),'evaluations':len(results),'grouped':grouped,
             'upstream':upstream,'model':model_info,'training':False,
             'notice':'HPF is a standard fixed DSP comparator, not a proposed neural innovation; target stays unfiltered'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({'status':'ok','grouped':grouped},indent=2))


if __name__=='__main__':main()
