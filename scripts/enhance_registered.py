#!/usr/bin/env python3
"""Local6-channel16kHz inference with archived, registered adapter IDs."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.runtime import registered_systems,read_input,load_system,enhance_array,sha256

def run(args):
    if args.list_systems:
        print('\n'.join(sorted(registered_systems())));return None
    if args.input is None or args.output is None or args.report is None:raise ValueError('--input,--output,--report are required for inference')
    input_path=args.input.resolve();output=args.output.resolve();report=args.report.resolve()
    if output==report:raise ValueError('Output audio and report must have different paths')
    for path in (output,report):
        if path.exists():raise FileExistsError(f'Refusing to overwrite: {path}')
    audio=read_input(input_path)
    if args.system not in registered_systems():raise ValueError('Unknown registered system; use --list-systems')
    runtime=load_system(args.system,args.device);enhanced,seconds=enhance_array(runtime,audio)
    output.parent.mkdir(parents=True,exist_ok=True);report.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:sf.write(stream,enhanced,16000,format='WAV',subtype='FLOAT')
    written,sr=sf.read(output,dtype='float32');assert sr==16000 and np.array_equal(written,enhanced)
    result={'status':'ok','created_utc':datetime.now(timezone.utc).isoformat(),'model':runtime.metadata,
       'input':{'path':str(input_path),'sha256':sha256(input_path),'channels':6,'frames':len(audio),'sample_rate':16000,'duration_seconds':len(audio)/16000},
       'output':{'path':str(output),'sha256':sha256(output),'channels':1,'frames':len(written),'sample_rate':16000,'subtype':'FLOAT','abs_peak':float(np.abs(written).max()),'samples_above_one':int(np.count_nonzero(np.abs(written)>1))},
       'array_to_array_seconds_including_transfers':seconds,'inference_rtf_excluding_load_and_file_io':seconds/(len(audio)/16000),
       'quality_metrics_computed':False,'ASR_run':False,'limitations':'Offline model,<=4s interface, nominal microphone ordering, no quality/generalization guarantee. Local research use; not public redistribution.'}
    with report.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps({'status':'ok','system':args.system,'output':str(output),'report':str(report),'frames':len(written)},ensure_ascii=False));return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--report',type=Path)
    p.add_argument('--system',default='mcnet');p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--list-systems',action='store_true');args=p.parse_args()
    try:run(args)
    except (ValueError,FileExistsError,FileNotFoundError,RuntimeError) as error:p.exit(1,f'{type(error).__name__}: {error}\n')
if __name__=='__main__':main()
