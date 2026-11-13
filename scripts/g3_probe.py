#!/usr/bin/env python3
import json,runpy,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.adaptation import attach_lora,mask_loss


def main():
    out=ROOT/'artifacts/g3_overnight_v1/resource_probe.json'
    if out.exists():raise RuntimeError('Probe report already exists')
    adapter=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g3_runtime_adapter')
    repo=ROOT/'third_party/McNet';checkpoint=repo/adapter['OFFLINE_CHECKPOINT'];adapter['verify_upstream'](repo,checkpoint)
    model,_,metadata=adapter['load_model'](repo,checkpoint,torch.device('cuda'))
    case=json.loads((ROOT/'artifacts/g2_confirmation_v1/mcnet/cases.jsonl').read_text().splitlines()[0])
    raw,_=sf.read(case['noisy_path'],dtype='float32',always_2d=True);clean,_=sf.read(case['reference_path'],dtype='float32')
    x=torch.from_numpy(np.ascontiguousarray(raw[:16000].T)).unsqueeze(0).cuda();s=torch.from_numpy(clean[:16000].copy()).unsqueeze(0).cuda()
    with torch.no_grad():before,_=model(x)
    count=attach_lora(model);model.eval()
    with torch.no_grad():after,_=model(x)
    initial_error=float((before-after).abs().max());assert initial_error<1e-6 and count==3584
    records=[];successful=None
    for amp in (True,False):
        try:
            model.train();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(5):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                with torch.autocast('cuda',enabled=amp,dtype=torch.float16):prediction,data=model(x)
                loss=mask_loss(model,x,s,prediction,data);loss.backward();torch.cuda.synchronize()
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                gradients=[p.grad for p in model.parameters() if p.requires_grad]
                if any(g is None or not torch.isfinite(g).all() for g in gradients):raise ValueError('Invalid adapter gradients')
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen parameter received gradient')
                times.append(time.perf_counter()-start)
            successful={'amp':amp,'segment_samples':16000,'batch_size':1,'mean_step_seconds':float(np.mean(times[1:])),
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                        'loss':float(loss),'backward_passes':5,'optimizer_steps':0}
            records.append({'status':'ok',**successful});break
        except Exception as error:
            records.append({'status':'error','amp':amp,'error_type':type(error).__name__,'error':str(error)})
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    result={'status':'ok' if successful else 'failed','adapter_parameters':count,'zero_initialization_max_error':initial_error,
            'attempts':records,'selected':successful,'original_checkpoint_unchanged':True,'capacity_input_role':'existing diagnostic waveform only; no optimization; actual training reloads pristine base','model_parameters':metadata['parameter_count']}
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not successful:raise RuntimeError('Backward capacity probe failed')

if __name__=='__main__':main()
