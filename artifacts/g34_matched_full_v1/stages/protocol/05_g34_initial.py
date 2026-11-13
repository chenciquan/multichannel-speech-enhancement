"""Six TRAIN-interface controls; do not loosen the failed G33 cIRM gate."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import runpy
import time
import traceback
import numpy as np
import torch
from g34_common import ROOT,OUT,DATA,sha,dump,now,meta,guard,configure,state_digest,status
from mcse.adaptation import attach_lora


def main():
    plan=guard();started=time.monotonic();backend=configure(torch)
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g34_initial_runtime')
    repo=ROOT/'third_party/McNet';cp=repo/rt['OFFLINE_CHECKPOINT'];models={};decoders={};initial={}
    for mode in ('full','lora'):
        torch.manual_seed(plan['seed']);model,icirm,_=rt['load_model'](repo,cp,torch.device('cuda'))
        if mode=='lora':assert attach_lora(model)==3584
        else:model.requires_grad_(True)
        model.train();models[mode]=model;decoders[mode]=icirm
        initial[mode]={'base':state_digest(model.state_dict(),'base'),'all':state_digest(model.state_dict()),
            'adapter':state_digest(model.state_dict(),'adapter') if mode=='lora' else None}
    assert initial['full']['base']==initial['lora']['base']=='ac8e5ec85f4dc02e545a817b8f636627918efc2c4288da48af7f2f8b71d6a6d8'
    assert all(torch.count_nonzero(p)==0 for n,p in models['lora'].named_parameters() if n.endswith('lora_B'))
    xmap=np.load(DATA/'train_x.npy',mmap_mode='r');records=[];predictions={};waves={}
    for index in plan['initial_indices']:
        assert time.monotonic()-started<plan['max_initial_seconds']
        x=torch.from_numpy(np.array(xmap[index],copy=True))[None].cuda()
        for mode,model in models.items():
            torch.manual_seed(plan['seed']+index);pred,_=model(x)
            with torch.no_grad():wave=rt['reconstruct_waveform'](model,decoders[mode],x,pred.to(torch.complex64),x.shape[-1])[0]
            predictions[f'{mode}_{index}']=pred.detach().cpu().numpy();waves[f'{mode}_{index}']=wave.cpu().numpy()
            del pred,wave
        def compare(a,b):
            assert np.isfinite(a).all() and np.isfinite(b).all()
            delta=np.abs(a-b)
            return {'max_abs':float(delta.max()),'within_tolerance':bool(np.allclose(a,b,rtol=plan['initial_wave_rtol'],atol=plan['initial_wave_atol'])),
                'failing_elements':int((delta>plan['initial_wave_atol']+plan['initial_wave_rtol']*np.abs(b)).sum())}
        records.append({'index':index,'prediction':compare(predictions[f'lora_{index}'],predictions[f'full_{index}']),
            'waveform':compare(waves[f'lora_{index}'],waves[f'full_{index}'])});del x
    dump(OUT/'initial_records.json',records,True)
    with (OUT/'initial_predictions.npz').open('xb') as f:np.savez(f,**predictions)
    with (OUT/'initial_waveforms.npz').open('xb') as f:np.savez(f,**waves)
    assert all(state_digest(m.state_dict())==initial[k]['all'] and all(p.grad is None for p in m.parameters()) for k,m in models.items())
    assert all(r['waveform']['within_tolerance'] for r in records),'Initial six-record waveform gate failed; preserve and stop'
    schema={mode:{n:list(t.shape) for n,t in model.state_dict().items()} for mode,model in models.items()}
    guard();dump(OUT/'INITIAL_COMPLETE.json',{'utc':now(),'status':'ok','backend':backend,'initial':initial,'state_schema':schema,'records':6,
        'wave_passes':6,'cirm_passes':sum(r['prediction']['within_tolerance'] for r in records),'optimizer_updates':0,
        'old_G33_gate_still_failed':True,'checkpoint_sha256':sha(cp),'elapsed_seconds':time.monotonic()-started,
        'hashes':{n:sha(OUT/n) for n in ('initial_records.json','initial_predictions.npz','initial_waveforms.npz')}},True)
    print({'status':'initial_six_waveforms_passed','cirm_passes':sum(r['prediction']['within_tolerance'] for r in records)},flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:dump(OUT/'initial_failure.json',{'utc':now(),'error':repr(e)});traceback.print_exc();raise
