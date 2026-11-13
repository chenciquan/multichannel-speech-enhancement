#!/usr/bin/env python3
"""Bounded, paired LoRA experiment. Final test files are never opened here."""
import argparse, hashlib, json, runpy, sys, time, traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.adaptation import attach_lora,adapter_state,mask_loss
from mcse.evaluation import si_sdr
from pystoi import stoi
BASE=ROOT/'data/g3_v1/dataset';OUT=ROOT/'artifacts/g3_overnight_v1'
PLAN=json.loads((ROOT/'configs/g3_overnight_plan.json').read_text())

def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def utc():return datetime.now(timezone.utc).isoformat()

def digest_base(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():
        if 'lora_' not in n:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_validation():
    folder=BASE/'validation';rows=[]
    for line in (folder/'manifest.jsonl').read_text().splitlines():
        r=json.loads(line);x,sr=sf.read(folder/r['noisy_path'],dtype='float32',always_2d=True)
        s,_=sf.read(folder/r['reference_path'],dtype='float32');assert sr==16000 and x.shape[1]==6
        rows.append((r,torch.from_numpy(np.ascontiguousarray(x.T))[None].cuda(),s))
    return rows

def validate(model,icirm,rt,rows):
    model.eval();scores=[]
    with torch.inference_mode():
        for r,x,s in rows:
            pred,data=model(x);y=rt['reconstruct_waveform'](model,icirm,x,pred.to(torch.complex64),x.shape[-1])[0].cpu().numpy()
            if not np.isfinite(y).all():raise ValueError('Nonfinite validation output')
            scores.append({'sample_id':r['sample_id'],'source_id':r['source_id'],'si_sdr_db':si_sdr(s,y),
                           'stoi':float(stoi(s,y,16000,extended=False))})
    model.train()
    return {'si_sdr_db':float(np.mean([r['si_sdr_db'] for r in scores])),
            'stoi':float(np.mean([r['stoi'] for r in scores])),'cases':scores}

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['ordinary','complete','without_stratification','without_clean_identity'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=3000);a=p.parse_args()
    folder=OUT/'train'/f'{a.variant}_s{a.seed}';folder.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=datetime.fromisoformat(PLAN['training_cutoff_utc'].replace('Z','+00:00')).timestamp()
    dump(folder/'status.json',{'status':'starting','at':utc(),'variant':a.variant,'seed':a.seed})
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.backends.cudnn.benchmark=False
    rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g3_train_runtime')
    repo=ROOT/'third_party/McNet';checkpoint=repo/rt['OFFLINE_CHECKPOINT'];rt['verify_upstream'](repo,checkpoint)
    model,icirm,metadata=rt['load_model'](repo,checkpoint,torch.device('cuda'));count=attach_lora(model)
    assert count==3584;base_before=digest_base(model);model.train()
    optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=PLAN['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=False);rng=np.random.default_rng(a.seed);identity_rng=np.random.default_rng(a.seed+100000)
    xmap=np.load(BASE/'train_x.npy',mmap_mode='r');ymap=np.load(BASE/'train_y.npy',mmap_mode='r');cleanmap=np.load(BASE/'clean_x.npy',mmap_mode='r')
    catalog=json.loads((BASE/'train_catalog.json').read_text());all_ids=np.arange(len(catalog))
    multi=np.array([i for i,r in enumerate(catalog) if r['field']=='multi8'])
    point_by_angle={v:np.array([i for i,r in enumerate(catalog) if r['field']=='point' and r['angle']==v]) for v in PLAN['train_angles']}
    balanced=a.variant in ('complete','without_clean_identity');identity=a.variant in ('complete','without_stratification')
    rows=load_validation();initial=validate(model,icirm,rt,rows);dump(folder/'validation_initial.json',initial)
    config={'variant':a.variant,'seed':a.seed,'steps_requested':a.steps,'batch_size':1,'gradient_accumulation':1,
            'segment_samples':int(xmap.shape[-1]),'speech_core_samples':16000,'guard_samples':128,
            'amp_training':False,'inference_fp32':True,'optimizer':'Adam','learning_rate':PLAN['learning_rate'],
            'gradient_clip_norm':5.,'identity_every':5,'identity_weight':.1 if identity else 0.,
            'balanced_sampling':balanced,'point_sampling_fraction':.5 if balanced else .25,
            'same_fixed_pool':str(BASE/'train_catalog.json'),'rank':4,'alpha':4,'trainable_parameters':count,
            'checkpoint_selection':'max validation mean SI-SDR among checkpoints whose STOI >= initial STOI - 0.005; fallback max SI-SDR with explicit gate failure; no WER tuning',
            'validation_cases':48,'test_access':False,'base_digest_before':base_before,'started_at':utc(),
            'training_cutoff_utc':PLAN['training_cutoff_utc'],'torch_version':torch.__version__}
    dump(folder/'config.json',config);history=[];counts=Counter();losses=[];completed=0
    torch.cuda.reset_peak_memory_stats()
    with (folder/'steps.jsonl').open('x') as log:
        for step in range(1,a.steps+1):
            if time.time()>=deadline:break
            if balanced:
                if rng.random()<.5:
                    angle=int(rng.choice(PLAN['train_angles']));idx=int(rng.choice(point_by_angle[angle]))
                else:idx=int(rng.choice(multi))
            else:idx=int(rng.choice(all_ids))
            row=catalog[idx];counts[row['field']]+=1;counts[f"{row['field']}:{row['angle']}"]+=1
            x=torch.from_numpy(np.array(xmap[idx],copy=True))[None].cuda();s=torch.from_numpy(np.array(ymap[idx],copy=True))[None].cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=False):pred,data=model(x)
            main_loss=mask_loss(model,x,s,pred,data);clean_loss=torch.zeros((),device='cuda')
            if identity and step%5==0:
                ci=int(identity_rng.integers(len(cleanmap)));cx=torch.from_numpy(np.array(cleanmap[ci],copy=True))[None].cuda()
                with torch.autocast('cuda',enabled=False):cp,cd=model(cx)
                cy=rt['reconstruct_waveform'](model,icirm,cx,cp.to(torch.complex64),cx.shape[-1]);cs=cx[:,4,:]
                clean_loss=torch.mean((cy-cs)**2)/(torch.mean(cs**2)+1e-8)
            loss=main_loss+(.1*clean_loss if identity else 0.)
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss at {step}')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            grad_norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],5.,error_if_nonfinite=True)
            scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<scale_before:raise ValueError(f'AMP skipped update at {step}; do not silently count it')
            completed=step;losses.append(float(loss.detach()))
            if step%50==0:
                record={'step':step,'mean_loss_last50':float(np.mean(losses[-50:])), 'mask_loss':float(main_loss.detach()),
                        'clean_loss':float(clean_loss.detach()),'grad_norm':float(grad_norm),'elapsed_seconds':time.perf_counter()-start}
                log.write(json.dumps(record)+'\n');log.flush()
                dump(folder/'status.json',{'status':'training','at':utc(),**record})
            if step%500==0 or step==a.steps:
                val=validate(model,icirm,rt,rows);val.update(step=step,elapsed_seconds=time.perf_counter()-start)
                dump(folder/f'validation_{step:05d}.json',val)
                ckpt=folder/f'adapter_{step:05d}.pt'
                torch.save({'adapter':adapter_state(model),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                            'numpy_sampler_state':rng.bit_generator.state,'identity_sampler_state':identity_rng.bit_generator.state,'torch_rng_state':torch.get_rng_state(),
                            'cuda_rng_state':torch.cuda.get_rng_state(),'step':step,'variant':a.variant,'seed':a.seed,
                            'base_checkpoint_sha256':rt['sha256'](checkpoint) if 'sha256' in rt else '24b8c556cc52990e4d7db83da4909b7e8a1dc4a0c14880b98381bb62818beecf'},ckpt)
                history.append({'step':step,'path':str(ckpt.relative_to(ROOT)), 'si_sdr_db':val['si_sdr_db'],'stoi':val['stoi']})
                print(json.dumps({'variant':a.variant,'seed':a.seed,**history[-1]}),flush=True)
    base_after=digest_base(model);assert base_before==base_after,'Frozen base changed'
    if any(v.grad is not None for v in model.parameters() if not v.requires_grad):raise AssertionError('Base gradient found')
    if not history:raise RuntimeError('No completed checkpoint')
    eligible=[r for r in history if r['stoi']>=initial['stoi']-.005]
    selected=max(eligible or history,key=lambda r:r['si_sdr_db'])
    result={'status':'ok' if completed==a.steps else 'cutoff','variant':a.variant,'seed':a.seed,
            'completed_steps':completed,'selected':selected,'stoi_gate_passed':bool(eligible),'history':history,
            'initial_si_sdr_db':initial['si_sdr_db'],'initial_stoi':initial['stoi'],
            'base_digest_before':base_before,'base_digest_after':base_after,'base_unchanged':True,
            'elapsed_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'sampler_counts':dict(counts),'finished_at':utc()}
    dump(folder/'result.json',result);dump(folder/'status.json',result)

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();sys.exit(1)
