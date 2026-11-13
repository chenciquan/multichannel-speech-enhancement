"""Shared local inference interface to the original verified McNet and LoRA."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import runpy
import time
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def archived_json(source,archive):
    manifest=json.loads(archive.read_text(encoding='utf-8'))
    match=[r for r in manifest['files'] if r['source']==str(source.relative_to(ROOT))]
    if len(match)!=1 or sha256(source)!=match[0]['sha256']:raise ValueError(f'Registry differs from archived evidence: {source}')
    return json.loads(source.read_text(encoding='utf-8'))

def registered_systems():
    systems={'mcnet':{'system':'mcnet','kind':'mcnet'}}
    entries=[('g3',ROOT/'artifacts/g3_overnight_v1/evaluation/systems.json',ROOT/'artifacts/g3_overnight_v1/stages/signal/archive_manifest.json'),
             ('g5',ROOT/'artifacts/g5_rotation_v1/selected_checkpoints.json',ROOT/'artifacts/g5_rotation_v1/stages/training/archive_manifest.json'),
             ('g15',ROOT/'artifacts/g15_rigid_training_v1/registered_checkpoints.json',ROOT/'artifacts/g15_rigid_training_v1/stages/training/archive_manifest.json')]
    for prefix,source,archive in entries:
        if not source.exists():continue
        payload=archived_json(source,archive);rows=payload if isinstance(payload,list) else payload['systems']
        for row in rows:
            if 'path' not in row:continue
            if not isinstance(row.get('checkpoint_sha256'),str):raise ValueError('Registered adapter lacks SHA256')
            system_id=prefix+'/'+row['system'];systems[system_id]={**row,'kind':'adapter','registry_source':str(source.relative_to(ROOT)),
                'registry_sha256':sha256(source),'registry_archive':str(archive.relative_to(ROOT))}
    return systems

def validate_array(audio,sample_rate,max_seconds=4.):
    a=np.asarray(audio)
    if sample_rate!=16000:raise ValueError('Expected16000Hz; resampling is not implicit')
    if a.ndim!=2 or a.shape[1]!=6:raise ValueError('Expected six actual microphone channels')
    if a.shape[0]<512 or a.shape[0]>int(16000*max_seconds):raise ValueError(f'Input must be512 samples to{max_seconds:g}s; no silent chunking')
    if not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a):raise ValueError('Audio must contain real numeric samples')
    with np.errstate(over='ignore',invalid='ignore'):a=np.ascontiguousarray(a,dtype=np.float32)
    if not np.isfinite(a).all():raise ValueError('Input contains NaN, infinity or float32 overflow')
    return a

def read_input(path,max_seconds=4.):
    path=Path(path);info=sf.info(path)
    if info.samplerate!=16000 or info.channels!=6 or not 512<=info.frames<=int(16000*max_seconds):
        raise ValueError('Require six-channel16kHz audio,512 samples to4s; no automatic conversion')
    audio,sr=sf.read(path,dtype='float32',always_2d=True);return validate_array(audio,sr,max_seconds)

@lru_cache(maxsize=1)
def bridge():return runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='shared_verified_mcnet')

@dataclass
class InferenceRuntime:
    model:object
    icirm:object
    device:object
    metadata:dict

def load_system(system_id,device='cuda'):
    systems=registered_systems()
    if system_id not in systems:raise ValueError(f'Unknown registered system: {system_id}')
    record=systems[system_id];base=bridge();repo=ROOT/'third_party/McNet';checkpoint=repo/base['OFFLINE_CHECKPOINT']
    verified=base['verify_upstream'](repo,checkpoint)
    if record['kind']=='adapter':
        adapter_path=ROOT/record['path']
        if sha256(adapter_path)!=record['checkpoint_sha256']:raise ValueError('Adapter SHA256 differs from frozen registry')
    import torch
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    target=torch.device(device);start=time.perf_counter();model,icirm,metadata=base['load_model'](repo,checkpoint,target);adapter_parameters=0
    if record['kind']=='adapter':
        from mcse.adaptation import attach_lora,load_adapter
        adapter_parameters=attach_lora(model);assert adapter_parameters==3584
        saved=torch.load(adapter_path,map_location='cpu',weights_only=False)
        if saved['step']!=record['step']:raise ValueError('Adapter step differs from registry')
        if 'seed' in record and saved['seed']!=record['seed']:raise ValueError('Adapter seed differs from registry')
        load_adapter(model,saved['adapter'])
    model.eval().requires_grad_(False)
    if target.type=='cuda':torch.cuda.synchronize()
    meta={'system_id':system_id,'kind':record['kind'],'base_checkpoint_sha256':verified['checkpoint']['sha256'],
       'adapter_sha256':record.get('checkpoint_sha256'),'adapter_path':record.get('path'),'registry_source':record.get('registry_source'),
       'registry_sha256':record.get('registry_sha256'),'adapter_parameters':adapter_parameters,'base_parameters':metadata['parameter_count'],
       'device':str(target),'dtype':'float32','load_seconds_in_shared_process':time.perf_counter()-start,'ref_channel_zero_based':4,
       'streaming':False,'resampling':False,'per_utterance_normalization':False,'chunking':False,'lora_merge':False}
    return InferenceRuntime(model,icirm,target,meta)

def enhance_tensor(runtime,x):
    """Tensor[B,6,T] to Tensor[B,T], no IO or normalization."""
    import torch
    with torch.inference_mode():
        prediction,data=runtime.model(x)
        return bridge()['reconstruct_waveform'](runtime.model,runtime.icirm,x,prediction.to(torch.complex64),x.shape[-1])

def enhance_array(runtime,audio):
    import torch
    a=validate_array(audio,16000);start=time.perf_counter()
    x=torch.from_numpy(np.ascontiguousarray(a.T))[None].to(runtime.device)
    y=enhance_tensor(runtime,x)[0].cpu().numpy()
    if y.shape!=(len(a),) or not np.isfinite(y).all():raise ValueError('Invalid enhanced waveform')
    return np.ascontiguousarray(y,dtype=np.float32),time.perf_counter()-start
