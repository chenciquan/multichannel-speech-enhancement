"""CPU-only saved-wave filters and metric worker for G22."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from functools import lru_cache
import time
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from g22_common import ROOT,OUT,sha,metadata
from mcse.beamforming import mask_mvdr,_input,_stft,_wave,REFERENCE
from mcse.wiener import mask_mwf,wiener_weights
from mcse.components import pcm24_field,component_metrics
from mcse.reverberation import full_speech
import g4_run as metric


def initialize():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def audio(path,digest):
    assert sha(path)==digest,path
    a,sr=sf.read(path,dtype='float32');assert sr==16000 and np.isfinite(a).all()
    return a


def save_and_score(record,y,folder):
    y=np.asarray(y)
    assert y.dtype==np.float32 and y.shape==(record['frames'],) and np.isfinite(y).all()
    path=folder/record['system']/(record['sample_id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
    saved,sr=sf.read(path,dtype='float32');assert sr==16000 and np.array_equal(saved,y)
    record.update(output_path=str(path),output_sha256=sha(path))
    return score(record)


def score(record):
    result=metric.score_signal(record)
    assert result['status']=='ok',result
    if result['condition']=='clean':
        ref=audio(result['reference_path'],result['reference_sha256']).astype(float)
        out=audio(result['output_path'],result['output_sha256']).astype(float)
        energy=float(ref@ref);assert energy>0
        result['clean_levels']={'rms_gain_ratio':float(np.sqrt((out@out)/energy)),
                                'reference_normalized_squared_error':float((out-ref)@(out-ref)/energy)}
    return result


@lru_cache(maxsize=8)
def source_audio(source_id):
    sources={r['source_id']:r for r in metadata(OUT/'selected_sources.json')}
    s=sources[source_id];path=ROOT/'data/g17_v1/sources'/s['path'];assert sha(path)==s['sha256']
    mono,sr=sf.read(path,dtype='float64');assert sr==16000 and len(mono)==s['samples']
    return mono


@lru_cache(maxsize=24)
def clean_field(source_id,rir_path,scale):
    rir_hashes={r['path']:r['sha256'] for r in metadata(OUT/'RIR_manifest.json')}
    rir_hashes.update({r['path']:r['sha256'] for r in metadata(ROOT/'artifacts/g18_room_stress_v1/RIR_manifest.json')})
    assert sha(rir_path)==rir_hashes[rir_path]
    return pcm24_field(full_speech(source_audio(source_id),np.load(rir_path))*scale)


@torch.inference_mode()
def filtered(job):
    method,case,direct=job
    assert not torch.cuda.is_initialized()
    x=audio(case['noisy_path'],case['input_sha256']);start=time.perf_counter()
    row={**case,'audio_seconds':case['frames']/16000}
    if direct is not None:
        e=audio(direct['output_path'],direct['output_sha256'])
        if method=='mvdr':y,info=mask_mvdr(x,e)
        else:y,info=mask_mwf(x,e,mu=1.)
        info.pop('oracle_information_inside_function',None)
        info.update(oracle_clean_reference=False,oracle_DOA=False)
        row.update(system=direct['system']+'_mask_'+method,source_system=direct['system'],
                   mask_estimate_path=direct['output_path'],mask_estimate_sha256=direct['output_sha256'],
                   direct_inference_seconds=direct['inference_seconds'],beamforming_diagnostics=info,
                   oracle_clean_information=False,deployable=True)
    else:
        assert case['condition']!='clean'
        s=clean_field(case['source_id'],case['speech_RIR_path'],case['common_scale'])
        ref=audio(case['reference_path'],case['reference_sha256']);assert np.array_equal(s[:,4],ref)
        n=x.astype(float)-s.astype(float)
        X,S,N=(_stft(_input(a)) for a in (x,s,n));ones=torch.ones(X.shape[1:],dtype=torch.float64)
        ps=AF.psd(S,ones,normalize=False)/X.shape[-1];pn=AF.psd(N,ones,normalize=False)/X.shape[-1]
        px=AF.psd(X,ones,normalize=False)/X.shape[-1]
        if method=='mvdr':w=AF.mvdr_weights_souden(ps,pn,REFERENCE,diagonal_loading=True,diag_eps=1e-7,eps=1e-8)
        else:w,_=wiener_weights(ps,pn,mu=1.)
        y,ys,yn=(_wave(AF.apply_beamforming(w,z),case['frames']) for z in (X,S,N))
        linearity=float(np.abs(y.astype(float)-ys.astype(float)-yn.astype(float)).max())
        assert linearity<1e-6*(1+float(np.abs(y).max()))
        row.update(system='truth_covariance_'+method,oracle_clean_information=True,deployable=False,
                   covariance_cross_relative_max=float((px-ps-pn).abs().max()/(px.abs().max()+1e-30)),
                   linearity_max_abs=linearity,components=component_metrics(ref,n[:,4],ys,yn))
    row['inference_seconds']=time.perf_counter()-start
    return save_and_score(row,y,OUT/'evaluation'/('blind_audio' if direct is not None else 'truth_audio'))
