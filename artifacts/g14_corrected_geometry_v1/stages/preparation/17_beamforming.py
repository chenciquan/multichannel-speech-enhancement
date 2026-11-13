"""Fixed, offline reference-4 baselines using official torchaudio primitives.

Estimated-mask MVDR accepts only noisy channels and a mono speech estimate.
Oracle variants must be labelled by the calling experiment, never implied blind.
"""
import numpy as np
import torch
import torchaudio.functional as AF

SAMPLE_RATE=16000
REFERENCE=4
N_FFT=512
HOP=128


def _input(noisy):
    a=np.asarray(noisy)
    if np.iscomplexobj(a) or a.ndim!=2 or a.shape[1]!=6 or not 512<=len(a)<=64000:
        raise ValueError('Expected real512..64000 by6 mono16k sample frames')
    a=np.ascontiguousarray(a,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError('Nonfinite input')
    return torch.from_numpy(a.T.copy())


def _stft(x):
    return torch.stft(x,n_fft=N_FFT,hop_length=HOP,win_length=N_FFT,
        window=torch.hann_window(N_FFT,dtype=torch.float64),center=True,pad_mode='reflect',return_complex=True)


def _wave(spec,length):
    x=torch.istft(spec,n_fft=N_FFT,hop_length=HOP,win_length=N_FFT,
        window=torch.hann_window(N_FFT,dtype=torch.float64),center=True,length=length)
    a=x.numpy().astype(np.float32)
    if not np.isfinite(a).all():raise FloatingPointError('Nonfinite beamformed output; no fallback')
    return a


@torch.inference_mode()
def mask_mvdr(noisy,speech_estimate):
    x=_input(noisy);e=np.asarray(speech_estimate)
    if np.iscomplexobj(e) or e.shape!=(x.shape[-1],) or not np.isfinite(e).all():
        raise ValueError('Estimate must be finite real mono with exact input length')
    X=_stft(x);E=_stft(torch.from_numpy(np.ascontiguousarray(e,dtype=np.float64)))
    speech_power=E.abs().square();residual_power=(X[REFERENCE]-E).abs().square()
    mask=speech_power/(speech_power+residual_power+1e-10)
    ps=AF.psd(X,mask,normalize=True,eps=1e-10);pn=AF.psd(X,1-mask,normalize=True,eps=1e-10)
    weights=AF.mvdr_weights_souden(ps,pn,REFERENCE,diagonal_loading=True,diag_eps=1e-7,eps=1e-8)
    if not torch.isfinite(weights).all():raise FloatingPointError('Nonfinite MVDR weights; no fallback')
    output=AF.apply_beamforming(weights,X)
    return _wave(output,x.shape[-1]),{'mask_mean':float(mask.mean()),'mask_fraction_below_001':float((mask<.01).double().mean()),
       'weight_max_magnitude':float(weights.abs().max()),'oracle_information_inside_function':False}


@torch.inference_mode()
def oracle_delay_and_sum(noisy,delays_samples):
    x=_input(noisy);d=np.asarray(delays_samples,dtype=np.float64)
    if d.shape!=(6,) or not np.isfinite(d).all() or abs(d[REFERENCE])>1e-12:
        raise ValueError('Expected six finite arrival delays relative to reference4')
    bins=torch.arange(N_FFT//2+1,dtype=torch.float64)
    steering=torch.exp(-2j*torch.pi*bins[:,None]*torch.from_numpy(d)[None,:]/N_FFT)
    weights=steering/6
    return _wave(AF.apply_beamforming(weights,_stft(x)),x.shape[-1])
