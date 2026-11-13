"""Conditional component response of an already fixed MVDR filter.

True clean fields enter evaluation only after weights have been estimated.
No component metric here establishes linguistic content or human intelligibility.
"""
import io
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from .beamforming import _input,_stft,_wave,REFERENCE
from .evaluation import si_sdr,AudioValidationError


def pcm24_field(field):
    a=np.asarray(field,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=6 or not np.isfinite(a).all() or np.abs(a).max()>=1:
        raise ValueError('Expected finite unclipped6-channel clean field')
    buffer=io.BytesIO();sf.write(buffer,a,16000,format='WAV',subtype='PCM_24');buffer.seek(0)
    result,sr=sf.read(buffer,dtype='float32',always_2d=True);assert sr==16000
    return result


def component_metrics(reference_speech,reference_noise,output_speech,output_noise,power_floor=1e-30):
    if any(np.iscomplexobj(a) for a in (reference_speech,reference_noise,output_speech,output_noise)):
        raise ValueError('Time-domain components must be real')
    if not np.isfinite(power_floor) or power_floor<=0:raise ValueError('Positive finite floor required')
    arrays=[np.asarray(a,dtype=np.float64) for a in (reference_speech,reference_noise,output_speech,output_noise)]
    if any(a.ndim!=1 or not np.isfinite(a).all() for a in arrays) or len({a.shape for a in arrays})!=1 or len(arrays[0])<2:
        raise ValueError('Finite aligned mono component arrays required,no fitting')
    s,n,ys,yn=arrays;ps,pn,pys,pyn=[float(np.mean(a*a)) for a in arrays]
    if ps<=power_floor or pn<=power_floor:raise ValueError('Reference speech/noise power must exceed floor; clean/silent cases excluded')
    logratio=lambda a,b:float(10*np.log10(max(a,power_floor)/max(b,power_floor)))
    record={'speech_error_nmse':float(np.mean((ys-s)**2)/ps),'target_gain_db':logratio(pys,ps),
      'noise_reduction_db':logratio(pn,pyn),'component_input_snr_db':logratio(ps,pn),'component_output_snr_db':logratio(pys,pyn),
      'input_speech_power':ps,'input_noise_power':pn,'output_speech_power':pys,'output_noise_power':pyn,
      'output_speech_power_floored':pys<power_floor,'output_noise_power_floored':pyn<power_floor,'power_floor':power_floor}
    try:record.update(target_component_si_sdr_db=si_sdr(s,ys),target_component_si_sdr_status='ok')
    except AudioValidationError as e:record.update(target_component_si_sdr_db=None,target_component_si_sdr_status='undefined',target_component_si_sdr_reason=str(e))
    record['snr_accounting_error_db']=abs((record['component_output_snr_db']-record['component_input_snr_db'])-(record['target_gain_db']+record['noise_reduction_db']))
    return record


@torch.inference_mode()
def decompose_fixed_mvdr(noisy,estimate,clean_field,power_floor=1e-30):
    x=_input(noisy);s=_input(clean_field);e=np.asarray(estimate)
    if s.shape!=x.shape or e.shape!=(x.shape[-1],) or np.iscomplexobj(e) or not np.isfinite(e).all():raise ValueError('Component/estimate shape or finite-value mismatch')
    X=_stft(x);E=_stft(torch.from_numpy(np.ascontiguousarray(e,dtype=np.float64)))
    p=E.abs().square();mask=p/(p+(X[REFERENCE]-E).abs().square()+1e-10)
    ps=AF.psd(X,mask,normalize=True,eps=1e-10);pn=AF.psd(X,1-mask,normalize=True,eps=1e-10)
    w=AF.mvdr_weights_souden(ps,pn,REFERENCE,diagonal_loading=True,diag_eps=1e-7,eps=1e-8)
    if not torch.isfinite(w).all():raise FloatingPointError('Nonfinite fixed weights')
    y=_wave(AF.apply_beamforming(w,X),x.shape[-1]);ys=_wave(AF.apply_beamforming(w,_stft(s)),x.shape[-1])
    n=x-s;yn=_wave(AF.apply_beamforming(w,_stft(n)),x.shape[-1])
    metrics=component_metrics(s[REFERENCE].numpy(),n[REFERENCE].numpy(),ys,yn,power_floor)
    metrics.update(linearity_max_abs=float(np.max(np.abs(y.astype(np.float64)-ys.astype(np.float64)-yn.astype(np.float64)))),
      weight_norm_squared_mean=float(w.abs().square().sum(-1).mean()),weight_norm_squared_max=float(w.abs().square().sum(-1).max()))
    return y,ys,yn,metrics
