"""Classical regularized MWF, using a common-time mask covariance estimator.

This is a mature linear-filter control, not a new enhancement network.
No clean source, transcript, geometry or oracle statistic enters the filter.
"""
import math
import numpy as np
import torch
import torchaudio.functional as AF
from .beamforming import _input,_stft,_wave,REFERENCE


def masked_covariances(X,mask):
    if X.ndim!=3 or X.shape[0]!=6 or not torch.is_complex(X) or mask.shape!=X.shape[1:]:raise ValueError('Require complex[6,F,T] and mask[F,T]')
    if not torch.isfinite(X).all() or not torch.isfinite(mask).all() or (mask<0).any() or (mask>1).any():raise ValueError('Invalid spectrum or mask')
    frames=X.shape[-1]
    if not frames:raise ValueError('Empty spectrum')
    return AF.psd(X,mask,normalize=False)/frames,AF.psd(X,1-mask,normalize=False)/frames


def wiener_weights(ps,pn,mu=1.,relative_loading=1e-7,absolute_loading=1e-10):
    if ps.shape!=pn.shape or ps.ndim!=3 or ps.shape[-2:]!=(6,6) or not torch.is_complex(ps) or not torch.is_complex(pn):raise ValueError('Require complex[F,6,6] covariances')
    if not math.isfinite(mu) or mu<=0 or not math.isfinite(relative_loading) or relative_loading<0 or not math.isfinite(absolute_loading) or absolute_loading<0:raise ValueError('Invalid weighting/loading')
    if not torch.isfinite(ps).all() or not torch.isfinite(pn).all():raise ValueError('Nonfinite covariance')
    for p in (ps,pn):
        if not torch.allclose(p,p.mH,atol=1e-10,rtol=1e-9) or (p.diagonal(dim1=-2,dim2=-1).real.sum(-1)<-1e-12).any():raise ValueError('Invalid Hermitian covariance')
    a=ps+mu*pn;trace=a.diagonal(dim1=-2,dim2=-1).real.sum(-1);loading=relative_loading*trace.clamp_min(0)/6+absolute_loading
    regularized=a+loading[:,None,None]*torch.eye(6,dtype=a.dtype,device=a.device)
    rhs=ps[:,:,REFERENCE];weights=torch.linalg.solve(regularized,rhs)
    if not torch.isfinite(weights).all():raise FloatingPointError('Nonfinite MWF weights; no fallback')
    residual=torch.linalg.vector_norm(torch.einsum('fcd,fd->fc',regularized,weights)-rhs,dim=-1)/(torch.linalg.vector_norm(rhs,dim=-1)+1e-30)
    return weights,{'solve_relative_residual_max':float(residual.max()),'loading_min':float(loading.min()),'loading_max':float(loading.max())}


@torch.inference_mode()
def mask_mwf(noisy,speech_estimate,mu=1.):
    x=_input(noisy);e=np.asarray(speech_estimate)
    if np.iscomplexobj(e) or e.shape!=(x.shape[-1],) or not np.isfinite(e).all():raise ValueError('Estimate must be finite real mono of exact input length')
    X=_stft(x);E=_stft(torch.from_numpy(np.ascontiguousarray(e,dtype=np.float64)))
    power=E.abs().square();residual=(X[REFERENCE]-E).abs().square();mask=power/(power+residual+1e-10)
    ps,pn=masked_covariances(X,mask);weights,info=wiener_weights(ps,pn,mu)
    total=AF.psd(X,torch.ones_like(mask),normalize=False)/X.shape[-1]
    partition_error=float((ps+pn-total).abs().max()/(total.abs().max()+1e-30));assert partition_error<1e-10
    y=_wave(AF.apply_beamforming(weights,X),x.shape[-1])
    info.update(mu=float(mu),mask_mean=float(mask.mean()),covariance_partition_relative_error=partition_error,
        covariance_estimator='complementary mixture masks,common frame count T',relative_loading=1e-7,absolute_loading=1e-10,
        precision='float64/complex128',weight_max_magnitude=float(weights.abs().max()),reference_channel_zero_based=REFERENCE)
    return y,info
