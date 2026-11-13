"""Full speech and steady background convolutions for the frozen room probe."""
import numpy as np
from scipy.signal import fftconvolve


def _inputs(mono, response):
    x=np.asarray(mono,dtype=np.float64);h=np.asarray(response,dtype=np.float64)
    if x.ndim!=1 or not x.size or h.ndim!=2 or h.shape[1]!=6 or not h.shape[0]:
        raise ValueError('Expected nonempty mono source and [RIR samples,6] response')
    if not np.isfinite(x).all() or not np.isfinite(h).all():raise ValueError('Nonfinite convolution input')
    return x,h


def full_speech(mono,response):
    x,h=_inputs(mono,response)
    return np.stack([fftconvolve(x,h[:,i],mode='full') for i in range(6)],axis=1)


def continuous_noise(mono,response,preroll,frames):
    x,h=_inputs(mono,response)
    if type(preroll) is not int or type(frames) is not int or preroll<h.shape[0]-1 or frames<1 or len(x)!=preroll+frames:
        raise ValueError('Noise requires complete preroll and exactly the requested continuous segment')
    return np.stack([fftconvolve(x,h[:,i],mode='full')[preroll:preroll+frames] for i in range(6)],axis=1)


def schroeder_t20(response,sample_rate=16000):
    h=np.asarray(response,dtype=np.float64)
    if h.ndim!=1 or not np.isfinite(h).all() or not np.any(h):raise ValueError('Invalid response')
    edc=np.cumsum(h[::-1]**2)[::-1];db=10*np.log10(np.maximum(edc/edc[0],1e-30));selected=(db<=-5)&(db>=-25)
    if selected.sum()<20:return None
    slope,_=np.polyfit(np.arange(len(h))[selected]/sample_rate,db[selected],1)
    return float(-60/slope) if slope<0 else None
