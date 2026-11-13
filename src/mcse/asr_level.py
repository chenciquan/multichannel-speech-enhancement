"""Declared global input-level control for recognition diagnostics only."""
import math
import numpy as np


def standardize_level(samples,target_rms=.05,peak_limit=.95):
    a=np.asarray(samples)
    if a.dtype!=np.float32 or a.ndim!=1 or not a.size or not np.isfinite(a).all():
        raise ValueError('Expected nonempty finite mono float32 samples')
    if not math.isfinite(target_rms) or not math.isfinite(peak_limit) or not 0<target_rms<=peak_limit<=1:
        raise ValueError('Require 0<target RMS<=peak limit<=1')
    x=a.astype(np.float64);rms=float(np.sqrt(np.mean(x*x)));peak=float(np.abs(x).max())
    if rms==0:
        return a.copy(),{'silent':True,'gain':1.,'input_rms':0.,'output_rms':0.,'input_peak':0.,'output_peak':0.,'peak_limited':False}
    by_rms=target_rms/rms;by_peak=peak_limit/peak;gain=min(by_rms,by_peak)
    y=(x*gain).astype(np.float32)
    assert np.isfinite(y).all() and np.abs(y).max()<=peak_limit+1e-7
    return y,{'silent':False,'gain':gain,'input_rms':rms,'input_peak':peak,
              'output_rms':float(np.sqrt(np.mean(y.astype(float)**2))),'output_peak':float(np.abs(y).max()),
              'peak_limited':bool(by_peak<by_rms)}
