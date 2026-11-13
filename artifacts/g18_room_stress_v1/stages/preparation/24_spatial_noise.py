"""Controlled noise-field diagnostics; no enhancement model or training.

The conditional finite-wave field fixes its reference waveform analytically.
It represents a sum of physically delayed source signals, not a real recording.
"""
from __future__ import annotations
import math
import numpy as np

from .simulation import geometric_delays, render_plane_wave, GUARD_SAMPLES


def highpass40(signal, sample_rate=16000):
    """Fixed offline diagnostic control; no tuning or speech-reference access."""
    from scipy.signal import butter, sosfiltfilt
    data=np.asarray(signal,dtype=np.float64)
    if sample_rate != 16000 or data.ndim not in (1,2) or len(data)<512 or not np.isfinite(data).all():
        raise ValueError("Expected finite mono/multichannel16k audio with >=512 samples")
    sos=butter(4,40,btype="highpass",fs=sample_rate,output="sos")
    return sosfiltfilt(sos,data,axis=0,padlen=min(4000,len(data)-1))


def sphere_directions(count=12):
    if type(count) is not int or count < 4:
        raise ValueError("At least four fixed sphere directions are required")
    k = np.arange(count)
    z = 1-2*(k+.5)/count
    phi = k*math.pi*(3-math.sqrt(5))
    r = np.sqrt(1-z*z)
    return np.column_stack([r*np.sin(phi),r*np.cos(phi),z])


def direction(azimuth, elevation):
    az, el = np.deg2rad([azimuth,elevation])
    return np.array([np.sin(az)*np.cos(el),np.cos(az)*np.cos(el),np.sin(el)])


def conditioned_source_signals(noise, count, rng):
    if type(count) is not int or count < 4:
        raise ValueError("At least four conditional source signals are required")
    n = np.asarray(noise,dtype=np.float64)
    if n.ndim != 1 or n.size < 512 or not np.isfinite(n).all():
        raise ValueError("Expected finite mono noise with >=512 samples")
    magnitudes = np.abs(np.fft.rfft(n))
    phase = rng.uniform(-np.pi,np.pi,(count,len(magnitudes)))
    phase[:,0] = 0
    if len(n)%2 == 0:
        phase[:,-1] = rng.integers(0,2,count)*np.pi
    q = np.fft.irfft(magnitudes[None,:]*np.exp(1j*phase),n=len(n),axis=-1)
    return n[None,:]/count+(q-q.mean(axis=0,keepdims=True))/math.sqrt(count)


def noise_fields(noise, positions, target_direction, seed, *, count=12, speed=343.0):
    n = np.asarray(noise,dtype=np.float64)
    analytic_reference = np.pad(n,(GUARD_SAMPLES,GUARD_SAMPLES))
    kinds = {"original_point":direction(-60,-15),"opposite_azimuth_point":direction(60,-15),
             "target_direction_point":np.asarray(target_direction,dtype=np.float64)}
    fields = {name:render_plane_wave(n,geometric_delays(positions,u,speed_of_sound=speed)) for name,u in kinds.items()}
    sources = conditioned_source_signals(n,count,np.random.default_rng(seed))
    field = np.zeros_like(next(iter(fields.values())))
    for source,u in zip(sources,sphere_directions(count)):
        field += render_plane_wave(source,geometric_delays(positions,u,speed_of_sound=speed))
    fields["conditioned_12wave"] = field
    errors = {}
    for name, data in fields.items():
        errors[name] = float(np.max(np.abs(data[:,4]-analytic_reference)))
        if errors[name] > 1e-10:
            raise ValueError("Noise field violates analytic fixed-reference constraint")
        data[:,4] = analytic_reference
    return fields, {"reference_roundoff_errors_before_correction":errors,
                    "conditional_source_sum_max_abs_error":float(np.max(np.abs(sources.sum(axis=0)-n))),
                    "directions":sphere_directions(count).tolist()}


def spatial_coherence(field, sample_rate=16000):
    from scipy.signal import stft
    data = np.asarray(field,dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 6 or not np.isfinite(data).all():
        raise ValueError("Expected finite [samples,6] field")
    f,_,z = stft(data.T,fs=sample_rate,nperseg=512,noverlap=256,boundary=None,padded=False)
    power = np.mean(np.abs(z)**2,axis=-1)
    band = (f>=300)&(f<=3500)
    weights = power[4,band]
    values = []
    for channel in (0,1,2,3,5):
        cross = np.mean(z[channel]*z[4].conj(),axis=-1)
        coherence = np.abs(cross)**2/np.maximum(power[channel]*power[4],1e-30)
        values.append(float(np.sum(coherence[band]*weights)/max(float(weights.sum()),1e-30)))
    return {"reference_pair_msc_300_3500hz":values,"mean":float(np.mean(values)),
            "definition":"Welch/STFT512 hop256 mean-time cross spectrum squared / powers, weighted by reference PSD; diagnostic only"}
