#!/usr/bin/env python3
"""Audit raw and mixed noise spectra before choosing any enhancement method."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
BANDS=[(0,.1),(.1,20),(20,40),(40,80),(80,100),(100,300),(300,1000),(1000,3500),(3500,8001)]


def spectrum_summary(signal):
    x=np.asarray(signal,dtype=np.float64)
    f=np.fft.rfftfreq(len(x),1/16000)
    energy=np.abs(np.fft.rfft(x))**2
    if len(x)%2==0:energy[1:-1]*=2
    else:energy[1:]*=2
    fractions={f'{lo}-{hi}Hz':float(energy[(f>=lo)&(f<hi)].sum()/energy.sum()) for lo,hi in BANDS}
    return {'band_energy_fractions':fractions,'nonDC_peak_frequency_hz':float(f[np.argmax(energy[1:])+1]),
            'rms':float(np.sqrt(np.mean(x*x))),'mean':float(x.mean()),
            'energy_below20hz_fraction':float(energy[f<20].sum()/energy.sum()),
            'energy_below40hz_fraction':float(energy[f<40].sum()/energy.sum()),
            'energy_below100hz_fraction':float(energy[f<100].sum()/energy.sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Refusing to overwrite spectrum evidence')
    noise_rows=[json.loads(t) for t in (ROOT/'data/sources/g1_v1/noise_manifest.jsonl').read_text().splitlines()]
    originals=[]
    for r in noise_rows:
        path=ROOT/'data/sources/g1_v1'/r['path']
        data,sr=sf.read(path)
        if sr!=16000:raise ValueError('No resampling')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=r['sha256']:raise ValueError('Original noise changed')
        originals.append({'noise_id':r['noise_id'],'source_sha256':digest,**spectrum_summary(data)})
    mixed=[]
    for r in map(json.loads,(ROOT/'artifacts/g1_v1/mcnet/cases.jsonl').read_text().splitlines()):
        s,_=sf.read(r['reference_path']);x,_=sf.read(r['noisy_path']);n=x[:,4]-s
        mixed.append({'sample_id':r['sample_id'],'source_id':r['source_id'],'noise_id':r['noise_id'],
                      'speech':spectrum_summary(s),'noise':spectrum_summary(n)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump({'raw_recordings':originals,'all_g1_mixtures':mixed,
                   'definition':'Whole-signal one-sided FFT Parseval energy; no filter/normalization applied',
                   'notice':'Spectral energy concentration is measured; recording-device origin is not established'},stream,ensure_ascii=False,indent=2)
    print(json.dumps({'raw_recordings':originals,'g1_mixtures':len(mixed)},ensure_ascii=False))


if __name__=='__main__':main()
