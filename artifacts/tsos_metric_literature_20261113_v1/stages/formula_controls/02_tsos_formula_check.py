"""Algebraic controls for a published suppression formula, not a metric replica."""
import sys
import numpy as np
import g8_common as common


def main():
    folder=common.ROOT/'artifacts/tsos_metric_literature_20261113_v1'
    folder.mkdir(exist_ok=False)
    p=.3;gamma=.1;attenuation=.2
    def ratio(clean,estimate):return float(np.maximum(clean**p-estimate**p,0.)**2/clean**p)
    controls=[]
    for amplitude in (.1,1.):
        r=ratio(amplitude,amplitude*attenuation)
        controls.append({'clean_magnitude':amplitude,'estimate_magnitude':amplitude*attenuation,
                         'constant_relative_gain':attenuation,'suppression_ratio':r,'flag':r>gamma})
    scaling_error=abs(controls[0]['suppression_ratio']/controls[1]['suppression_ratio']-.1**p)
    assert scaling_error<1e-14 and controls[0]['flag'] is False and controls[1]['flag'] is True
    masked=ratio(1.,.2+.8);assert masked==0
    result={'status':'ok','p':p,'gamma':gamma,'controls':controls,'scaling_law_error':scaling_error,
            'one_bin_coherent_noise_example':{'clean':1.,'target_component':.2,'noise_component':.8,
                'mixture_magnitude':1.,'suppression_ratio':masked,'flag':False},
            'scope':'Single-bin algebra with clean C and estimate E, no input/STFT amplitude normalization. Not speech experiments or an official TSOS implementation.',
            'model_data_or_G22_outputs_changed':False}
    common.dump(folder/'formula_controls.json',result,True)
    report=common.ROOT/'docs/研究补查_TSOS来源与尺度控制_20261113.md'
    previous=common.OUT
    try:
        common.OUT=folder
        common.archive('formula_controls',report,[folder/'formula_controls.json',common.ROOT/'scripts/tsos_formula_check.py'])
    finally:common.OUT=previous
    print('Formula controls passed; no speech/model computations performed')


if __name__=='__main__':main()
