import hashlib,json,sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import g8_common as common
ROOT=common.ROOT;OUT=ROOT/'artifacts/g11_components_v1';G10=ROOT/'artifacts/g10_fresh_voice_v1';BASE=ROOT/'data/g10_v1'
now=common.now;sha=common.sha;read=common.read;dump=common.dump
from mcse.components import pcm24_field,decompose_fixed_mvdr
from mcse.simulation import render_plane_wave,geometric_delays
from mcse.spatial_noise import direction


def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})


def archive(stage,report,files):
    prior=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=prior


class Evaluator:
    def __init__(self):
        self.plan=json.loads((OUT/'frozen_plan.json').read_text());self.geometry=json.loads((ROOT/'configs/g1_geometry.json').read_text())
        self.sources={r['source_id']:r for r in read(BASE/'sources/speech_manifest.jsonl')}
        self.wi={(r['system'],r['sample_id']):r for r in read(G10/'evaluation/Whisper/asr_scores.jsonl')}
        self.ci={(r['system'],r['sample_id']):r for r in read(G10/'evaluation/CTC/asr_scores.jsonl')}
        self.models={r['system']:r for r in json.loads((G10/'systems.json').read_text())['beam']}
        self.verified={};torch.set_num_threads(2)

    def verify(self,path,expected):
        if path in self.verified:
            if self.verified[path]!=expected:raise ValueError('Conflicting expected hash')
        else:
            if sha(path)!=expected:raise ValueError('File hash mismatch:'+str(path))
            self.verified[path]=expected

    @lru_cache(maxsize=80)
    def clean(self,source_id,target,scale):
        row=self.sources[source_id];path=BASE/'sources'/row['path'];self.verify(str(path),row['sha256'])
        a,sr=sf.read(path,dtype='float64');assert sr==16000
        delays=geometric_delays(self.geometry['microphone_positions_m'],direction(target,10))
        return pcm24_field(render_plane_wave(a,delays)*scale)

    def evaluate(self,r):
        for p,h in ((r['noisy_path'],r['input_sha256']),(r['reference_path'],r['reference_sha256']),
          (r['mask_estimate_path'],r['mask_estimate_sha256']),(r['output_path'],r['output_sha256'])):self.verify(p,h)
        x,sr=sf.read(r['noisy_path'],dtype='float32',always_2d=True);reference,rr=sf.read(r['reference_path'],dtype='float32')
        estimate,er=sf.read(r['mask_estimate_path'],dtype='float32');saved,yr=sf.read(r['output_path'],dtype='float32');assert sr==rr==er==yr==16000
        s=self.clean(r['source_id'],r['target_azimuth'],r['common_scale']);referror=float(np.max(np.abs(s[:,4].astype(float)-reference)))
        if not np.array_equal(s[:,4],reference) or referror>self.plan['reference_match_max_abs']:raise ValueError('Reference reconstruction not exact')
        y,ys,yn,metrics=decompose_fixed_mvdr(x,estimate,s,self.plan['power_floor']);error=float(np.max(np.abs(y.astype(float)-saved)))
        if error>self.plan['saved_beam_match_max_abs']:raise ValueError('Frozen beam output mismatch')
        linearity_tolerance=1e-6*(1+float(np.abs(y).max()))
        if metrics['linearity_max_abs']>linearity_tolerance or metrics['snr_accounting_error_db']>1e-8:raise ValueError('Component linearity/energy accounting failure')
        model=self.models[r['system']];name='original' if model['mask_system']=='mcnet' else model['geometry'];key=(r['system'],r['sample_id'])
        fields={k:r[k] for k in ('system','sample_id','source_id','speaker_id','condition','noise_id','relative_angle','target_azimuth','snr_db','common_scale')}
        fields.update(group=name,seed=model.get('seed'),metrics=metrics,reference_reconstruction_max_abs=referror,reproduced_beam_max_abs=error,
          linearity_tolerance=linearity_tolerance,source_record_sha256=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest(),
          speech_reference_samples_sha256=hashlib.sha256(s[:,4].tobytes()).hexdigest(),status='ok',
          original_signal_metrics=r['metrics'],Whisper={k:self.wi[key]['score'][k] for k in ('N','S','D','I','correct','errors','wer')},CTC={k:self.ci[key]['score'][k] for k in ('N','S','D','I','correct','errors','wer')})
        return fields


def selected_rows():
    rows=read(G10/'evaluation/combined_signals.jsonl');chosen=[r for r in rows if r['system'].endswith('_mask_mvdr') and r['condition']!='clean']
    assert len(chosen)==4032 and len({r['system'] for r in chosen})==7
    return sorted(chosen,key=lambda r:(r['system'],r['sample_id']))
