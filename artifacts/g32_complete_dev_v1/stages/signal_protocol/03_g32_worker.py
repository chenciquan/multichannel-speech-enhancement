import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import soundfile as sf
import torch
from g32_common import OUT,sha
from mcse.evaluation import MetricEvaluator,waveform_diagnostics
from mcse.beamforming import mask_mvdr
from mcse.wiener import mask_mwf

_evaluator=None


def initialize():torch.set_num_threads(1);torch.set_num_interop_threads(1)


def save_and_score(job):
    global _evaluator
    r,y=job;y=np.asarray(y);assert y.dtype==np.float32 and y.shape==(r['frames'],) and np.isfinite(y).all()
    path=OUT/'audio'/r['system']/(r['sample_id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
    saved,sr=sf.read(path,dtype='float32');assert sr==16000 and np.array_equal(saved,y)
    assert sha(r['reference_path'])==r['reference_sha256'];ref,rr=sf.read(r['reference_path'],dtype='float32');assert rr==16000
    if _evaluator is None:_evaluator=MetricEvaluator()
    metrics=_evaluator.evaluate(ref,saved);assert all(metrics['status'][m]['status']=='ok' for m in ('si_sdr_db','pesq_wb','stoi'))
    row={**r,'output_path':str(path),'output_sha256':sha(path),'metrics':metrics,'status':'ok','waveform_diagnostics':waveform_diagnostics(ref,saved)}
    if r['condition']=='clean':
        a=ref.astype(float);b=saved.astype(float);energy=float(a@a)
        row['clean_levels']={'rms_gain_ratio':float(np.sqrt((b@b)/energy)),
            'reference_normalized_squared_error':float((b-a)@(b-a)/energy)}
    return row


def filtered(job):
    r,method=job;assert not torch.cuda.is_initialized()
    assert sha(r['noisy_path'])==r['input_sha256'] and sha(r['output_path'])==r['output_sha256']
    x,sr=sf.read(r['noisy_path'],dtype='float32');e,rr=sf.read(r['output_path'],dtype='float32');assert sr==rr==16000
    y,info=(mask_mvdr(x,e) if method=='mvdr' else mask_mwf(x,e,mu=1.))
    row={k:v for k,v in r.items() if k not in ('output_path','output_sha256','metrics','status','waveform_diagnostics','clean_levels')}
    row.update(system=r['system']+'_mask_'+method,source_system=r['system'],method=method,filter_diagnostics=info,
        mask_estimate_path=r['output_path'],mask_estimate_sha256=r['output_sha256'])
    return save_and_score((row,y))
