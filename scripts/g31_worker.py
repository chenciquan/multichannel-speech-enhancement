import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from functools import lru_cache
import numpy as np
import soundfile as sf
import torch
from g31_common import OUT,DATA,sha
from mcse.evaluation import MetricEvaluator,waveform_diagnostics
from mcse.beamforming import mask_mvdr
from mcse.wiener import mask_mwf

_evaluator=None


def initialize():torch.set_num_threads(1);torch.set_num_interop_threads(1)


@lru_cache(maxsize=1)
def inputs():return np.load(DATA/'validation_x.npy',mmap_mode='r')


def save_and_score(job):
    global _evaluator
    r,y=job;y=np.asarray(y);assert y.dtype==np.float32 and y.shape==(23679,) and np.isfinite(y).all()
    path=OUT/'audio'/r['system']/(r['sample_id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
    saved,sr=sf.read(path,dtype='float32');assert sr==16000 and np.array_equal(saved,y)
    assert sha(r['reference_path'])==r['reference_sha256']
    reference,rr=sf.read(r['reference_path'],dtype='float32');assert rr==16000
    if _evaluator is None:_evaluator=MetricEvaluator()
    metrics=_evaluator.evaluate(reference,saved)
    assert all(metrics['status'][k]['status']=='ok' for k in ('si_sdr_db','pesq_wb','stoi'))
    result={**r,'output_path':str(path),'output_sha256':sha(path),'metrics':metrics,
            'waveform_diagnostics':waveform_diagnostics(reference,saved),'status':'ok'}
    if r['condition']=='clean':
        s=reference.astype(float);out=saved.astype(float);energy=float(s@s);assert energy>0
        result['clean_levels']={'rms_gain_ratio':float(np.sqrt((out@out)/energy)),
            'reference_normalized_squared_error':float((out-s)@(out-s)/energy)}
    return result


def filtered(job):
    r,method=job;assert not torch.cuda.is_initialized()
    x=np.array(inputs()[r['index']].T,copy=True);assert sha(r['output_path'])==r['output_sha256']
    e,sr=sf.read(r['output_path'],dtype='float32');assert sr==16000
    y,info=(mask_mvdr(x,e) if method=='mvdr' else mask_mwf(x,e,mu=1.))
    case={k:v for k,v in r.items() if k not in ('output_path','output_sha256','metrics','waveform_diagnostics','clean_levels','status')}
    case.update(system=r['system']+'_mask_'+method,source_system=r['system'],method=method,
        mask_estimate_path=r['output_path'],mask_estimate_sha256=r['output_sha256'],filter_diagnostics=info)
    return save_and_score((case,y))
