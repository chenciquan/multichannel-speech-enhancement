#!/usr/bin/env python3
"""Read-only verification of actual saved paired arrays, independent of renderer."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from g4_build import sha,dump
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'data/g5_v1';OUT=ROOT/'artifacts/g5_rotation_v1/paired_data_audit.json'
def main():
    if OUT.exists():raise RuntimeError('Readback evidence exists')
    arrays={g:{n:np.load(BASE/g/(n+'.npy'),mmap_mode='r') for n in ('train_x','train_y','clean_x')} for g in ('fixed','rotated')}
    fixed,rotated=arrays['fixed'],arrays['rotated'];checks={
      'shapes':all(a['train_x'].shape==(2048,6,16128) and a['train_y'].shape==(2048,16128) and a['clean_x'].shape==(128,6,16128) for a in arrays.values()),
      'noisy_reference_mic_equal':True,'clean_supervision_equal':True,'clean_bank_reference_equal':bool(np.array_equal(fixed['clean_x'][:,4,:],rotated['clean_x'][:,4,:]))}
    for lo in range(0,2048,64):
        checks['noisy_reference_mic_equal'] &= np.array_equal(fixed['train_x'][lo:lo+64,4,:],rotated['train_x'][lo:lo+64,4,:])
        checks['clean_supervision_equal'] &= np.array_equal(fixed['train_y'][lo:lo+64],rotated['train_y'][lo:lo+64])
    hashes={g:{n:sha(BASE/g/(n+'.npy')) for n in a} for g,a in arrays.items()}
    result={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'array_sha256':hashes,'readback_at_utc':datetime.now(timezone.utc).isoformat(),
       'role':'Read-only saved-array verification during training; not a model or confirmation-data selection step'}
    dump(OUT,result);print(json.dumps({'status':result['status'],'checks':checks}))
    if result['status']!='ok':raise RuntimeError('Paired data invariant failed')
if __name__=='__main__':main()
