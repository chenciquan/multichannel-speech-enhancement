#!/usr/bin/env python3
"""Frozen-model spatial counterfactual dataset; never optimize any model."""
import hashlib,json,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.simulation import render_plane_wave,geometric_delays,speech_activity_mask,mix_at_snr
from mcse.spatial_noise import highpass40,direction
OUT=ROOT/'artifacts/g4_counterfactual_v1';BASE=ROOT/'data/g4_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def fault(x,kind):
    if kind=='gain':return x*10**(np.array([1.5,-1.5,.75,-.75,0,1.])/20)
    out=np.zeros_like(x)
    for ch,shift in enumerate([0,1,-1,2,0,-2]):
        if shift>0:out[shift:,ch]=x[:-shift,ch]
        elif shift<0:out[:shift,ch]=x[-shift:,ch]
        else:out[:,ch]=x[:,ch]
    return out
def main():
    g3=ROOT/'artifacts/g3_overnight_v1';assert (g3/'MAJOR_STAGE_COMPLETE.json').exists()
    assert json.loads((g3/'completion_audit.json').read_text())['status']=='ok'
    assert (g3/'stages/completion/archive_manifest.json').exists()
    OUT.mkdir(parents=True,exist_ok=False);BASE.mkdir(parents=True,exist_ok=False)
    source=read(ROOT/'data/g3_v1/sources/speech_manifest.jsonl');groups=defaultdict(list)
    for r in source:
        if r['split']=='test':groups[r['speaker_id']].append(r)
    for values in groups.values():values.sort(key=lambda r:r['source_id'])
    chosen=[];depth=0
    while len(chosen)<8:
        for speaker in sorted(groups):
            if depth<len(groups[speaker]):chosen.append(groups[speaker][depth])
            if len(chosen)==8:break
        depth+=1
    selected=json.loads((g3/'selected_checkpoints_extended.json').read_text())['systems']
    systems=[{'system':'noisy','kind':'noisy'},{'system':'mcnet','kind':'mcnet'}]
    for variant in ('ordinary','complete','without_stratification','without_clean_identity'):
        r=next(s for s in selected if s['variant']==variant and s['seed']==20261112)
        systems.append({'system':r['system'],'kind':'adapter','path':r['path'],'step':r['step'],'checkpoint_sha256':sha(ROOT/r['path'])})
    plan=json.loads((ROOT/'configs/g4_candidate_draft.json').read_text());plan.update(status='frozen_before_G4_outputs',frozen_utc=datetime.now(timezone.utc).isoformat(),
        previous_major_report_sha256=sha(ROOT/'docs/G3_五小时实验验收报告.md'),previous_completion_archive_sha256=sha(g3/'stages/completion/archive_manifest.json'),
        selected_source_ids=[r['source_id'] for r in chosen],systems=systems,seed=2026111401)
    dump(OUT/'frozen_protocol.json',plan);dump(ROOT/'configs/g4_counterfactual_protocol.json',plan)
    geom=json.loads((ROOT/'configs/g1_geometry.json').read_text());positions=geom['microphone_positions_m']
    noise={}
    for r in json.loads((ROOT/'data/g3_v1/sources/noise_manifest.json').read_text()):
        if r['noise_id'] in plan['noise_types']:
            a,sr=sf.read(r['path'],dtype='float64');assert sr==16000 and sha(r['path'])==r['sha256'];noise[r['noise_id']]=highpass40(a)
    for name in ('noisy','reference'):(BASE/name).mkdir()
    rng=np.random.default_rng(plan['seed']);manifest=[];counter=defaultdict(set)
    def save(r,x,s,condition,noise_id,snr,ta,na,scale,offset):
        sid=f'g4_{len(manifest):04d}_{r["source_id"]}';xp=BASE/'noisy'/(sid+'.wav');sp=BASE/'reference'/(sid+'.wav')
        sf.write(xp,x*scale,16000,subtype='PCM_24');sf.write(sp,s*scale,16000,subtype='PCM_24')
        saved,_=sf.read(xp,dtype='float32',always_2d=True);reference,_=sf.read(sp,dtype='float32')
        rawhash=hashlib.sha256(saved[:,4].tobytes()).hexdigest();rh=hashlib.sha256(reference.tobytes()).hexdigest()
        key=f'{r["source_id"]}|{noise_id}|{snr}';counter[key].add((rawhash,rh))
        manifest.append({'sample_id':sid,'source_id':r['source_id'],'speaker_id':r['speaker_id'],'transcript':r['transcript'],
          'split':'clean_test' if condition=='clean' else 'stress_test','condition':condition,'noise_id':noise_id,'snr_db':snr,
          'target_azimuth':ta,'noise_azimuth':na,'counterfactual_group':key,'noisy_path':str(xp),'reference_path':str(sp),
          'input_sha256':sha(xp),'reference_sha256':sha(sp),'raw_mic5_sample_sha256':rawhash,'reference_sample_sha256':rh,
          'common_scale':scale,'noise_offset_samples':offset,'sample_rate':16000,'ref_channel':4})
    for r in chosen:
        path=ROOT/'data/g3_v1/sources'/r['path'];assert sha(path)==r['sha256'];mono,sr=sf.read(path,dtype='float64');assert sr==16000
        speech={ta:render_plane_wave(mono,geometric_delays(positions,direction(ta,10))) for ta in (-30,20,60)}
        s=speech[20][:,4];activity=speech_activity_mask(s)
        for noise_id in plan['noise_types']:
            offset=int(rng.integers(180*16000,295*16000-len(mono)+1));clip=noise[noise_id][offset:offset+len(mono)]
            fields={na:render_plane_wave(clip,geometric_delays(positions,direction(na,-15))) for na in (-60,45)}
            for snr in (-5,0,5):
                variants=[]
                for ta in (-30,20,60):
                    for na in (-60,45):
                        mix,_,_=mix_at_snr(speech[ta],fields[na],snr,activity);variants.append((mix,'normal',ta,na))
                        if ta==20 and snr==0:
                            for kind in ('gain','latency'):variants.append((fault(mix,kind),kind,ta,na))
                peak=max([float(np.abs(x).max()) for x,_,_,_ in variants]+[float(np.abs(s).max())]);scale=min(1.,.95/peak)
                for x,condition,ta,na in variants:save(r,x,s,condition,noise_id,snr,ta,na,scale,offset)
        scale=min(1.,.95/max(float(np.abs(x).max()) for x in speech.values()))
        for ta,x in speech.items():save(r,x,s,'clean','none',None,ta,None,scale,None)
    assert len(manifest)==376 and all(len(v)==1 for v in counter.values()),'Counterfactual reference microphone differs'
    with (BASE/'manifest.jsonl').open('x') as f:
        for r in manifest:f.write(json.dumps(r)+'\n')
    counts={sp:sum(r['speaker_id']==sp for r in chosen) for sp in sorted(groups)}
    dump(OUT/'BUILD_COMPLETE.json',{'cases':376,'sources':8,'source_counts_per_speaker':counts,'canonical_raw_and_reference_hashes_per_group':{k:len(v) for k,v in counter.items()},
       'all_counterfactual_mic5_and_reference_samples_identical':True,'manifest_sha256':sha(BASE/'manifest.jsonl')})
    print(json.dumps({'stage':'G4_build','cases':len(manifest),'speakers':counts,'mic5_counterfactual_equality':True}))
if __name__=='__main__':main()
