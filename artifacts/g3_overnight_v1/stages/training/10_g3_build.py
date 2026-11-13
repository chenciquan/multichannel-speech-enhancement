#!/usr/bin/env python3
"""Freeze isolated sources and generate a fixed reusable train/validation/test pool."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.simulation import render_plane_wave,geometric_delays,speech_activity_mask,mix_at_snr
from mcse.spatial_noise import highpass40,direction
PLAN=json.loads((ROOT/'configs/g3_overnight_plan.json').read_text())
BASE=ROOT/'data/g3_v1/dataset';SOURCES=ROOT/'data/g3_v1/sources'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    if BASE.exists():raise RuntimeError('Dataset exists; refusing overwrite')
    if json.loads((SOURCES/'status.json').read_text())['status']!='ok':raise RuntimeError('Source collection incomplete')
    source=[json.loads(t) for t in (SOURCES/'speech_manifest.jsonl').read_text().splitlines()]
    prior=[]
    for rel in ['data/sources/g1_v1/source_manifest.jsonl','data/sources/g2_confirmation_v1/source_manifest.jsonl']:
        prior.extend(json.loads(t) for t in (ROOT/rel).read_text().splitlines())
    assert not {r['source_id'] for r in source}&{r['source_id'] for r in prior}
    assert not {r['sha256'] for r in source}&{r['sha256'] for r in prior}
    roles={split:[r for r in source if r['split']==split] for split in ('train','validation','test')}
    speaker_sets={k:{r['speaker_id'] for r in v} for k,v in roles.items()}
    assert all(not speaker_sets[a]&speaker_sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    assert not speaker_sets['test']&{r['speaker_id'] for r in prior}
    speech={}
    for r in source:
        p=SOURCES/r['path'];assert sha(p)==r['sha256'];a,sr=sf.read(p,dtype='float64');assert sr==16000;speech[r['source_id']]=a
    noise={};noise_rows=json.loads((SOURCES/'noise_manifest.json').read_text())
    for r in noise_rows:
        assert sha(r['path'])==r['sha256'];a,sr=sf.read(r['path'],dtype='float64');assert sr==16000
        noise[r['noise_id']]=highpass40(a)
    geometry=json.loads((ROOT/'configs/g1_geometry.json').read_text());positions=geometry['microphone_positions_m'];target=geometry['target_direction_unit']
    BASE.mkdir(parents=True)
    dump(BASE/'frozen_protocol.json',{'plan':PLAN,'source_manifest_sha256':sha(SOURCES/'speech_manifest.jsonl'),
        'noise_manifest_sha256':sha(SOURCES/'noise_manifest.json'),'speaker_sets':{k:sorted(v) for k,v in speaker_sets.items()},
        'excluded_diagnostic_sources':sorted(r['source_id'] for r in prior),'test_tuning_forbidden':True,
        'training_field':'unconditioned independent segments from separate temporal blocks; same physical geometry',
        'generator_sha256':sha(Path(__file__)),'noise_time_partitions':PLAN['noise_time_partitions_seconds']})
    def make(r,rng,split,noise_id,field,angle,snr,crop=False):
        mono=speech[r['source_id']];start=0
        if crop:
            for _ in range(30):
                start=int(rng.integers(0,max(1,len(mono)-16000+1)));piece=mono[start:start+16000]
                if np.sqrt(np.mean(piece**2))>1e-4:break
            mono=piece
        clean=render_plane_wave(mono,geometric_delays(positions,target));activity=speech_activity_mask(clean[:,4])
        lo,hi=PLAN['noise_time_partitions_seconds'][split];k=1 if field=='point' else 8;nfield=np.zeros_like(clean);intervals=[]
        for i in range(k):
            lower=int((lo+(hi-lo)*i/k)*16000);upper=int((lo+(hi-lo)*(i+1)/k)*16000)-len(mono)
            offset=int(rng.integers(lower,upper+1));clip=noise[noise_id][offset:offset+len(mono)]
            u=direction(angle,-15) if k==1 else rng.normal(size=3)
            u=u/np.linalg.norm(u);nfield+=render_plane_wave(clip,geometric_delays(positions,u))/np.sqrt(k)
            intervals.append([offset,offset+len(mono)])
        mixture,_,gain=mix_at_snr(clean,nfield,float(snr),activity)
        scale=min(1.,.95/max(float(np.abs(mixture).max()),float(np.abs(clean).max())))
        return (mixture*scale).astype('float32'),(clean*scale).astype('float32'),{'source_crop_start':start,'noise_intervals':intervals,'noise_gain':gain,'common_scale':scale}
    n=PLAN['train_examples'];frames=16128
    xmap=np.lib.format.open_memmap(BASE/'train_x.npy',mode='w+',dtype='float32',shape=(n,6,frames))
    ymap=np.lib.format.open_memmap(BASE/'train_y.npy',mode='w+',dtype='float32',shape=(n,frames))
    cleanmap=np.lib.format.open_memmap(BASE/'clean_x.npy',mode='w+',dtype='float32',shape=(128,6,frames))
    rng=np.random.default_rng(2026111201);catalog=[]
    for i in range(n):
        r=roles['train'][int(rng.integers(len(roles['train'])))];field='point' if i<n//4 else 'multi8'
        angle=int(rng.choice(PLAN['train_angles']));noise_id=str(rng.choice(PLAN['noise_train']));snr=int(rng.choice(PLAN['snr_db']))
        x,clean,meta=make(r,rng,'train',noise_id,field,angle,snr,True)
        xmap[i]=x.T;ymap[i]=clean[:,4]
        if i<128:cleanmap[i]=clean.T
        catalog.append({'index':i,'source_id':r['source_id'],'speaker_id':r['speaker_id'],'field':field,'angle':angle,'noise_id':noise_id,'snr_db':snr,**meta})
        if (i+1)%256==0:print(json.dumps({'stage':'train_pool','examples':i+1,'total':n}),flush=True)
    xmap.flush();ymap.flush();cleanmap.flush();del xmap,ymap,cleanmap
    dump(BASE/'train_catalog.json',catalog)
    for split in ('validation','test'):
        folder=BASE/split;folder.mkdir();(folder/'noisy').mkdir();(folder/'reference').mkdir();manifest=[]
        rng=np.random.default_rng(2026111202 if split=='validation' else 2026111203)
        for r in roles[split]:
            if split=='validation':conditions=[('OOFFICE','point',-60,0),('OOFFICE','multi8',60,0),('DKITCHEN','point',60,-5),('DKITCHEN','multi8',-60,-5)]
            else:
                conditions=[(noise_id,field,(-60 if ci%2==0 else 45),snr) for ci,(noise_id,field,snr) in enumerate(
                    (a,b,c) for a in PLAN['noise_test'] for b in ('point','multi8') for c in PLAN['snr_db'])]
            for noise_id,field,angle,snr in conditions:
                x,clean,meta=make(r,rng,split,noise_id,field,angle,snr)
                sid=f'{split}_{len(manifest):04d}_{r["source_id"]}';xp=folder/'noisy'/(sid+'.wav');yp=folder/'reference'/(sid+'.wav')
                sf.write(xp,x,16000,subtype='PCM_24');sf.write(yp,clean[:,4],16000,subtype='PCM_24')
                manifest.append({'sample_id':sid,'source_id':r['source_id'],'speaker_id':r['speaker_id'],'split':split,
                    'noisy_path':'noisy/'+xp.name,'reference_path':'reference/'+yp.name,'transcript':r['transcript'],
                    'noise_id':noise_id,'snr_db':snr,'sample_rate':16000,'ref_channel':4,'noise_field':field,'noise_azimuth':angle,
                    'noisy_sha256':sha(xp),'reference_sha256':sha(yp),**meta})
        with (folder/'manifest.jsonl').open('x',encoding='utf-8') as f:
            for r in manifest:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(json.dumps({'stage':split,'mixtures':len(manifest),'sources':len(roles[split])}),flush=True)
    dump(BASE/'BUILD_COMPLETE.json',{'train_examples':n,'train_seconds':sum(r['duration_seconds'] for r in roles['train']),
        'validation_sources':len(roles['validation']),'test_sources':len(roles['test']),'heldout_test_noise':'PCAFETER',
        'test_speaker_disjoint':True,'all_previous32_source_ids_and_hashes_excluded':True,'finished_at':time.time()})

if __name__=='__main__':main()
