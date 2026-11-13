"""Lossless 256-MiB transport segments and safe reconstruction of ZIP archives."""
import argparse,hashlib,json,time
from pathlib import Path
CHUNK=256*1024**2
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def prepare(cache,verified):
    path=cache/'release-transport-manifest.json'
    if path.exists():
        transport=json.loads(path.read_text(encoding='utf-8'))
    else:
        source=json.loads((cache/'research-file-manifest.json').read_text(encoding='utf-8'))
        folder=cache/'transport_chunks';folder.mkdir(exist_ok=True)
        parts=[]
        for part in source['parts']:
            name=part['file'];original=cache/name;assets=[]
            existing=verified.get(name)
            if existing:
                assert existing['size']==part['bytes'] and existing['digest']=='sha256:'+part['sha256']
                assets=[{'name':name,'bytes':part['bytes'],'sha256':part['sha256'],'relative_local_path':name}]
                mode='whole_zip'
            else:
                count=(part['bytes']+CHUNK-1)//CHUNK;full=hashlib.sha256()
                with original.open('rb') as f:
                    for i in range(1,count+1):
                        segment_name=f'{name}.chunk-{i:03d}-of-{count:03d}';dest=folder/segment_name
                        h=hashlib.sha256();n=0
                        if dest.exists():
                            # Reuse a fully written segment, never overwrite it.
                            expected_size=min(CHUNK,part['bytes']-(i-1)*CHUNK)
                            assert dest.stat().st_size==expected_size
                            with dest.open('rb') as saved:
                                while n<expected_size:
                                    a=f.read(min(1024**2,expected_size-n));b=saved.read(len(a))
                                    assert a==b and a;h.update(a);full.update(a);n+=len(a)
                        else:
                            with dest.open('xb') as out:
                                while n<CHUNK:
                                    b=f.read(min(1024**2,CHUNK-n))
                                    if not b:break
                                    out.write(b);h.update(b);full.update(b);n+=len(b)
                        assets.append({'name':segment_name,'bytes':n,'sha256':h.hexdigest(),'relative_local_path':'transport_chunks/'+segment_name})
                assert full.hexdigest()==part['sha256'] and sum(a['bytes'] for a in assets)==part['bytes']
                mode='split_zip'
            parts.append({'archive':name,'archive_bytes':part['bytes'],'archive_sha256':part['sha256'],'mode':mode,'assets':assets})
            print('Prepared transport '+name+' '+mode,flush=True)
        transport={'version':2,'segment_max_bytes':CHUNK,'original_archives':len(parts),
            'archive_bytes':source['archive_bytes'],'parts':parts,'restore':'python scripts/release_transport.py --restore DOWNLOADED_ASSET_FOLDER'}
        assert sum(p['archive_bytes'] for p in parts)==source['archive_bytes']
        with path.open('x',encoding='utf-8') as f:json.dump(transport,f,ensure_ascii=False,indent=2)
    payloads=[]
    for p in transport['parts']:
        for a in p['assets']:
            local=cache/a['relative_local_path']
            assert local.exists() and local.stat().st_size==a['bytes']
            payloads.append((local,a['sha256']))
    return transport,payloads

def restore(folder,only=None):
    folder=Path(folder).resolve();manifest=json.loads((folder/'release-transport-manifest.json').read_text(encoding='utf-8'))
    count=0
    for part in manifest['parts']:
        if only and part['archive']!=only:continue
        archive=folder/part['archive']
        assert archive.parent==folder,'Unsafe archive path'
        if archive.exists():
            assert archive.stat().st_size==part['archive_bytes'] and sha(archive)==part['archive_sha256']
            count+=1;continue
        assert part['mode']=='split_zip','Whole ZIP asset missing: '+part['archive']
        full=hashlib.sha256()
        with archive.open('xb') as out:
            for asset in part['assets']:
                source=folder/asset['name']
                if not source.exists():source=folder/asset['relative_local_path']
                source=source.resolve()
                assert source.is_relative_to(folder) and source.stat().st_size==asset['bytes']
                h=hashlib.sha256()
                with source.open('rb') as f:
                    for b in iter(lambda:f.read(1024**2),b''):h.update(b);full.update(b);out.write(b)
                assert h.hexdigest()==asset['sha256'],'Segment differs: '+asset['name']
        assert archive.stat().st_size==part['archive_bytes'] and full.hexdigest()==part['archive_sha256']
        count+=1;print('Restored '+archive.name,flush=True)
    return count
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--restore',required=True);args=p.parse_args()
    print('Verified ZIP archives:',restore(args.restore))
