"""Lossless, bounded ZIP parts for existing research files; no model execution."""
import hashlib,json,os,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.cache/github_upload_20261113'
LIMIT=1024**3

def dump(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'archive_complete.json').exists() or list(OUT.glob('research-part-*.zip')):
        raise SystemExit('Existing snapshot parts; refusing overwrite')
    files=[]
    for name in ('data','artifacts','third_party'):
        for folder,dirs,names in os.walk(ROOT/name):
            dirs[:]=sorted(d for d in dirs if d!='__pycache__')
            for name in sorted(names):
                p=Path(folder)/name
                if p.suffix in ('.pyc','.pyo'):continue
                if p.is_symlink():raise RuntimeError('Unexpected symlink; review before archival: '+str(p))
                files.append(p)
    groups=[];current=[];size=0
    for p in files:
        n=p.stat().st_size
        if n>=2*1024**3:raise RuntimeError('File needs separate split protocol: '+str(p))
        if current and size+n>LIMIT:groups.append(current);current=[];size=0
        current.append(p);size+=n
    if current:groups.append(current)
    started=time.monotonic();inventory=[];parts=[]
    dump(OUT/'progress.json',{'status':'packing','total_files':len(files),'total_parts':len(groups),'completed_parts':0})
    for i,group in enumerate(groups,1):
        archive=OUT/f'research-part-{i:03d}-of-{len(groups):03d}.zip'
        entries=[]
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for p in group:
                before=p.stat();relative=p.relative_to(ROOT).as_posix();h=hashlib.sha256()
                info=zipfile.ZipInfo.from_file(p,arcname=relative)
                info.compress_type=zipfile.ZIP_DEFLATED;info._compresslevel=1
                with p.open('rb') as source,z.open(info,'w',force_zip64=True) as target:
                    for block in iter(lambda:source.read(1024**2),b''):
                        h.update(block);target.write(block)
                after=p.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('File changed during snapshot: '+relative)
                entries.append({'path':relative,'bytes':before.st_size,'sha256':h.hexdigest(),'archive':archive.name})
        h=hashlib.sha256()
        with archive.open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        with zipfile.ZipFile(archive) as z:
            assert len(z.infolist())==len(entries)
            assert sum(r.file_size for r in z.infolist())==sum(r['bytes'] for r in entries)
        part={'file':archive.name,'bytes':archive.stat().st_size,'sha256':h.hexdigest(),'files':len(entries)}
        assert part['bytes']<2*1024**3
        inventory.extend(entries);parts.append(part)
        dump(OUT/'inventory_in_progress.json',inventory)
        dump(OUT/'progress.json',{'status':'packing','total_files':len(files),'total_parts':len(groups),'completed_parts':i,
            'completed_files':len(inventory),'archive_bytes':sum(r['bytes'] for r in parts),'elapsed_seconds':time.monotonic()-started})
        print(json.dumps(part),flush=True)
    manifest={'version':1,'date_local':'2026-11-13','repository':'chenciquan/multichannel-speech-enhancement',
        'scope':['data/','artifacts/','third_party/'],'excludes':['__pycache__/','*.pyc','*.pyo'],
        'environment_and_cache_not_project_backup':['.venv/','.cache/','.tools/'],
        'files':inventory,'parts':parts,'raw_bytes':sum(r['bytes'] for r in inventory),
        'archive_bytes':sum(r['bytes'] for r in parts),'elapsed_seconds':time.monotonic()-started}
    dump(OUT/'research-file-manifest.json',manifest)
    with (OUT/'SHA256SUMS.txt').open('x') as f:
        for part in parts:f.write(part['sha256']+'  '+part['file']+'\n')
    dump(OUT/'archive_complete.json',{'status':'complete','files':len(inventory),'parts':len(parts),
        'raw_bytes':manifest['raw_bytes'],'archive_bytes':manifest['archive_bytes'],'elapsed_seconds':manifest['elapsed_seconds']})
    print('archive complete',flush=True)

if __name__=='__main__':main()
