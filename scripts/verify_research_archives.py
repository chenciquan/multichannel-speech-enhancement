"""Read every ZIP member and verify the lossless snapshot manifest; no extraction."""
import hashlib,json,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'.cache/github_upload_20261113'
def main():
    start=time.monotonic();manifest=json.loads((CACHE/'research-file-manifest.json').read_text(encoding='utf-8'))
    expected={r['path']:r for r in manifest['files']};seen=set();results=[]
    assert len(expected)==len(manifest['files'])
    for part in manifest['parts']:
        path=CACHE/part['file'];h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024**2),b''):h.update(b)
        assert h.hexdigest()==part['sha256'] and path.stat().st_size==part['bytes']
        count=0
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                assert entry.filename in expected and entry.filename not in seen
                row=expected[entry.filename];assert row['archive']==part['file'] and entry.file_size==row['bytes']
                h=hashlib.sha256()
                with archive.open(entry) as f:
                    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
                assert h.hexdigest()==row['sha256'],entry.filename
                seen.add(entry.filename);count+=1
        results.append({'part':part['file'],'files_verified':count,'passed':True})
        print(json.dumps(results[-1]),flush=True)
    assert seen==set(expected)
    output={'status':'passed','files':len(seen),'parts':len(results),'all_CRC_and_SHA256_verified':True,
        'seconds':time.monotonic()-start,'part_results':results}
    with (CACHE/'archive_validation.json').open('x',encoding='utf-8') as f:json.dump(output,f,indent=2)
    print('All archived files verified',flush=True)
if __name__=='__main__':main()
