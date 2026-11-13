"""Three streaming uploads; resumes verified assets, never stores credentials."""
import argparse,concurrent.futures,hashlib,json,subprocess,threading,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'.cache/github_upload_20261113'
API='https://api.github.com/repos/chenciquan/multichannel-speech-enhancement'
LOCK=threading.Lock();TRANSFERS={}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def atomic(p,v):
    tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def main(check_only=False):
    env=__import__('os').environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    raw=subprocess.check_output(['git','credential','fill'],input=b'protocol=https\nhost=github.com\n\n',env=env,cwd=ROOT)
    credential=dict(line.split('=',1) for line in raw.decode().splitlines() if '=' in line)
    headers={'Authorization':'Bearer '+credential['password'],'Accept':'application/vnd.github+json',
             'X-GitHub-Api-Version':'2022-11-28','User-Agent':'mcse-research-backup'}
    def api(url,method='GET',body=None):
        payload=json.dumps(body).encode() if body is not None else None
        req=urllib.request.Request(url,data=payload,headers={**headers,'Content-Type':'application/json'},method=method)
        with urllib.request.urlopen(req,timeout=60) as response:
            data=response.read();return json.loads(data) if data else None
    assert api('https://api.github.com/user')['login']=='chenciquan'
    assert api(API)['private'] is True
    previous=json.loads((CACHE/'upload_state.json').read_text(encoding='utf-8-sig'))
    release=api(API+'/releases/'+str(previous['release_id']))
    assert release['tag_name']=='research-snapshot-g35-20261113'
    if check_only:print('Native HTTPS authenticated; intended private release verified');return
    assets=api(API+'/releases/'+str(release['id'])+'/assets?per_page=100')
    verified={};removed=[]
    for a in assets:
        if a['state']=='uploaded' and a.get('digest'):
            verified[a['name']]=a
        elif a['name'].startswith('research-part-') and release['draft']:
            # Only unfinished assets from this new draft, never completed/history assets.
            api(API+'/releases/assets/'+str(a['id']),method='DELETE')
            removed.append({'id':a['id'],'name':a['name'],'state':a['state']})
        else:raise RuntimeError('Unexpected asset state; no replacement: '+a['name'])
    state={'status':'uploading','repository':'chenciquan/multichannel-speech-enhancement','private':True,
           'release_id':release['id'],'tag':release['tag_name'],'workers':3,'assets':[],
           'resumed_completed_assets':len(verified),'removed_unfinished_draft_assets':removed}
    started=time.monotonic();completed={};inflight={};checksums={}
    upload_base=release['upload_url'].split('{')[0]
    def upload(p,expected):
        name=p.name;size=p.stat().st_size
        if name in verified:
            a=verified[name]
            assert a['size']==size and a['digest']=='sha256:'+expected
            return {'name':name,'bytes':size,'sha256':expected,'url':a['browser_download_url'],'id':a['id'],'provider_digest':a['digest']}
        with LOCK:TRANSFERS[name]={'sent':0,'bytes':size}
        def chunks():
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024**2),b''):
                    yield block
                    with LOCK:TRANSFERS[name]['sent']+=len(block)
        req=urllib.request.Request(upload_base+'?name='+urllib.parse.quote(name),data=chunks(),method='POST',
            headers={**headers,'Content-Type':'application/octet-stream','Content-Length':str(size)})
        with urllib.request.urlopen(req,timeout=5400) as response:a=json.loads(response.read())
        assert a['state']=='uploaded' and a['size']==size and a.get('digest')=='sha256:'+expected
        with LOCK:TRANSFERS.pop(name,None)
        return {'name':name,'bytes':size,'sha256':expected,'url':a['browser_download_url'],'id':a['id'],'provider_digest':a['digest']}
    def status():
        state['assets']=list(completed.values());state['elapsed_seconds']=time.monotonic()-started
        with LOCK:state['transfers']={k:dict(v) for k,v in TRANSFERS.items()}
        atomic(CACHE/'upload_state.json',state)
    status()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            while True:
                progress=json.loads((CACHE/'progress.json').read_text())
                parts=sorted(CACHE.glob('research-part-*.zip'))[:progress['completed_parts']]
                for p in parts:
                    if len(inflight)>=3:break
                    if p.name not in completed and p.name not in {v[0].name for v in inflight.values()}:
                        expected=checksums.setdefault(p.name,sha(p))
                        inflight[pool.submit(upload,p,expected)]=(p,expected)
                if inflight:
                    ready,_=concurrent.futures.wait(inflight,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in ready:
                        p,_=inflight.pop(future);completed[p.name]=future.result()
                        print('Verified '+p.name,flush=True)
                else:time.sleep(2)
                status()
                complete=CACHE/'archive_complete.json'
                if complete.exists() and not inflight and len(completed)==json.loads(complete.read_text())['parts']:break
        for name in ['research-file-manifest.json','SHA256SUMS.txt']:
            p=CACHE/name;completed[name]=upload(p,sha(p));status();print('Verified '+name,flush=True)
        remote={a['name']:a for a in api(API+'/releases/'+str(release['id'])+'/assets?per_page=100')}
        for a in completed.values():
            r=remote[a['name']]
            assert r['state']=='uploaded' and r['size']==a['bytes'] and r['digest']=='sha256:'+a['sha256']
        published=api(API+'/releases/'+str(release['id']),method='PATCH',body={'draft':False})
        state['status']='complete';state['release_url']=published['html_url'];status()
        print('Release complete: '+published['html_url'],flush=True)
    except BaseException as e:
        state['status']='failed';state['error']=type(e).__name__+': '+str(e);status();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    main(args.check_only)
