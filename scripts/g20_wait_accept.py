"""Boundedly follow one observed live process and accept its archived output."""
import os
import subprocess
import sys
import time
from pathlib import Path
from g20_replay import ROOT, OUT, dump, now, sha


def main():
    pid = int(sys.argv[1])
    proc = Path('/proc')/str(pid)/'cmdline'
    initial = proc.read_bytes()
    assert b'scripts/g20_filter.py' in initial
    ready = {'pid':pid,'observed_cmdline':initial.replace(b'\0',b' ').decode(),
             'started_utc':now(),'max_wait_seconds':3600,
             'accept_code_sha256':sha(ROOT/'scripts/g20_accept.py'),
             'monitor_code_sha256':sha(__file__)}
    dump(OUT/'ACCEPTANCE_WATCH.json',ready,True)
    deadline=time.monotonic()+3600
    while proc.exists():
        try: current=proc.read_bytes()
        except FileNotFoundError: break
        if current!=initial: break
        assert time.monotonic()<deadline, 'Observed process wait exceeded limit; no retry or kill'
        time.sleep(5)
    assert (OUT/'filter/COMPLETE.json').exists(), 'Process ended without complete output'
    for _ in range(12):
        import json
        power=json.loads((OUT/'power_helper.json').read_text(encoding='utf-8-sig'))
        if power['status']=='released':break
        time.sleep(5)
    assert power['status']=='released'
    assert sha(ROOT/'scripts/g20_accept.py')==ready['accept_code_sha256']
    with (OUT/'acceptance.log').open('x') as stream:
        subprocess.run([sys.executable,str(ROOT/'scripts/g20_accept.py')],cwd=ROOT,
                       stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
    print('G20 acceptance completed',flush=True)


if __name__=='__main__':main()
