#!/usr/bin/env python3
"""Follow one live local signal job, then accept and score its frozen outputs."""
import os
import signal
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime,timedelta,timezone
from pathlib import Path
from g22_common import ROOT,OUT,sha,dump,now,metadata,status


def run(script,limit,label):
    with (OUT/(label+'.log')).open('x') as stream:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,
                                  stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=limit)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    assert code==0,label+' failed; do not retry automatically'


def await_release(path):
    for _ in range(24):
        if path.exists() and metadata(path)['status']=='released':return
        time.sleep(2)
    raise RuntimeError('Temporary power helper release not observed')


def main():
    pid=int(sys.argv[1]);proc=Path('/proc')/str(pid)/'cmdline';initial=proc.read_bytes()
    assert b'scripts/g22_signal_queue.py' in initial
    names=['scripts/g22_continue.py','scripts/g22_signal_accept.py','scripts/g22_asr.py',
           'scripts/g22_asr_finish.py','scripts/g22_final_accept.py','scripts/keep_awake_job.ps1']
    ready={'started_utc':now(),'signal_pid':pid,'observed_cmdline':initial.replace(b'\0',b' ').decode(),
           'max_signal_wait_seconds':22260,'max_ASR_process_seconds':7800,'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'CONTINUATION_READY.json',ready,True)
    def unchanged():assert all(sha(ROOT/n)==h for n,h in ready['hashes'].items())
    deadline=time.monotonic()+ready['max_signal_wait_seconds']
    while proc.exists():
        try:current=proc.read_bytes()
        except FileNotFoundError:break
        if current!=initial:break
        assert time.monotonic()<deadline,'Bounded wait exceeded; original job not restarted'
        time.sleep(5)
    unchanged()
    assert metadata(OUT/'evaluation/SIGNAL_COMPLETE.json')['status']=='ok'
    assert metadata(OUT/'status.json')['stage']=='signal_complete_pending_readback'
    await_release(OUT/'signal_power_helper.json')
    run('g22_signal_accept.py',600,'signal_acceptance')
    assert metadata(OUT/'SIGNAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    print('G22 signal stage accepted and archived',flush=True)
    unchanged()
    # A separate temporary request is tied to the ASR job, with its own stop path.
    powershell=shutil.which('powershell.exe');assert powershell
    windows_root='C:\\Users\\chenciquan\\Desktop\\multichannel_speech_enhancement'
    expiry=(datetime.now(timezone.utc)+timedelta(hours=3)).isoformat()
    args=[powershell,'-WindowStyle','Hidden','-NoProfile','-ExecutionPolicy','Bypass','-File',
          windows_root+'\\scripts\\keep_awake_job.ps1','-DeadlineUtc',expiry,
          '-StopFile',windows_root+'\\artifacts\\g22_matched_rooms_v1\\ASR_keep_awake_stop',
          '-StatusPath',windows_root+'\\artifacts\\g22_matched_rooms_v1\\ASR_power_helper.json']
    with (OUT/'ASR_power_helper.log').open('x') as log:
        helper=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(30):
                if (OUT/'ASR_power_helper.json').exists() and metadata(OUT/'ASR_power_helper.json')['status']=='requested':break
                assert helper.poll() is None,'Power helper exited before requesting wake state'
                time.sleep(1)
            assert metadata(OUT/'ASR_power_helper.json')['status']=='requested'
            print('G22 fixed ASR started; uncached references will also be decoded',flush=True)
            run('g22_asr.py',ready['max_ASR_process_seconds'],'ASR_job')
        finally:
            (OUT/'ASR_keep_awake_stop').write_text('G22 ASR continuation ended\n')
            helper.wait(timeout=30)
    await_release(OUT/'ASR_power_helper.json');unchanged()
    run('g22_final_accept.py',600,'final_acceptance')
    assert metadata(OUT/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    dump(OUT/'CONTINUATION_COMPLETE.json',{'status':'ok','utc':now(),
          'readiness_sha256':sha(OUT/'CONTINUATION_READY.json'),'acceptance_sha256':sha(OUT/'FINAL_ACCEPTANCE_AUDIT.json')},True)
    print('G22 complete and independently accepted',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(OUT/'CONTINUATION_FAILURE.json',{'utc':now(),'error':repr(e)});traceback.print_exc();raise
