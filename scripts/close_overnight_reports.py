#!/usr/bin/env python3
"""Verify final documentation/playback links and snapshot the closing state."""
import hashlib,json,re,shutil
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/overnight_acceptance_v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    evidence=json.loads((OUT/'readback_audit.json').read_text());assert evidence['status']=='ok'
    proc=json.loads((OUT/'process_check.json').read_text());assert proc['experiment_process_scan_exit_code']==1 and not proc['experiment_process_output'].strip() and not proc['keep_awake_pid19648_present']
    docs=[ROOT/'docs/夜间实验_验收汇总.md',ROOT/'docs/连续实验_阶段总览.md',ROOT/'README.md']
    for name in ('G3_五小时实验验收报告.md','G4_空间泛化压力测试报告.md','G5_整场景旋转与新说话人确认报告.md'):docs.append(ROOT/'docs'/name)
    link_count=0
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            p=path.parent/target.split('#',1)[0];assert p.exists(),f'Missing link: {path.name}: {target}';link_count+=1
    class Audio(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            if tag=='audio':self.paths.append(dict(attrs)['src'])
    audio_count=0
    for dirname in ('g3_overnight_v1','g4_counterfactual_v1','g5_rotation_v1'):
        folder=ROOT/'artifacts'/dirname/'review';p=Audio();p.feed((folder/'index.html').read_text(encoding='utf-8'))
        for target in p.paths:
            path=folder/target;info=sf.info(path);assert info.samplerate==16000 and info.channels==1 and info.frames>0;audio_count+=1
    status=json.loads((ROOT/'configs/project_status.json').read_text());session=json.loads((ROOT/'configs/continuous_experiment_session.json').read_text())
    assert status['G6_executed'] is False and status['experiment_processes_active'] is False and session['session_status']=='closed_after_G5'
    folder=OUT/'closing_snapshot';folder.mkdir(exist_ok=False);entries=[]
    files=[docs[0],docs[1],ROOT/'README.md',ROOT/'configs/project_status.json',ROOT/'configs/improvement_direction.json',ROOT/'configs/continuous_experiment_session.json',
           ROOT/'docs/本科毕业设计_研究路线与创新方向分析.md',ROOT/'docs/开发阶段记录.md',OUT/'readback_audit.json',OUT/'process_check.json',OUT/'terminal_summary.txt']
    for i,path in enumerate(files):
        dst=folder/f'{i:02d}_{path.name}';shutil.copy2(path,dst);entries.append({'source':str(path.relative_to(ROOT)),'snapshot':dst.name,'sha256':sha(path)})
    result={'status':'ok','closed_utc':datetime.now(timezone.utc).isoformat(),'local_document_links_checked':link_count,'mono_playback_links_checked':audio_count,
      'new_experiments_during_closing':False,'G6_executed':False,'files':entries,'explicit_limit':'G3-G5 complete; no universal winner, real-room validation, new human verdict, G6 latency benchmark or standalone CLI claimed.'}
    (OUT/'closing_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('status','local_document_links_checked','mono_playback_links_checked','G6_executed')}))
if __name__=='__main__':main()
