#!/usr/bin/env python3
"""Complete seed blocks only; archive training evidence before confirmation."""
import json,subprocess,sys,time,traceback,shutil
from datetime import datetime,timezone
from pathlib import Path
from g4_build import sha,dump
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g5_rotation_v1';PLAN=json.loads((ROOT/'configs/g5_rotation_plan.json').read_text())
CUTOFF=datetime.fromisoformat(PLAN['training_cutoff_utc'].replace('Z','+00:00')).timestamp()
def state(**kw):dump(OUT/'queue_status.json',{'at_utc':datetime.now(timezone.utc).isoformat(),**kw})
def main():
    while not (OUT/'PREPARATION_COMPLETE.json').exists():
        if time.time()>datetime.fromisoformat(PLAN['data_cutoff_utc'].replace('Z','+00:00')).timestamp()+120:raise TimeoutError('Preparation not complete; inspect saved partial data, no old-source fallback')
        time.sleep(10)
    state(status='starting');all_results=[];selected=[];excluded=[];completed_seeds=[]
    cells=[('fixed','without_clean_identity'),('rotated','without_clean_identity'),('fixed','complete'),('rotated','complete')]
    for seed in [PLAN['mandatory_seed']]+PLAN['optional_seeds']:
        if all_results and time.time()+4*max(r['elapsed_seconds'] for r in all_results)+600>CUTOFF:break
        block=[];failure=None
        for geometry,variant in cells:
            state(status='training',geometry=geometry,variant=variant,seed=seed,completed_seed_blocks=completed_seeds)
            logfile=OUT/f'train_{geometry}_{variant}_s{seed}.log'
            with logfile.open('x') as log:
                p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/g5_train.py'),'--geometry',geometry,'--variant',variant,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,CUTOFF-time.time()))
            if p.returncode:failure=f'{geometry}/{variant} exited {p.returncode}';break
            folder=OUT/geometry/'train'/f'{variant}_s{seed}';r=json.loads((folder/'result.json').read_text());r['geometry']=geometry
            if r['status']!='ok':failure=f'{geometry}/{variant} status {r["status"]}';break
            block.append(r)
        if failure:
            excluded.append({'seed':seed,'reason':failure,'completed_cells_preserved':len(block),'rule':'Exclude entire incomplete seed block, not only failed cell'});break
        assert len(block)==4;all_results.extend(block);completed_seeds.append(seed)
        for r in block:
            path=ROOT/r['selected']['path'];selected.append({'system':f'{r["geometry"]}_{r["variant"]}_s{seed}','geometry':r['geometry'],'variant':r['variant'],'seed':seed,
              'kind':'adapter',**r['selected'],'checkpoint_sha256':sha(path),'completed_steps':r['completed_steps'],'stoi_gate_passed':r['stoi_gate_passed']})
    if not completed_seeds:raise RuntimeError('No full seed block completed; no confirmation evaluation')
    selection={'selected_at_utc':datetime.now(timezone.utc).isoformat(),'completed_seeds':completed_seeds,'excluded_seed_blocks':excluded,'confirmation_opened':False,'systems':selected,
       'selection_rule':PLAN['selection'],'confirmation_manifest_sha256':sha(ROOT/'data/g5_v1/confirmation/manifest.jsonl')}
    dump(OUT/'selected_checkpoints.json',selection)
    report=ROOT/'docs/G5B_整场景旋转训练报告.md';lines=['# G5B 整场景旋转训练报告','','配对数据和新说话人确认清单在训练前冻结。确认结果尚未打开。每种子完整执行固定/旋转×干净约束开/关四个单元；额外种子只由时间余量决定。','','|系统|有效更新|选定步数|验证SI-SDR dB|验证STOI|原模型冻结|','|---|---:|---:|---:|---:|---|']
    for r,s in zip(all_results,selected):lines.append(f'|{s["system"]}|{r["completed_steps"]}|{s["step"]}|{s["si_sdr_db"]:.3f}|{s["stoi"]:.4f}|{r["base_unchanged"]}|')
    lines+=['',f'完成{len(all_results)}次训练、{len(all_results)*3000}次有效更新，完整种子组：{completed_seeds}。',
       '', '两套训练池在源语音、噪声片段、SNR、相对空间关系和共同幅度尺度上配对，第5麦样本和监督目标相同。旋转的是完整场景，绝非只让目标靠近/远离固定噪声方向。干净约束组有额外计算，不是严格等算力。',
       '', '复用G3训练入口与原始监督目标，唯一日志泛化为验证条数从固定48改为实际条数54；原G3执行代码已在其阶段快照保留。原官方模型代码和权重没有修改。',
       '', f'排除的不完整种子组：{excluded}。没有按指标好坏删除结果。下一阶段评估3位新说话人、每人4句；本阶段验证结果不能当作确认结论。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8');folder=OUT/'stages/training';folder.mkdir(parents=True);entries=[]
    for i,p in enumerate([report,OUT/'selected_checkpoints.json',ROOT/'configs/g5_rotation_plan.json',ROOT/'scripts/g5_train.py',ROOT/'scripts/g3_train.py',Path(__file__)]):
        dst=folder/f'{i:02d}_{p.name}';shutil.copy2(p,dst);entries.append({'source':str(p.relative_to(ROOT)),'snapshot':dst.name,'sha256':sha(p)})
    dump(folder/'archive_manifest.json',{'saved_at_utc':datetime.now(timezone.utc).isoformat(),'files':entries,'checkpoint_hashes':[{k:s[k] for k in ('system','path','checkpoint_sha256')} for s in selected]})
    dump(OUT/'TRAIN_COMPLETE.json',{'runs':len(all_results),'complete_seed_blocks':completed_seeds,'selected_before_confirmation':True});state(status='training_archived',runs=len(all_results),completed_seeds=completed_seeds)
    while not (ROOT/'scripts/g5_evaluate.py').exists():
        if time.time()>CUTOFF+120:raise TimeoutError('Evaluation entry not ready')
        time.sleep(5)
    with (OUT/'evaluation.log').open('x') as log:p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/g5_evaluate.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,datetime.fromisoformat(PLAN['session_deadline_utc'].replace('Z','+00:00')).timestamp()-time.time()-120))
    if p.returncode:raise RuntimeError('G5 evaluation failed, inspect log; do not rerun training')
    state(status='complete',runs=len(all_results),completed_seeds=completed_seeds)
if __name__=='__main__':
    try:main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True);state(status='failed',error_type=type(e).__name__,error=str(e));traceback.print_exc();sys.exit(1)
