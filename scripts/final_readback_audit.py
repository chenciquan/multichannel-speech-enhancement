#!/usr/bin/env python3
"""Post-window read-only evidence checks, not a new experiment or metric run."""
import hashlib,json,runpy,sys
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/overnight_acceptance_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def main():
    OUT.mkdir(parents=True,exist_ok=False);checks={};stages={};cache={};snapshots=0;audio=0
    def verified_hash(path,expected):
        p=str(Path(path).resolve())
        if p not in cache:cache[p]=sha(Path(path))
        return cache[p]==expected
    for name,dirname,expected in [('G3','g3_overnight_v1',5850),('G4','g4_counterfactual_v1',2832),('G5','g5_rotation_v1',3528)]:
        base=ROOT/'artifacts'/dirname;previous=json.loads((base/'completion_audit.json').read_text());checks[name+'_stage_audit_passed']=previous['status']=='ok' and all(previous['checks'].values())
        marker=json.loads((base/'MAJOR_STAGE_COMPLETE.json').read_text());checks[name+'_finished_before_deadline']=datetime.fromisoformat(marker['completed_utc'])<datetime.fromisoformat('2026-11-11T21:54:15+00:00')
        snapshots_ok=True
        for manifest in sorted((base/'stages').glob('*/archive_manifest.json')):
            m=json.loads(manifest.read_text())
            for item in m['files']:
                snapshots+=1;snapshots_ok &= verified_hash(manifest.parent/item['snapshot'],item['sha256'])
            for item in m.get('referenced_artifacts',[]):snapshots_ok &= verified_hash(ROOT/item['path'],item['sha256'])
        checks[name+'_stage_snapshots_hash_match']=bool(snapshots_ok)
        rows=read(base/'evaluation/signals.jsonl');asr=read(base/'evaluation/asr_scores.jsonl');keys={(r['system'],r['sample_id']) for r in rows};akeys={(r['system'],r['sample_id']) for r in asr}
        checks[name+'_exact_signal_asr_coverage']=len(rows)==len(asr)==len(keys)==len(akeys)==expected and keys==akeys
        checks[name+'_all_saved_scores_valid']=all(r['status']=='ok' and all(v['status']=='ok' for v in r['metrics']['status'].values()) for r in rows)
        checks[name+'_all_output_audio_unchanged']=all(verified_hash(r['output_path'],r['output_sha256']) for r in rows);audio+=len(rows)
        # Independently validate micro-WER counts already saved in every ASR row.
        checks[name+'_wer_arithmetic']=all(r['score']['errors']==r['score']['S']+r['score']['D']+r['score']['I'] and r['score']['N']>0 and abs(r['score']['wer']-r['score']['errors']/r['score']['N'])<1e-12 for r in asr)
        summaries=json.loads((base/'evaluation/summary.json').read_text())
        if name=='G3':
            means_ok=True;ai={(r['system'],r['sample_id']):r for r in asr}
            for split in ('test','clean_test'):
                for system,summary in summaries[split].items():
                    subset=[r for r in rows if r['system']==system and r['split']==split]
                    means_ok &= len(subset)==summary['cases']
                    for metric in ('si_sdr_db','pesq_wb','stoi'):means_ok &= abs(np.mean([r['metrics'][metric] for r in subset])-summary['metrics'][metric])<1e-10
                    counts={k:sum(ai[(r['system'],r['sample_id'])]['score'][k] for r in subset) for k in ('N','S','D','I')}
                    means_ok &= all(counts[k]==summary['asr'][k] for k in counts)
            checks['G3_summary_recomputed_from_saved_rows']=bool(means_ok)
        stages[name]={'completed_utc':marker['completed_utc'],'signal_pairs':len(rows),'asr_pairs':len(asr),'systems':len({r['system'] for r in rows}),'report':marker['report']}
        checks[name+'_report_matches_completion_snapshot']=any(item['source']==marker['report'] and verified_hash(ROOT/marker['report'],item['sha256']) for item in json.loads((base/'stages/completion/archive_manifest.json').read_text())['files'])
    checkpoints=[]
    for sy in json.loads((ROOT/'artifacts/g3_overnight_v1/evaluation/systems.json').read_text()):
        if 'path' in sy:checkpoints.append(sy)
    checkpoints+=json.loads((ROOT/'artifacts/g5_rotation_v1/selected_checkpoints.json').read_text())['systems']
    checks['all24_selected_checkpoints_unchanged']=len(checkpoints)==24 and all(verified_hash(ROOT/r['path'],r['checkpoint_sha256']) for r in checkpoints)
    results=[json.loads((ROOT/r['path']).parent.joinpath('result.json').read_text()) for r in checkpoints]
    checks['all24_runs_3000_effective_updates_and_frozen_base']=all(r['status']=='ok' and r['completed_steps']==3000 and r['base_unchanged'] and r['base_digest_before']==r['base_digest_after'] for r in results)
    paired=json.loads((ROOT/'artifacts/g5_rotation_v1/paired_data_audit.json').read_text())
    checks['G5_saved_array_hashes_unchanged']=all(verified_hash(ROOT/'data/g5_v1'/g/(n+'.npy'),h) for g,items in paired['array_sha256'].items() for n,h in items.items())
    runtime=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='final_baseline_readback');repo=ROOT/'third_party/McNet';runtime['verify_upstream'](repo,repo/runtime['OFFLINE_CHECKPOINT']);checks['official_source_and_weight_hashes_unchanged']=True
    checks['failed_AMP_evidence_preserved']=(ROOT/'artifacts/g3_overnight_v1/train_first.log').is_file() and (ROOT/'artifacts/g3_overnight_v1/train/failed_amp_ordinary_s20261112/adapter_01500.pt').is_file()
    result={'status':'ok' if all(checks.values()) else 'failed','checked_utc':datetime.now(timezone.utc).isoformat(),'role':'Readback, checksums and arithmetic only after the authorized window; no new inference, metric estimation, training or download',
      'checks':checks,'stages':stages,'formal_training_runs':len(results),'effective_optimizer_updates':sum(r['completed_steps'] for r in results),
      'training_program_minutes_total':sum(r['elapsed_seconds'] for r in results)/60,'signal_pairs':audio,'asr_pairs':audio,'stage_snapshot_files_checked':snapshots,'unique_files_hashed':len(cache),
      'G6_proposal_executed':False,'quota_percent_verified':False,'time_window':'2026-11-12 01:54:15 to05:54:15 Asia/Shanghai; G3 began earlier under initial five-hour authorization'}
    (OUT/'readback_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('status','formal_training_runs','effective_optimizer_updates','signal_pairs','stage_snapshot_files_checked','training_program_minutes_total')}))
    if result['status']!='ok':sys.exit(1)
if __name__=='__main__':main()
