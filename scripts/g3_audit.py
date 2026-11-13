#!/usr/bin/env python3
"""Check completed G3 evidence against immutable manifests and original baseline."""
import hashlib,importlib.metadata,json,runpy,sys
from datetime import datetime
from pathlib import Path
from collections import Counter
import numpy as np
import soundfile as sf
import torch
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'artifacts/g3_overnight_v1';BASE=ROOT/'data/g3_v1/dataset';EVAL=OUT/'evaluation'
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    checks={};rt=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g3_audit_runtime');repo=ROOT/'third_party/McNet'
    rt['verify_upstream'](repo,repo/rt['OFFLINE_CHECKPOINT']);checks['official_files_and_weight_hashes_unchanged']=True
    protocol=json.loads((BASE/'frozen_protocol.json').read_text());source=read(ROOT/'data/g3_v1/sources/speech_manifest.jsonl')
    checks['source_manifest_matches_frozen_hash']=sha(ROOT/'data/g3_v1/sources/speech_manifest.jsonl')==protocol['source_manifest_sha256']
    sets={split:{r['speaker_id'] for r in source if r['split']==split} for split in ('train','validation','test')}
    checks['speakers_disjoint']=all(not sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    checks['old_diagnostic_ids_excluded']=not {r['source_id'] for r in source}&set(protocol['excluded_diagnostic_sources'])
    checks['source_audio_sha256']=all(sha(ROOT/'data/g3_v1/sources'/r['path'])==r['sha256'] for r in source)
    selected=json.loads((OUT/'selected_checkpoints_extended.json').read_text());sig=read(EVAL/'signals.jsonl');asr=read(EVAL/'asr_scores.jsonl')
    ep=json.loads((EVAL/'protocol.json').read_text());checks['selection_manifest_unchanged']=sha(OUT/'selected_checkpoints_extended.json')==ep['selection_manifest_sha256']
    checks['original_primary_selection_preserved']=sha(OUT/'selected_checkpoints.json')==selected['primary_selection_sha256']
    checks['selection_before_test']=datetime.fromisoformat(selected['selected_at_utc'])<datetime.fromisoformat(ep['started_utc'])
    ta=json.loads((OUT/'stages/training/archive_manifest.json').read_text());sa=json.loads((OUT/'stages/signal/archive_manifest.json').read_text())
    ap=json.loads((EVAL/'asr_provenance.json').read_text())
    checks['training_report_archived_before_test']=datetime.fromisoformat(ta['saved_at_utc'])<datetime.fromisoformat(ep['started_utc'])
    checks['signal_report_archived_before_asr']=datetime.fromisoformat(sa['saved_at_utc'])<datetime.fromisoformat(ap['started_utc'])
    results=[]
    for s in selected['systems']:
        r=json.loads((OUT/'train'/s['system']/'result.json').read_text());results.append(r)
        assert r['selected']['path']==s['path'] and r['completed_steps']==3000
        cp=torch.load(ROOT/s['path'],map_location='cpu',weights_only=False)
        assert cp['step']==s['step'] and sum(t.numel() for t in cp['adapter'].values())==3584
    checks['all_runs_same_3000_updates']=all(r['completed_steps']==3000 for r in results)
    checks['all_base_parameters_frozen']=all(r['base_unchanged'] and r['base_digest_before']==r['base_digest_after'] for r in results)
    checks['same_original_base_across_runs']=len({r['base_digest_before'] for r in results})==1
    checks['all_checkpoint_stoi_gates_passed']=all(r['stoi_gate_passed'] for r in results)
    names={r['system'] for r in sig};test=read(BASE/'test/manifest.jsonl');testids={r['sample_id'] for r in test}
    checks['all_systems_exact_same_360_test_cases']=all({r['sample_id'] for r in sig if r['system']==s and r['split']=='test'}==testids for s in names)
    checks['all_systems_30_clean_controls']=all(len([r for r in sig if r['system']==s and r['split']=='clean_test'])==30 for s in names)
    checks['no_missing_or_duplicate_signal_pairs']=len(sig)==len(names)*390==len({(r['system'],r['sample_id']) for r in sig})
    checks['no_metric_failures']=all(r['status']=='ok' and all(v['status']=='ok' for v in r['metrics']['status'].values()) for r in sig)
    checks['asr_coverage_matches_signals']={(r['system'],r['sample_id']) for r in sig}=={(r['system'],r['sample_id']) for r in asr} and len(asr)==len(sig)
    checks['audio_output_hashes']=all(sha(r['output_path'])==r['output_sha256'] for r in sig if r['status']=='ok')
    checks['test_mixture_and_reference_hashes']=all(sha(BASE/'test'/r['noisy_path'])==r['noisy_sha256'] and sha(BASE/'test'/r['reference_path'])==r['reference_sha256'] for r in test)
    # Recompute all HPF controls from the actual saved noisy reference microphone.
    from mcse.spatial_noise import highpass40
    checks['hpf_outputs_match_fixed_control']=True
    for r in sig:
        if r['system']!='hpf40':continue
        x,sr=sf.read(r['input_path'],dtype='float32',always_2d=True);y,_=sf.read(r['output_path'],dtype='float32')
        if sr!=16000 or not np.array_equal(y,highpass40(x[:,4]).astype('float32')):checks['hpf_outputs_match_fixed_control']=False
    checks['loralib_pinned']=importlib.metadata.version('loralib')=='0.1.2'
    checks['unit_tests_55_passed']='Ran 55 tests' in (OUT/'unit_tests.log').read_text() and '\nOK' in (OUT/'unit_tests.log').read_text()
    class AudioLinks(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            if tag=='audio':self.paths.append(dict(attrs).get('src'))
    parser=AudioLinks();parser.feed((OUT/'review/index.html').read_text())
    checks['review_links_all_mono_16k']=bool(parser.paths) and all(p and (OUT/'review'/p).is_file() and sf.info(OUT/'review'/p).channels==1 and sf.info(OUT/'review'/p).samplerate==16000 for p in parser.paths)
    review=json.loads((OUT/'review/selection.json').read_text());checks['review_five_roles_per_selected_case']=len(parser.paths)==len(review)*5
    checks['review_common_gain_and_no_clipping']=True
    sigidx={(r['system'],r['sample_id']):r for r in sig}
    for selection in review:
        sid=selection['id'];gain=np.float32(selection['listening_common_gain']);row=sigidx[('noisy',sid)]
        raw,_=sf.read(row['input_path'],dtype='float32',always_2d=True)
        originals={'reference':sf.read(row['reference_path'],dtype='float32')[0],'noisy':raw[:,4]}
        for role in ('mcnet','ordinary_s20261112','complete_s20261112'):originals[role]=sf.read(sigidx[(role,sid)]['output_path'],dtype='float32')[0]
        for role,a in originals.items():
            path=OUT/'review'/('noisy_mic5' if role=='noisy' else 'audio/'+role)/(sid+'.wav');copy,_=sf.read(path,dtype='float32')
            if not np.array_equal(copy,(a*gain).astype('float32')) or np.abs(copy).max()>1:checks['review_common_gain_and_no_clipping']=False
    # A failed selection gate is scientifically reportable, not silent success.
    mandatory={k:v for k,v in checks.items() if k!='all_checkpoint_stoi_gates_passed'}
    result={'status':'ok' if all(mandatory.values()) else 'failed','checks':checks,'counts':{'systems':len(names),'runs':len(results),'signal_pairs':len(sig),'asr_pairs':len(asr)},
       'limitations_not_resolved_by_audit':['no formal human listening','four test speakers only','no real room or physical-array recordings','single ASR','novelty not established','two ablation seeds added after first-seed validation; classified exploratory']}
    (OUT/'completion_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['status']!='ok':sys.exit(1)
if __name__=='__main__':main()
