#!/usr/bin/env python3
"""Verify actual G2 inputs, results, confirmation contrasts and selected research direction."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
from urllib.parse import unquote,urlsplit
from zoneinfo import ZoneInfo
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):return [json.loads(t) for t in Path(path).read_text().splitlines() if t.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_si(reference,estimate):
    s=reference-reference.mean();y=estimate-estimate.mean()
    target=s*float(s@y)/float(s@s);residual=y-target
    return float(10*np.log10(float(target@target)/float(residual@residual)))


class ResourceLinks(HTMLParser):
    def __init__(self):super().__init__();self.audio=[];self.resources=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='audio':self.audio.append(a['src'])
        for key in ('src','href'):
            if key in a:self.resources.append(a[key])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Refusing to replace previous audit')
    checks=[];evidence={}
    def check(name,passed,detail=None):
        checks.append({'requirement':name,'passed':bool(passed),'detail':detail})
        if not passed:raise ValueError(name)
    try:
        old_sources=rows(ROOT/'data/sources/g1_v1/source_manifest.jsonl')
        new_sources=rows(ROOT/'data/sources/g2_confirmation_v1/source_manifest.jsonl')
        old_ids={r['source_id'] for r in old_sources};old_hashes={r['sha256'] for r in old_sources}
        check('Eight new complete utterances exclude all24 old IDs and file hashes',len(new_sources)==8
              and len({r['source_id'] for r in new_sources})==8
              and not old_ids.intersection(r['source_id'] for r in new_sources)
              and not old_hashes.intersection(r['sha256'] for r in new_sources)
              and len({r['speaker_id'] for r in new_sources})==6)
        for r in new_sources:
            path=ROOT/'data/sources/g2_confirmation_v1'/r['path'];a,sr=sf.read(path)
            check('New source bytes/format/full length: '+r['source_id'],sha(path)==r['sha256'] and sr==16000
                  and a.ndim==1 and len(a)==r['samples'] and 1<=len(a)/sr<=3.8 and bool(r['transcript']))
        fetch=read(ROOT/'data/sources/g2_confirmation_v1/fetch_report.json')
        sourceplan=ROOT/'configs/g2_confirmation_plan.json'
        check('Bounded official speech acquisition completed without full archive',fetch['status']=='ok'
              and fetch['compressed_bytes_streamed']==43028480 and not fetch['whole_archive_saved']
              and not fetch['whole_archive_hash_verified'] and fetch['plan_sha256']==sha(sourceplan)
              and (ROOT/'data/sources/g2_confirmation_v1/source_plan_frozen.json').read_bytes()==sourceplan.read_bytes())
        newfiles=[ROOT/'data/sources/g2_confirmation_v1'/r['path'] for r in new_sources]
        check('Confirmation hypothesis frozen after discovery and before new-source acquisition',
              (ROOT/'artifacts/g2_factorial_v1/mcnet/cases.jsonl').stat().st_mtime<=sourceplan.stat().st_mtime
              <=min(path.stat().st_mtime for path in newfiles))
        cp=ROOT/'third_party/McNet/model_checkpoints/offline/epoch494_criteria18.78_sdr18.78.ckpt'
        adapter=runpy.run_path(str(ROOT/'scripts/mcnet_infer.py'),run_name='g2_audit_adapter')
        verified=adapter['verify_upstream'](ROOT/'third_party/McNet',cp)
        check('Official checkpoint and9 imported code/config files remain pinned',verified['checkpoint']['sha256']=='24b8c556cc52990e4d7db83da4909b7e8a1dc4a0c14880b98381bb62818beecf'
              and len(verified['verified_source_files'])==9 and not subprocess.check_output(
                  ['git','-C',str(ROOT/'third_party/McNet'),'status','--porcelain'],text=True).strip())
        # Confirm that earlier evidence was preserved, rather than rewriting its metrics.
        oldcasepath=ROOT/'artifacts/g1_v1/mcnet/cases.jsonl'
        initial_audit=read(ROOT/'artifacts/g1_v1/completion_audit.json')
        check('Original G1 case/ASR/metric/manifest bytes unchanged',all(
              sha(v['path'])==v['sha256'] for v in initial_audit['evidence'].values()))
        for c in rows(oldcasepath):
            check('Original enhanced file intact: '+c['sample_id'],sha(c['enhanced_path'])==c['enhanced_file']['sha256'])
        all_manifests={};all_cases={}
        for name,expected in [('g2_spatial_v1',48),('g2_factorial_v1',48),('g2_angle_v1',48),('g2_confirmation_v1',128)]:
            data=ROOT/'data/diagnostic'/name;base=ROOT/'artifacts'/name/'mcnet'
            manifest=rows(data/'manifest.jsonl');case=rows(base/'cases.jsonl')
            generation=read(data/'frozen_plan.json')
            source_audio={r['source_id']:sf.read(r['path'],dtype='float64')[0] for r in generation['sources']}
            m={r['sample_id']:r for r in manifest};c={r['sample_id']:r for r in case}
            check(name+': all predeclared cases are present and successful',len(manifest)==len(m)==len(case)==expected
                  and set(m)==set(c) and all(r['status']=='ok' for r in case))
            provenance=read(base/'run_provenance.json')
            check(name+': unchanged model/input, no fitting or normalization',provenance['input_manifest_sha256']==sha(data/'manifest.jsonl')
                  and provenance['upstream']['checkpoint']['sha256']==verified['checkpoint']['sha256']
                  and provenance['model']['strict_state_dict_load'] and not provenance['training_performed']
                  and not provenance['upstream_source_edited'] and not provenance['per_utterance_peak_normalization']
                  and not provenance['resampling_or_chunking'])
            matched=defaultdict(list)
            for sid,r in m.items():
                item=c[sid];signals={}
                for role in ('reference','noisy','enhanced'):
                    path=Path(item[role+'_path']);a,sr=sf.read(path,dtype='float64',always_2d=True)
                    digest=item['enhanced_file']['sha256'] if role=='enhanced' else r[role+'_sha256']
                    check(name+': '+sid+' '+role+' file/metadata',sha(path)==digest and sr==16000
                          and a.shape==(item['samples'],6 if role=='noisy' else 1)
                          and len(a)/sr<=4 and np.isfinite(a).all())
                    signals[role]=a[:,4] if role=='noisy' else a[:,0]
                    if role!='enhanced':
                        check(name+': '+sid+' input peak bounded',float(np.max(np.abs(a)))<=.950001)
                for role in ('noisy','enhanced'):
                    metric=item['metrics'][role]
                    check(name+': '+sid+' '+role+' three real metrics and independent SI check',
                          all(metric.get(k) is not None and np.isfinite(metric[k]) and metric['status'][k]['status']=='ok'
                              for k in ('si_sdr_db','pesq_wb','stoi'))
                          and abs(independent_si(signals['reference'],signals[role])-metric['si_sdr_db'])<1e-7)
                expected_reference=np.pad(source_audio[r['source_id']],(64,64))*r['common_scale']
                check(name+': '+sid+' unfiltered complete speech target retained',
                      expected_reference.shape==signals['reference'].shape and np.max(np.abs(expected_reference-signals['reference']))<=1.3e-7)
                # Independently reconstruct the declared20ms/10ms activity mask.
                s=signals['reference'];n=signals['noisy']-s;starts=range(0,len(s),160)
                powers=np.array([np.mean(s[i:i+320]**2) for i in starts]);active=np.zeros(len(s),dtype=bool)
                for i,power in zip(range(0,len(s),160),powers):
                    if power>=powers.max()*1e-4:active[i:i+320]=True
                measured_snr=float(10*np.log10(np.mean(s[active]**2)/np.mean(n[active]**2)))
                check(name+': '+sid+' actual active reference SNR matches frozen0dB',r['snr_db']==0 and abs(measured_snr)<.01)
                matched[(r['source_id'],r['noise_id'],r.get('noise_spectrum','raw'))].append(signals['noisy'])
            check(name+': noisy reference identical across every paired spatial/seed group',all(
                  all(np.array_equal(g[0],a) for a in g[1:]) for g in matched.values()))
            all_manifests[name]=m;all_cases[name]=c
            evidence[str((data/'manifest.jsonl').relative_to(ROOT))]=sha(data/'manifest.jsonl')
            evidence[str((base/'cases.jsonl').relative_to(ROOT))]=sha(base/'cases.jsonl')
        conf=all_manifests['g2_confirmation_v1'];cases=all_cases['g2_confirmation_v1']
        check('Confirmation is8 new sources, confirmation split only',len({r['source_id'] for r in conf.values()})==8
              and {r['split'] for r in conf.values()}=={'confirmation'} and not old_ids.intersection(r['source_id'] for r in conf.values()))
        frozen=read(ROOT/'data/diagnostic/g2_confirmation_v1/frozen_plan.json')
        check('Confirmation render plan unchanged since generation',frozen['plan_sha256']==sha(ROOT/'configs/g2_confirmation_render_plan.json')
              and frozen['generation_script_sha256']==sha(ROOT/'scripts/build_spatial_probe.py')
              and frozen['spatial_noise_module_sha256']==sha(ROOT/'src/mcse/spatial_noise.py'))
        check('Render plan precedes every enhanced confirmation output',
              (ROOT/'data/diagnostic/g2_confirmation_v1/frozen_plan.json').stat().st_mtime
              <=min(Path(r['enhanced_path']).stat().st_mtime for r in cases.values()))
        oldmix=rows(ROOT/'data/diagnostic/g1_v1/manifest.jsonl')
        source_lengths={r['source_id']:r['samples'] for r in new_sources};segments={}
        for r in conf.values():segments[(r['source_id'],r['noise_id'])]=(r['noise_segment_start_sample'],r['noise_segment_start_sample']+source_lengths[r['source_id']])
        for (source,noise),(a,b) in segments.items():
            old_intervals=[(r['noise_segment_start_sample'],r['noise_segment_start_sample']+r['noise_segment_samples']) for r in oldmix if r['noise_id']==noise]
            other_intervals=[v for key,v in segments.items() if key[1]==noise and key[0]!=source]
            check('No noise-clip overlap with original24 or other new sources: '+source+noise,
                  all(b+4000<=x or a-4000>=y for x,y in old_intervals+other_intervals))
        summaries=read(ROOT/'artifacts/g2_confirmation_v1/analysis_asr/summary.json')
        pairs=defaultdict(dict)
        for sid,r in conf.items():pairs[(r['source_id'],r['noise_id'],r['noise_spectrum'],r['seed_replica'])][r['noise_field']]=cases[sid]
        primary=defaultdict(list)
        for (source,noise,spectrum,seed),fields in pairs.items():
            if spectrum=='highpass40_then_snr0':primary[(noise,seed)].append(fields['conditioned_12wave']['metrics']['enhanced']['si_sdr_db']-fields['original_point']['metrics']['enhanced']['si_sdr_db'])
        check('Predeclared primary confirmation rule independently passes all four strata',len(primary)==4
              and all(len(v)==8 and np.mean(v)>0 and sum(x>0 for x in v)>=6 for v in primary.values())
              and summaries['confirmation_rule_passed'])
        for (noise,seed),values in primary.items():
            g=next(g for g in summaries['paired_contrasts'] if g['noise_id']==noise and g['seed_replica']==seed and g['noise_spectrum']=='highpass40_then_snr0')
            check('Actual primary contrast matches published summary: '+noise+str(seed),
                  abs(float(np.mean(values))-g['mean_delta_si_sdr_db'])<1e-10 and sum(v>0 for v in values)==g['positive_source_contrasts'])
        # Check all three ASR lanes refer to the actual matched audio, not stale files.
        for name,expected in [('g2_factorial_v1',48),('g2_confirmation_v1',128)]:
            base=ROOT/'artifacts'/name;asr=rows(base/'asr/transcriptions.jsonl')
            check(name+': complete three-lane ASR',len(asr)==expected and {r['sample_id'] for r in asr}==set(all_cases[name]) and all(r['status']=='ok' for r in asr))
            for r in asr:
                c=all_cases[name][r['sample_id']]
                check(name+': '+r['sample_id']+' ASR uses unchanged audio bytes',all(
                      r['asr'][role]['audio']['sha256']==sha(c[role+'_path']) for role in ('reference','noisy','enhanced')))
            if name=='g2_confirmation_v1':
                unique={r['source_id']:r['asr']['reference']['score']['N'] for r in asr}
                check('ASR text denominator75 unique words,1200 repeated-condition words; not128 independent sentences',
                      len(unique)==8 and sum(unique.values())==75 and sum(r['asr']['reference']['score']['N'] for r in asr)==1200)
        # Actual stored DSP waveforms must support the low-frequency cause statement.
        controls=rows(ROOT/'artifacts/g2_confirmation_v1/filter_controls/saved_audio_scores/scores.jsonl')
        rescoring=read(ROOT/'artifacts/g2_confirmation_v1/filter_controls/saved_audio_scores/summary.json')
        check('DSP scores correspond to saved FLOAT bytes; original prewrite evidence preserved',
              rescoring['metrics_on_serialized_audio'] and not rescoring['audio_altered'] and not rescoring['model_rerun']
              and rescoring['source_cases_sha256']==sha(ROOT/'artifacts/g2_confirmation_v1/filter_controls/cases.jsonl'))
        check('All32 fixed-seed point inputs and six DSP controls retained',len(controls)==192
              and len({r['sample_id'] for r in controls})==32 and {r['method'] for r in controls}=={'noisy','mcnet','highpass_noisy','mcnet_then_highpass','highpass_then_mcnet','highpass_clean_sanity'})
        for r in controls:
            y,sr=sf.read(r['output_path']);s,_=sf.read(r['reference_path'])
            check('DSP waveform/hash and original clean target: '+r['sample_id']+r['method'],sha(r['output_path'])==r['output_sha256']
                  and sr==16000 and y.shape==s.shape and not r['target_filtered'] and not r['training']
                  and abs(independent_si(s,y)-r['metrics']['si_sdr_db'])<1e-7)
        raw=lambda mode:[r['metrics'] for r in controls if r['noise_spectrum']=='raw' and r['method']==mode]
        check('Low-frequency diagnostic reproduces large SI change without matching PESQ gain',
              np.mean([r['si_sdr_db'] for r in raw('highpass_noisy')])-np.mean([r['si_sdr_db'] for r in raw('noisy')])>15
              and np.mean([r['pesq_wb'] for r in raw('highpass_noisy')])<np.mean([r['pesq_wb'] for r in raw('noisy')]))
        spectra=read(ROOT/'artifacts/g2_spatial_v1/noise_spectrum_audit.json')
        noises={r['noise_id']:r for r in rows(ROOT/'data/sources/g1_v1/noise_manifest.jsonl')}
        for r in spectra['raw_recordings']:
            entry=noises[r['noise_id']];path=ROOT/'data/sources/g1_v1'/entry['path'];n,sr=sf.read(path)
            f=np.fft.rfftfreq(len(n),1/sr);power=np.abs(np.fft.rfft(n))**2
            if len(n)%2==0:power[1:-1]*=2
            else:power[1:]*=2
            fraction=float(power[f<20].sum()/power.sum())
            check('Measured original sub20Hz concentration and source hash: '+r['noise_id'],sha(path)==entry['sha256']==r['source_sha256']
                  and abs(fraction-r['energy_below20hz_fraction'])<1e-10
                  and abs(f[np.argmax(power[1:])+1]-r['nonDC_peak_frequency_hz'])<1e-10)
        review=ROOT/'artifacts/g2_confirmation_v1/review_v2';page=(review/'index.html').read_text(encoding='utf-8')
        links=ResourceLinks();links.feed(page)
        check('Final review uses correct128/8/6 counts and all384 mono audio links','128个混合，8条源语句、6位说话人' in page and len(links.audio)==384
              and all((review/unquote(v)).is_file() for v in links.resources if not urlsplit(v).scheme)
              and all(sf.info(review/unquote(v)).channels==1 and sf.info(review/unquote(v)).samplerate==16000 for v in links.audio))
        check('No new human word-loss judgments invented',read(review/'summary.json')['human_judgements_completed']==0
              and read(review/'summary.json')['confirmed_acoustic_deletions'] is None)
        for playback in read(review/'playback_manifest.json'):
            source,_=sf.read(playback['source_path'],dtype='float32',always_2d=True);copy,_=sf.read(playback['path'],dtype='float32')
            check('Review reference-mic copy exact: '+playback['sample_id'],sha(playback['path'])==playback['sha256'] and np.array_equal(source[:,4],copy))
        spec=read(ROOT/'configs/improvement_direction.json')
        model=read(ROOT/'artifacts/g1_v1/mcnet/run_provenance.json')['model']
        count=0
        for layer in spec['mechanism']['layers']:
            shape=model['state_dict_shapes'][layer['module']+'.weight']['shape'];count+=layer['rank']*sum(shape)
            check('Planned adapter budget uses actual model dimensions: '+layer['module'],shape==layer['weight_shape'])
        check('Selected direction follows confirmation; no unverified improvement or training claim',spec['status']=='selected_for_next_development_phase_not_implemented'
              and spec['confirmation_evidence_sha256']==sha(ROOT/spec['confirmation_evidence']) and count==3584==spec['mechanism']['trainable_parameters']
              and not spec['training_performed'] and not spec['adapter_implemented'] and not spec['training_memory_verified'])
        check('Direction selection recorded after completed confirmation summary',
              (ROOT/'configs/improvement_direction.json').stat().st_mtime>=(ROOT/spec['confirmation_evidence']).stat().st_mtime)
        verify=ROOT/'artifacts/g2_verification';log=(verify/'unit_tests.log').read_text(encoding='utf-8-sig')
        check('53 tests include spectral/physical known-answer checks and pass without skips',read(verify/'unit_tests_result.json')['exit_code']==0
              and 'Ran 53 tests' in log and '\nOK' in log and 'skipped' not in log.lower()
              and 'test_fixed_filter_removes16hz' in log and 'test_conditional_sources_sum_to_fixed_noise' in log)
        freeze=subprocess.check_output([str(ROOT/'.tools/uv-x86_64-unknown-linux-gnu/uv'),'pip','freeze','--python',str(ROOT/'.venv/bin/python')],text=True)
        check('No dependency additions; installed package snapshot equals original G1 lock',freeze.splitlines()==(ROOT/'requirements-diagnostics.lock.txt').read_text().splitlines()
              and 'loralib==' not in freeze)
        subprocess.run([str(ROOT/'.tools/uv-x86_64-unknown-linux-gnu/uv'),'pip','check','--python',str(ROOT/'.venv/bin/python')],check=True,capture_output=True)
        state=read(ROOT/'configs/project_status.json')
        check('Current project state registers selected direction and128-case confirmation',state['current_milestone']=='G2_cause_and_direction_selected'
              and state['method_selected'] and state['g2_confirmation_rule_passed'] and state['g2_confirmation_completed_mixtures']==128
              and not state['training_performed'] and not state['adapter_implemented'])
        report=ROOT/'docs/G2_原因分析与改进方向.md';text=report.read_text(encoding='utf-8');missing=[]
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if urlsplit(link).scheme or link.startswith('#'):continue
            path=(report.parent/unquote(link)).resolve()
            if path not in {args.output.resolve(),(ROOT/'artifacts/g2_verification/completion_audit.json').resolve()} and not path.is_file():missing.append(link)
        check('Final Chinese report and all cited local evidence exist',not missing,missing)
        evidence.update({str(path.relative_to(ROOT)):sha(path) for path in [report,ROOT/'configs/improvement_direction.json',sourceplan,
                        ROOT/'configs/g2_confirmation_render_plan.json',ROOT/'configs/g2_literature_evidence.json',ROOT/'configs/project_status.json',
                        ROOT/'artifacts/g2_confirmation_v1/analysis_asr/summary.json',ROOT/'artifacts/g2_confirmation_v1/asr/provenance.json',
                        ROOT/'scripts/audit_g2_delivery.py']})
        status,error='passed',None
    except Exception as exc:status,error='failed',{'type':type(exc).__name__,'message':str(exc)}
    result={'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'status':status,'checks':checks,
            'passed_checks':sum(r['passed'] for r in checks),'error':error,'evidence_sha256':evidence,
            'scope':'causes within controlled simulated conditions and an evidence-based next research direction',
            'not_claimed':['new method effectiveness or originality established','human word deletion established on G2','real-world or unseen-speaker/noise-type generalization','8GB training feasibility'],
            'training_performed':False}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':status,'passed_checks':result['passed_checks'],'error':error,'output':str(args.output)},ensure_ascii=False))
    return int(status!='passed')


if __name__=='__main__':raise SystemExit(main())
