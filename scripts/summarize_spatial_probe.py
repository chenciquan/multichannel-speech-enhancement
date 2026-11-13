#!/usr/bin/env python3
"""Aggregate full spatial diagnostics and predeclared paired confirmation checks."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def rows(path):return [json.loads(t) for t in Path(path).read_text().splitlines() if t.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--asr',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():p.error('Refusing to overwrite a previous summary')
    manifest={r['sample_id']:r for r in rows(args.manifest)};cases=rows(args.cases)
    if len(cases)!=len(manifest) or {r['sample_id'] for r in cases}!=set(manifest) or any(r['status']!='ok' for r in cases):
        raise ValueError('Expected complete matched inference evidence')
    args.output_dir.mkdir(parents=True)
    asr={r['sample_id']:r for r in rows(args.asr)} if args.asr else None
    if asr and (set(asr)!=set(manifest) or any(r['status']!='ok' for r in asr.values())):raise ValueError('ASR denominator mismatch')
    groups=defaultdict(list);pairs=defaultdict(dict)
    for c in cases:
        m=manifest[c['sample_id']]
        key=(c['noise_id'],m.get('noise_spectrum','raw'),m['noise_field'],m.get('seed_replica',0))
        groups[key].append(c)
        pairs[(c['source_id'],c['noise_id'],m.get('noise_spectrum','raw'),m.get('seed_replica',0))][m['noise_field']]=c
    grouped=[]
    for (noise,spectrum,field,seed),items in sorted(groups.items()):
        item={'noise_id':noise,'noise_spectrum':spectrum,'noise_field':field,'seed_replica':seed,'n':len(items),
              'mean_noise_msc':float(np.mean([manifest[c['sample_id']]['coherence']['mean'] for c in items]))}
        for metric in ('si_sdr_db','pesq_wb','stoi'):
            for role in ('noisy','enhanced'):
                values=[c['metrics'][role][metric] for c in items]
                if any(v is None or not np.isfinite(v) for v in values):raise ValueError('Incomplete metric backend')
                item[role+'_'+metric]=float(np.mean(values))
            item['gain_'+metric]=item['enhanced_'+metric]-item['noisy_'+metric]
        if asr:
            for role in ('reference','noisy','enhanced'):
                values=[asr[c['sample_id']]['asr'][role]['score'] for c in items]
                total={k:sum(r[k] for r in values) for k in ('N','S','D','I','errors')}
                item[role+'_asr']={**total,'wer':total['errors']/total['N']}
        grouped.append(item)
    contrasts=[]
    for (source,noise,spectrum,seed),fields in sorted(pairs.items()):
        a,b=fields['original_point'],fields['conditioned_12wave']
        ax,_=sf.read(a['noisy_path'],always_2d=True);bx,_=sf.read(b['noisy_path'],always_2d=True)
        if not np.array_equal(ax[:,4],bx[:,4]):raise ValueError('Reference not matched')
        item={'source_id':source,'speaker_id':a['speaker_id'],'noise_id':noise,'noise_spectrum':spectrum,'seed_replica':seed,
              'original_id':a['sample_id'],'conditioned_id':b['sample_id'],'noisy_reference_bit_identical':True}
        for metric in ('si_sdr_db','pesq_wb','stoi'):
            item['delta_'+metric]=b['metrics']['enhanced'][metric]-a['metrics']['enhanced'][metric]
        contrasts.append(item)
    paired_summary=[]
    for key in sorted({(r['noise_id'],r['noise_spectrum'],r['seed_replica']) for r in contrasts}):
        noise,spectrum,seed=key
        values=[r for r in contrasts if (r['noise_id'],r['noise_spectrum'],r['seed_replica'])==key]
        paired_summary.append({'noise_id':noise,'noise_spectrum':spectrum,'seed_replica':seed,'n_sources':len(values),
              'mean_delta_si_sdr_db':float(np.mean([r['delta_si_sdr_db'] for r in values])),
              'median_delta_si_sdr_db':float(np.median([r['delta_si_sdr_db'] for r in values])),
              'positive_source_contrasts':sum(r['delta_si_sdr_db']>0 for r in values),
              'min_delta_si_sdr_db':min(r['delta_si_sdr_db'] for r in values),
              'mean_delta_pesq_wb':float(np.mean([r['delta_pesq_wb'] for r in values])),
              'mean_delta_stoi':float(np.mean([r['delta_stoi'] for r in values]))})
    confirmation=[r for r in paired_summary if r['noise_spectrum']=='highpass40_then_snr0']
    is_confirmation={r.get('split') for r in manifest.values()}=={'confirmation'}
    rule_passed=all(r['n_sources']==8 and r['mean_delta_si_sdr_db']>0 and r['positive_source_contrasts']>=6 for r in confirmation) if is_confirmation else None
    summary={'status':'ok','mixtures':len(cases),'source_utterances':len({r['source_id'] for r in cases}),
             'speakers':len({r['speaker_id'] for r in cases}),'groups':grouped,'paired_contrasts':paired_summary,
             'reference_matched_pairs':len(contrasts),'confirmation_rule_applicable':is_confirmation,
             'confirmation_rule_passed':rule_passed,'asr_included':bool(asr),'human_word_deletion_verdict':'not_assigned',
             'training':False,'is_final_method_test':False,
             'evidence_sha256':{'manifest':sha(args.manifest),'cases':sha(args.cases),**({'asr':sha(args.asr)} if args.asr else {})}}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for name,data in [('paired_contrasts.csv',contrasts),('grouped_metrics.csv',grouped)]:
        with (args.output_dir/name).open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spectra=sorted({r['noise_spectrum'] for r in grouped})
    fig,axes=plt.subplots(1,len(spectra),figsize=(6*len(spectra),4),squeeze=False,layout='constrained')
    for ax,spectrum in zip(axes[0],spectra):
        for ni,noise in enumerate(sorted({r['noise_id'] for r in grouped})):
            for fi,field in enumerate(('original_point','conditioned_12wave')):
                vals=[r['gain_si_sdr_db'] for r in grouped if r['noise_id']==noise and r['noise_spectrum']==spectrum and r['noise_field']==field]
                ax.bar(ni*3+fi,np.mean(vals),color=('#b86631','#247488')[fi],label=field if ni==0 else None)
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_xticks([.5,3.5],['Kitchen','Washing'])
        ax.set(title=spectrum,ylabel='Mean SI-SDR improvement over same noisy reference (dB)')
        ax.legend(fontsize=8)
    fig.suptitle(f'{summary["source_utterances"]} source utterances; multiple conditions/seeds are not independent sentences',fontsize=11)
    fig.savefig(args.output_dir/'spatial_contrast.png',dpi=150);plt.close(fig)
    print(json.dumps({'status':'ok','source_utterances':summary['source_utterances'],'confirmation_rule_passed':rule_passed,'paired_contrasts':paired_summary},indent=2))


if __name__=='__main__':main()
