#!/usr/bin/env python3
"""Static figures from actual completed results; no smoothed/fabricated curves."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/g3_overnight_v1'
def main():
    summary=json.loads((OUT/'evaluation/summary.json').read_text());folder=OUT/'evaluation/figures';folder.mkdir(exist_ok=False)
    seeds=summary['coverage']['paired_seeds'];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for variant,color in [('ordinary','#4769a8'),('complete','#b54c35')]:
        runs=[json.loads((OUT/'train'/f'{variant}_s{s}'/'result.json').read_text()) for s in seeds]
        steps=[r['step'] for r in runs[0]['history']]
        for ax,key in zip(axes,('si_sdr_db','stoi')):
            vals=np.array([[r[key] for r in run['history']] for run in runs]);mean=vals.mean(axis=0);std=vals.std(axis=0,ddof=1) if len(vals)>1 else np.zeros_like(mean)
            ax.plot(steps,mean,'o-',color=color,label=f'{variant}, {len(seeds)} seeds');ax.fill_between(steps,mean-std,mean+std,color=color,alpha=.14)
            ax.set_xlabel('Optimizer updates');ax.set_ylabel('Validation SI-SDR (dB)' if key=='si_sdr_db' else 'Validation STOI');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Validation curves: mean and sample SD, fixed 48 mixtures');fig.savefig(folder/'validation_curves.png',dpi=170);plt.close(fig)
    comparison=next(r for r in summary['comparisons'] if r['label'].startswith('配对种子'))['source_paired_bootstrap_2000']
    fig,axes=plt.subplots(1,4,figsize=(11,3.8),layout='constrained')
    for ax,key,title,scale in zip(axes,('si_sdr_db','pesq_wb','stoi','wer'),('SI-SDR (dB)','PESQ','STOI','WER (percentage points)'),(1,1,1,100)):
        r=comparison[key];point=r['delta']*scale;lo,hi=np.array(r['ci95'])*scale
        ax.errorbar([0],[point],yerr=[[point-lo],[hi-point]],fmt='o',capsize=6,color='#b54c35');ax.axhline(0,color='gray',linestyle='--');ax.set_xlim(-.6,.6);ax.set_xticks([]);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        ax.set_xlabel('Higher is better' if key!='wer' else 'Lower is better')
    fig.suptitle('Complete minus ordinary: paired-seed mean; exploratory 95% source bootstrap');fig.savefig(folder/'test_paired_deltas.png',dpi=170);plt.close(fig)
    clean=summary['clean_test'];names=['mcnet','ordinary_s20261112','complete_s20261112','without_stratification_s20261112','without_clean_identity_s20261112']
    labels=['McNet','Ordinary','Complete','No balance','No clean loss'];fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].bar(labels,[clean[n]['metrics']['si_sdr_db'] for n in names],color='#4769a8');axes[0].set_ylabel('Clean-input SI-SDR (dB)')
    axes[1].bar(labels,[100*clean[n]['asr']['wer'] for n in names],color='#b54c35');axes[1].set_ylabel('Clean-input WER (%)')
    for ax in axes:ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
    fig.suptitle('30 held-out clean-input controls, first seed; no human verdict');fig.savefig(folder/'clean_controls.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
