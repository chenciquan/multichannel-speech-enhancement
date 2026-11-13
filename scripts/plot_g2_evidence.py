#!/usr/bin/env python3
"""Plot existing confirmation metrics on a common scale; do not rerun models."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing to replace a previous figure')
    data=json.loads(args.summary.read_text());assert data['source_utterances']==8 and data['confirmation_rule_passed']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),sharey=True,layout='constrained')
    for ax,spectrum,title in zip(axes,['raw','highpass40_then_snr0'],['Original noise','Noise highpass40, then restored SNR0']):
        for ni,noise in enumerate(['DKITCHEN_ch01','DWASHING_ch01']):
            for fi,field in enumerate(['original_point','conditioned_12wave']):
                values=[r['gain_si_sdr_db'] for r in data['groups'] if r['noise_id']==noise and r['noise_spectrum']==spectrum and r['noise_field']==field]
                assert len(values)==2
                value=float(np.mean(values));x=ni*3+fi
                ax.bar(x,value,color=['#b86631','#247488'][fi],label=['Point source (-60 deg)','Conditional12-wave'][fi] if ni==0 else None)
                ax.text(x,value+(.12 if value>=0 else -.12),f'{value:+.2f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=9)
        ax.axhline(0,color='black',linewidth=.8);ax.set_ylim(-2.3,5.5)
        ax.set_xticks([.5,3.5],['Kitchen','Washing']);ax.set_title(title,fontsize=10)
    axes[0].set_ylabel('SI-SDR gain (dB)');axes[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Unmodified McNet: 8 new utterances, 2 spatial seeds\nReference audio is identical within each spatial comparison',fontsize=11)
    fig.savefig(args.output,dpi=170);plt.close(fig)


if __name__=='__main__':main()
