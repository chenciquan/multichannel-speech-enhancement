#!/usr/bin/env python3
"""Reuse the audited G3 trainer; only dataset/output/deadline configuration changes."""
import argparse,json,sys,traceback
from pathlib import Path
import g3_train as engine
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',choices=['fixed','rotated'],required=True);p.add_argument('--variant',choices=['complete','without_clean_identity'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    plan=json.loads((ROOT/'configs/g5_rotation_plan.json').read_text());engine.BASE=ROOT/'data/g5_v1'/a.geometry;engine.OUT=ROOT/'artifacts/g5_rotation_v1'/a.geometry
    engine.PLAN={**engine.PLAN,'training_cutoff_utc':plan['training_cutoff_utc']}
    sys.argv=[__file__,'--variant',a.variant,'--seed',str(a.seed),'--steps','3000'];engine.main()
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();sys.exit(1)
