#!/usr/bin/env python3
"""Scoped original trainer, prescribed final-step registration elsewhere."""
import argparse,json,sys
from datetime import datetime,timedelta,timezone
import g3_train as engine
from g15_common import ROOT,OUT,BASE


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int);a=p.parse_args();plan=json.loads((OUT/'frozen_plan.json').read_text());assert a.seed in plan['seeds']
    assert (OUT/'stages/preparation/archive_manifest.json').exists()
    engine.BASE=BASE;engine.OUT=OUT
    engine.PLAN={**engine.PLAN,'training_cutoff_utc':(datetime.now(timezone.utc)+timedelta(seconds=plan['max_per_seed_seconds']-30)).isoformat()}
    assert engine.PLAN['learning_rate']==plan['learning_rate']
    sys.argv=[__file__,'--variant','without_clean_identity','--seed',str(a.seed),'--steps','3000'];engine.main()


if __name__=='__main__':main()
