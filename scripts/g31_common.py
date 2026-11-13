from g30_common import ROOT,DATA,sha,read,dump,now,meta
import g8_common as common
OUT=ROOT/'artifacts/g31_development_signals_v1';TRAIN=ROOT/'artifacts/g30_room_objectives_v1'


def archive(stage,report,files):
    old=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=old


def guard():
    p=meta(OUT/'frozen_plan.json')
    for n,h in p['hashes'].items():assert sha(ROOT/n)==h,n
    for r in p['registered'].values():assert sha(ROOT/r['path'])==r['sha256']
    return p


def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})
