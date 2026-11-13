from g32_common import ROOT,sha,read,dump,now,meta
import g8_common as common
OUT=ROOT/'artifacts/g33_full_gradient_probe_v1';DATA=ROOT/'data/g29_v1'


def archive(stage,report,files):
    old=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=old


def guard():
    plan=meta(OUT/'frozen_plan.json')
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h,n
    return plan


def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})
