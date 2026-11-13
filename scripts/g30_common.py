"""Same-budget objectives on G29; all decisions frozen before development scores."""
from g29_common import ROOT,BASE as DATA,sha,read,dump,now,meta
import g8_common as common
OUT=ROOT/'artifacts/g30_room_objectives_v1'


def archive(stage,report,files):
    old=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=old


def guard():
    plan=meta(OUT/'frozen_plan.json')
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h,n
    return plan


def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})
