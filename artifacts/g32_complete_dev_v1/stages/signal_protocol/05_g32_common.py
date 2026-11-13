from g8_common import ROOT,sha,read,dump,now
import json
import g8_common as common
OUT=ROOT/'artifacts/g32_complete_dev_v1';BASE=ROOT/'data/g32_v1'
SOURCES=ROOT/'data/g3_v1/sources';RIR=ROOT/'artifacts/g29_room_development_v1'
TRAIN=ROOT/'artifacts/g30_room_objectives_v1';PREV=ROOT/'artifacts/g31_development_signals_v1'


def meta(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def archive(stage,report,files):
    previous=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous


def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})


def data_guard():
    p=meta(OUT/'data_plan.json')
    for n,h in p['hashes'].items():assert sha(ROOT/n)==h,n
    return p
