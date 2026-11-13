"""Evidence roots for shared room-training and held-out development data."""
import json
import g8_common as common
ROOT=common.ROOT;OUT=ROOT/'artifacts/g29_room_development_v1';BASE=ROOT/'data/g29_v1'
SOURCES=ROOT/'data/g3_v1/sources'
sha,read,dump,now=common.sha,common.read,common.dump,common.now


def meta(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def archive(stage,report,files):
    old=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=old


def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})
