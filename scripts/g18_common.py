import g8_common as common
ROOT=common.ROOT;OUT=ROOT/'artifacts/g18_room_stress_v1';BASE=ROOT/'data/g18_v1'
now=common.now;sha=common.sha;dump=common.dump;read=common.read


def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})


def archive(stage,report,files):
    old=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=old
