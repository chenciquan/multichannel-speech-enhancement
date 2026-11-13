import g8_common as common
ROOT=common.ROOT;OUT=ROOT/'artifacts/g15_rigid_training_v1';BASE=ROOT/'data/g15_v1'
now=common.now;sha=common.sha;dump=common.dump;read=common.read


def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})


def archive(stage,report,files):
    previous=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous
