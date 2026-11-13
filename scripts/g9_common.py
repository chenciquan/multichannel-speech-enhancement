"""G9 artifact helpers with explicit output directory, no existing run edits."""
from pathlib import Path
import g8_common as common
ROOT=common.ROOT
OUT=ROOT/'artifacts/g9_beamforming_v1'
now=common.now;sha=common.sha;read=common.read;dump=common.dump


def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})


def archive(stage,report,files):
    previous=common.OUT
    try:
        common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous
