import g8_common as common

ROOT = common.ROOT
OUT = ROOT / 'artifacts/g19_room_mwf_v1'
PREV = ROOT / 'artifacts/g18_room_stress_v1'
now, sha, read, dump = common.now, common.sha, common.read, common.dump


def status(**fields):
    dump(OUT / 'status.json', {'utc': now(), **fields})


def archive(stage, report, files):
    previous = common.OUT
    try:
        common.OUT = OUT
        common.archive(stage, report, files)
    finally:
        common.OUT = previous
