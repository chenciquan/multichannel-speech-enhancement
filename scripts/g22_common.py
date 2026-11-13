"""Evidence helpers for the frozen matched-room signal comparison."""
import json
from g22_prepare import ROOT,OUT,BASE,sha,read,dump,now,archive


def metadata(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})


def guard():
    ready=metadata(OUT/'SIGNAL_PIPELINE_READY.json')
    for n,h in ready['hashes'].items():assert sha(ROOT/n)==h,n
    assert sha(BASE/'manifest.jsonl')==ready['manifest_sha256']
    assert sha(OUT/'signal_systems.json')==ready['registry_sha256']
    assert sha(OUT/'DATA_ACCEPTANCE_AUDIT.json')==ready['data_acceptance_sha256']
    return ready
