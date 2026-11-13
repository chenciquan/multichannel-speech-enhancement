#!/usr/bin/env python3
"""Standalone bounded G12A job; data stage requires its archived report."""
import json,subprocess,sys,traceback
from g12_common import ROOT,OUT,now,sha,dump,status


def main():
    gate=ROOT/'artifacts/g11_components_v1';assert json.loads((gate/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    assert (gate/'stages/acceptance/archive_manifest.json').exists();OUT.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'configs/g12_reverb_plan.json').read_text());plan.update(frozen_utc=now(),previous_archive_sha256=sha(gate/'stages/acceptance/archive_manifest.json'),
      geometry_sha256=sha(ROOT/'configs/g1_geometry.json'),G10_systems_sha256=sha(ROOT/'artifacts/g10_fresh_voice_v1/systems.json'),
      sources_manifest_sha256=sha(ROOT/'data/g10_v1/sources/speech_manifest.jsonl'),beamforming_sha256=sha(ROOT/'src/mcse/beamforming.py'))
    dump(OUT/'frozen_plan.json',plan,True)
    with (OUT/'asset.log').open('x') as log:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/g12_asset.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=plan['max_asset_seconds'])
    if r.returncode:raise RuntimeError('G12 asset stage failed; preserve log and environment evidence before retry')
    assert (OUT/'stages/asset/archive_manifest.json').exists();status(stage='asset_complete_data_not_started')
    dump(OUT/'ASSET_JOB_COMPLETE.json',{'status':'ok','completed_utc':now(),'next':'Data generation only after this report archive,no benchmark yet'},True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status(stage='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc();sys.exit(1)
    finally:
        if OUT.exists():(OUT/'asset_keep_awake_stop').write_text('asset job ended\n')
