#!/usr/bin/env python3
"""Bounded local G19 queue; archive every stage before advancing."""
import json
import os
import signal
import subprocess
import sys
import traceback

from g19_common import ROOT, OUT, dump, now, sha, status


def main():
    plan = json.loads((OUT / 'frozen_plan.json').read_text())
    assert (OUT / 'stages/preflight/archive_manifest.json').is_file()
    assert (OUT / 'stages/definition_clarification/archive_manifest.json').is_file()
    names = ['scripts/g19_common.py', 'scripts/g19_signal.py', 'scripts/g19_scores.py',
             'scripts/g19_finish.py', 'scripts/g19_queue.py', 'src/mcse/wiener.py',
             'src/mcse/beamforming.py', 'src/mcse/evaluation.py', 'src/mcse/asr_scoring.py',
             'src/mcse/ctc.py', 'scripts/g8_common.py']
    ready = {'frozen_utc': now(), 'files': {name: sha(ROOT / name) for name in names},
             'frozen_plan_sha256': sha(OUT / 'frozen_plan.json'),
             'preflight_archive_sha256': sha(OUT / 'stages/preflight/archive_manifest.json'),
             'clarification_archive_sha256': sha(OUT / 'stages/definition_clarification/archive_manifest.json')}
    dump(OUT / 'PIPELINE_READY.json', ready, True)

    def run(name, limit, log):
        assert all(sha(ROOT / path) == digest for path, digest in ready['files'].items())
        assert sha(OUT / 'frozen_plan.json') == ready['frozen_plan_sha256']
        with (OUT / log).open('x') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts' / name)],
                                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=limit)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        if code:
            raise RuntimeError(name + ' failed; evidence preserved, no automatic retry')

    status(stage='signal_starting')
    run('g19_signal.py', plan['max_signal_seconds'], 'signal.log')
    assert (OUT / 'stages/signal/archive_manifest.json').is_file()
    run('g19_scores.py', plan['max_ASR_seconds'], 'asr.log')
    assert (OUT / 'stages/ctc/archive_manifest.json').is_file()
    run('g19_finish.py', 600, 'finish.log')
    assert (OUT / 'MAJOR_STAGE_COMPLETE.json').is_file()
    status(stage='complete_pending_readback')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status(stage='failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        sys.exit(1)
    finally:
        (OUT / 'keep_awake_stop').write_text('G19 queue ended\n')
