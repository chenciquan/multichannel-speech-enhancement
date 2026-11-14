"""Append-only private Release backup of this approved round; no ML execution.

Each independent ZIP contains project-relative paths and can be extracted into
a NEW clone. Every file and ZIP is locally read back and SHA256 verified.
Credentials stay in memory; existing completed assets are never replaced.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/g35_multifocus_round1_v1'
CACHE = ROOT / '.cache/github_g35_multifocus_round1_20261114'
API = 'https://api.github.com/repos/chenciquan/multichannel-speech-enhancement'
TAG = 'g35-multifocus-r1-20261114'
CHUNK = 128 * 1024 ** 2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def save(path, data, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def prepare():
    assert json.loads((SOURCE / 'ROUND1_COMPLETE.json').read_text(encoding='utf-8'))['status'] == 'passed'
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / 'round1-file-manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert {r['path'] for r in manifest['files']} == {
            p.relative_to(ROOT).as_posix() for p in SOURCE.rglob('*') if p.is_file()}
        for r in manifest['files']:
            p = ROOT / r['path']
            assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256']
        for r in manifest['archives']:
            p = CACHE / r['name']
            assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256']
        return manifest
    files = sorted(p for p in SOURCE.rglob('*') if p.is_file())
    assert files and not any(p.is_symlink() for p in files)
    groups, current, size = [], [], 0
    for p in files:
        assert p.resolve().is_relative_to(SOURCE.resolve())
        assert p.suffix.lower() not in ('.pem', '.key') and not p.name.startswith('.env')
        if current and size + p.stat().st_size > CHUNK:
            groups.append(current); current, size = [], 0
        current.append(p); size += p.stat().st_size
    if current:
        groups.append(current)
    manifest = {'version': 1, 'tag': TAG, 'source': SOURCE.relative_to(ROOT).as_posix(),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'restore': 'Extract all independent ZIPs into a NEW clone, never overwrite an existing workspace.',
                'files': [], 'archives': []}
    for number, group in enumerate(groups, 1):
        archive = CACHE / f'g35-round1-part-{number:03d}-of-{len(groups):03d}.zip'
        assert not archive.exists(), 'Partial archive requires explicit inspection; no overwrite'
        records = []
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in group:
                record = {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                          'sha256': sha(p), 'archive': archive.name}
                z.write(p, record['path']); records.append(record)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for r in records:
                h = hashlib.sha256()
                with z.open(r['path']) as f:
                    for b in iter(lambda: f.read(1024 ** 2), b''):
                        h.update(b)
                assert h.hexdigest() == r['sha256'] and sha(ROOT / r['path']) == r['sha256']
        manifest['files'].extend(records)
        manifest['archives'].append({'name': archive.name, 'bytes': archive.stat().st_size,
                                     'sha256': sha(archive), 'file_count': len(records)})
        print('Archive verified:', archive.name, len(records), flush=True)
    manifest['file_count'] = len(manifest['files'])
    manifest['source_bytes'] = sum(r['bytes'] for r in manifest['files'])
    manifest['archive_bytes'] = sum(r['bytes'] for r in manifest['archives'])
    manifest['all_archive_CRC_and_per_file_SHA256_verified'] = True
    save(manifest_path, manifest, True)
    with (CACHE / 'ROUND1_SHA256SUMS.txt').open('x', encoding='utf-8') as f:
        for r in manifest['archives']:
            f.write(r['sha256'] + '  ' + r['name'] + '\n')
    return manifest


def upload(manifest):
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    raw = subprocess.check_output(['git', 'credential', 'fill'], cwd=ROOT, env=env,
                                  input=b'protocol=https\nhost=github.com\n\n')
    credential = dict(s.split('=', 1) for s in raw.decode().splitlines() if '=' in s)
    headers = {'Authorization': 'Bearer ' + credential['password'], 'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'mcse-round1-backup'}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})).open

    def api(url, method='GET', body=None):
        assert url.startswith(API) or url == 'https://api.github.com/user'
        request = urllib.request.Request(url, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={**headers, 'Content-Type': 'application/json'})
        for attempt in range(3):
            try:
                with opener(request, timeout=60) as response:
                    data = response.read()
                    return json.loads(data) if data else None
            except (urllib.error.URLError, TimeoutError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code not in (408, 429):
                    raise
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    assert api('https://api.github.com/user')['login'] == 'chenciquan'
    assert api(API)['private'] is True
    try:
        release = api(API + '/releases/tags/' + TAG)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        release = api(API + '/releases', 'POST', {'tag_name': TAG, 'target_commitish': manifest['git_commit'],
            'name': 'G35 five-focus LoRA round one: full research artifacts', 'draft': True,
            'prerelease': False, 'body': 'Upload in progress; only this approved training/evaluation round.'})
    assert release['tag_name'] == TAG
    assert release['target_commitish'] == manifest['git_commit']
    existing = {a['name']: a for a in api(API + '/releases/' + str(release['id']) + '/assets?per_page=100')}
    payloads = [(CACHE / r['name'], r['sha256']) for r in manifest['archives']]
    payloads += [(CACHE / n, sha(CACHE / n)) for n in ('round1-file-manifest.json', 'ROUND1_SHA256SUMS.txt')]
    assert len(payloads) < 100 and not (set(existing) - {p.name for p, h in payloads})
    base = release['upload_url'].split('{')[0]
    assert base.startswith('https://uploads.github.com/repos/chenciquan/multichannel-speech-enhancement/')
    started = time.monotonic()

    def send(pair):
        path, expected = pair
        a = existing.get(path.name)
        if a is not None:
            assert a['state'] == 'uploaded' and a['size'] == path.stat().st_size and a.get('digest') == 'sha256:' + expected
        else:
            for attempt in range(3):
                def chunks():
                    with path.open('rb') as f:
                        yield from iter(lambda: f.read(1024 ** 2), b'')
                req = urllib.request.Request(base + '?name=' + urllib.parse.quote(path.name), data=chunks(),
                    method='POST', headers={**headers, 'Content-Type': 'application/octet-stream',
                                           'Content-Length': str(path.stat().st_size)})
                try:
                    with opener(req, timeout=120) as response:
                        a = json.loads(response.read())
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError) as error:
                    if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code not in (408, 422, 429):
                        raise
                    matches = [x for x in api(API + '/releases/' + str(release['id']) + '/assets?per_page=100')
                               if x['name'] == path.name]
                    if matches:
                        a = matches[0]
                        if a['state'] == 'uploaded':
                            break
                        assert release['draft'], 'Never replace an unfinished published asset'
                        api(API + '/releases/assets/' + str(a['id']), 'DELETE')
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        assert a['state'] == 'uploaded' and a['size'] == path.stat().st_size and a.get('digest') == 'sha256:' + expected
        print('Remote verified:', path.name, flush=True)
        return {'name': path.name, 'bytes': a['size'], 'sha256': expected, 'provider_digest': a['digest'],
                'url': a['browser_download_url'], 'id': a['id']}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        assets = list(pool.map(send, payloads))
    remote = {a['name']: a for a in api(API + '/releases/' + str(release['id']) + '/assets?per_page=100')}
    for a in assets:
        assert remote[a['name']]['digest'] == a['provider_digest'] and remote[a['name']]['size'] == a['bytes']
    notes = (f'Approved G35 five-focus LoRA first round. {manifest["file_count"]} files; '
        f'{len(manifest["archives"])} independent ZIPs. Six matched runs, 18000 optimizer updates, '
        '6384 enhanced outputs, complete three-seed/19-angle/six-group metrics and all negative results. '
        'Per-file and per-archive SHA256 plus local ZIP CRC verified. Extract all ZIPs into a NEW project clone; '
        'never overwrite an existing research workspace. Historical Release assets unchanged. '
        'Source G15/G35 input data and official McNet remain in the earlier full snapshot. '
        'No new ASR, no second ratio, no extra model. Private repository visibility unchanged.')
    published = api(API + '/releases/' + str(release['id']), 'PATCH', {
        'tag_name': TAG, 'target_commitish': manifest['git_commit'], 'draft': False, 'prerelease': False, 'body': notes})
    assert not published['draft'] and published['tag_name'] == TAG
    # Draft upload responses can contain temporary untagged download URLs.
    # Read the published resources back before recording stable restore links.
    published_assets = {a['name']: a for a in api(API + '/releases/' + str(release['id']) + '/assets?per_page=100')}
    for a in assets:
        r = published_assets[a['name']]
        assert r['state'] == 'uploaded' and r['size'] == a['bytes'] and r['digest'] == a['provider_digest']
        a['url'] = r['browser_download_url']
    receipt = {'status': 'complete', 'repository': 'chenciquan/multichannel-speech-enhancement', 'private': True,
        'release_url': published['html_url'], 'release_id': release['id'], 'tag': TAG,
        'git_snapshot_commit': manifest['git_commit'], 'files': manifest['file_count'],
        'source_bytes': manifest['source_bytes'], 'archive_bytes': manifest['archive_bytes'],
        'archives': len(manifest['archives']), 'assets': assets,
        'all_remote_sizes_and_provider_SHA256_verified': True, 'elapsed_upload_seconds': time.monotonic() - started,
        'local_manifest_sha256': sha(CACHE / 'round1-file-manifest.json')}
    save(CACHE / 'upload_receipt.json', receipt)
    print('Release complete:', published['html_url'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'upload'])
    args = p.parse_args()
    manifest = prepare()
    if args.action == 'upload':
        upload(manifest)
