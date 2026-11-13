"""Read-only verification of the 2026-11-13 private GitHub backup."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/github_upload_20261113'
REPO = 'chenciquan/multichannel-speech-enhancement'
API = 'https://api.github.com/repos/' + REPO


def main():
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'Never'}
    raw = subprocess.check_output(
        ['git', 'credential', 'fill'], cwd=ROOT, env=env,
        input=b'protocol=https\nhost=github.com\n\n')
    credential = dict(line.split('=', 1) for line in raw.decode().splitlines() if '=' in line)
    headers = {'Authorization': 'Bearer ' + credential['password'],
               'Accept': 'application/vnd.github+json', 'User-Agent': 'mcse-backup-verifier'}

    def get(url):
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=45) as response:
                    return json.load(response)
            except urllib.error.URLError as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                    raise
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))

    assert get('https://api.github.com/user')['login'] == 'chenciquan'
    repository = get(API)
    assert repository['private'] is True
    local_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    remote_head = get(API + '/git/ref/heads/main')['object']['sha']
    assert remote_head == local_head, 'Local HEAD and remote main differ'
    release = get(API + '/releases/tags/research-snapshot-g35-20261113')
    assert release['draft'] is False and release['prerelease'] is False
    tag = get(API + '/git/ref/tags/research-snapshot-g35-20261113')['object']
    while tag['type'] == 'tag':
        tag = get(API + '/git/tags/' + tag['sha'])['object']
    assert tag['type'] == 'commit'
    tag_commit = tag['sha']
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', tag_commit, local_head], cwd=ROOT)
    subprocess.check_call(['git', 'cat-file', '-e', tag_commit + ':scripts/release_transport.py'], cwd=ROOT)
    transport = json.loads((CACHE / 'release-transport-manifest.json').read_text(encoding='utf-8'))
    expected = {asset['name']: (asset['bytes'], asset['sha256'])
                for part in transport['parts'] for asset in part['assets']}
    for name in ('research-file-manifest.json', 'SHA256SUMS.txt', 'release-transport-manifest.json'):
        path = CACHE / name
        expected[name] = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    assets = get(API + '/releases/' + str(release['id']) + '/assets?per_page=100')
    assert {asset['name'] for asset in assets} == set(expected), 'Missing or unexpected release asset'
    verified = []
    for asset in assets:
        size, sha256 = expected[asset['name']]
        assert asset['state'] == 'uploaded' and asset['size'] == size
        assert asset.get('digest') == 'sha256:' + sha256
        verified.append({'name': asset['name'], 'bytes': size, 'sha256': sha256,
                         'provider_digest': asset['digest'], 'url': asset['browser_download_url']})
    result = {'status': 'passed', 'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'repository': REPO, 'private': True, 'local_and_remote_commit': local_head,
              'release_tag_commit': tag_commit, 'restoration_tool_exists_in_tag': True,
              'release_url': release['html_url'], 'draft': False, 'release_asset_count': len(assets),
              'research_payload_bytes': transport['archive_bytes'],
              'assets_bytes_with_manifests': sum(a['bytes'] for a in verified), 'assets': verified}
    (CACHE / 'remote_assets_verified.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'assets'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
