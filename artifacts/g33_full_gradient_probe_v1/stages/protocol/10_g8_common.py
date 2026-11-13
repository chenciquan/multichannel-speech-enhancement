"""Shared G8 evidence helpers; existing artifacts are never overwritten."""
import hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'artifacts/g8_cross_asr_v1'
ASSET = ROOT / 'data/models/wav2vec2_base_960h_v1'
FILENAME = 'wav2vec2_fairseq_base_ls960_asr_ls960.pth'


def now(): return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def dump(p, x, exclusive=False):
    with Path(p).open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2); f.write('\n')


def status(**kw): dump(OUT/'status.json', {'utc': now(), **kw})


def archive(stage, report, files):
    folder = OUT/'stages'/stage; folder.mkdir(parents=True, exist_ok=False)
    entries = []
    for i, p in enumerate([report]+files):
        p = Path(p); dest = folder/f'{i:02d}_{p.name}'; shutil.copy2(p, dest)
        entries.append({'source': str(p.relative_to(ROOT)), 'snapshot': dest.name, 'sha256': sha(p)})
    dump(folder/'archive_manifest.json', {'saved_utc': now(), 'files': entries}, True)


def model_load():
    import torch, torchaudio
    meta = json.loads((OUT/'asset_provenance.json').read_text())
    path = ASSET/FILENAME
    if not path.is_file() or sha(path) != meta['local_sha256']:
        raise ValueError('CTC weight hash mismatch before model loading')
    if torchaudio.__version__ != '2.5.1+cu121': raise ValueError('Unregistered torchaudio version')
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    assert bundle._path == FILENAME and bundle.sample_rate == 16000 and not bundle._normalize_waveform
    model = bundle.get_model(dl_kwargs={'model_dir': str(ASSET), 'map_location': 'cpu', 'weights_only': True, 'progress': False})
    return model.eval().requires_grad_(False).cuda(), bundle.get_labels()
