"""Authorized fixed five-focus round one: six matched runs and G35 development only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import gc
import hashlib
import importlib.metadata
import json
from multiprocessing import get_context
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'artifacts/g35_multifocus_round1_v1'
DATA = ROOT / 'data/g15_v1'
G35 = ROOT / 'data/g35_azimuth_scan_v1'
SEEDS = (20261112, 20261113, 20261114)
METRICS = ('si_sdr_db', 'pesq_wb', 'stoi')
CODE_FILES = ('scripts/g35_multifocus_r1.py', 'scripts/g35_multifocus_r1_sampling.py',
              'scripts/g35_multifocus_r1_summary.py', 'scripts/run_g35_multifocus_r1.ps1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def meta(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def dump(path, value, exclusive=False):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if exclusive:
        with path.open('x', encoding='utf-8') as f:
            f.write(data + '\n')
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(data + '\n', encoding='utf-8')
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(.1)


def status(stage, **fields):
    dump(OUT / 'status.json', {'stage': stage, 'utc': __import__('datetime').datetime.now(
        __import__('datetime').timezone.utc).isoformat(), **fields})


def guard(start, seconds):
    if time.monotonic() - start >= seconds - 120:
        raise TimeoutError('Stage reserve reached; no automatic retry/resume')
    if shutil.disk_usage(ROOT).free < 6 * 1024 ** 3:
        raise RuntimeError('Storage reserve reached')


def code_guard():
    plan = meta(OUT / 'execution_manifest.json')
    for p, h in plan['code_hashes'].items():
        assert sha(ROOT / p) == h, p
    assert sha(DATA / 'train_catalog.json') == plan['catalog_sha256']
    return plan


def configure(torch):
    assert str(torch.__version__) == '2.5.1+cu121'
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.cuda.is_available(), 'No CPU training fallback'
    return {'torch': str(torch.__version__), 'numpy': np.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'matmul_tf32': False, 'cudnn_tf32': False, 'cudnn_enabled': True,
            'benchmark': False, 'cudnn_deterministic': False,
            'global_deterministic': False, 'AMP': False, 'dtype': 'float32', 'CPU_threads': 2}


def digest(model, mode='all'):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if mode == 'base' and 'lora_' in name:
            continue
        if mode == 'adapter' and 'lora_' not in name:
            continue
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gpu_snapshot():
    return subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
        '--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu',
        '--format=csv,noheader'], text=True).strip()


def load_fresh(seed):
    import torch
    from mcse.runtime import load_system
    from mcse.adaptation import attach_lora
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    runtime = load_system('mcnet', 'cuda')
    assert attach_lora(runtime.model, rank=4, alpha=4) == 3584
    shapes = {n: list(p.shape) for n, p in runtime.model.named_parameters() if p.requires_grad}
    expected = {'freq.fc_output_layer.lora_A': [4, 256], 'freq.fc_output_layer.lora_B': [64, 4],
                'narr.fc_output_layer.lora_A': [4, 512], 'narr.fc_output_layer.lora_B': [64, 4]}
    assert shapes == expected
    assert sum(p.numel() for p in runtime.model.parameters()) == 3332290
    runtime.metadata = {**runtime.metadata, 'adapter_parameters': 3584, 'initial_seed': seed,
                        'initialization': 'fresh official McNet; zero B, seeded A; no old adapter warmstart'}
    return runtime


def schedule_description(catalog, indices):
    from g35_multifocus_r1_sampling import group_of, stratum, GROUPS
    rows = [catalog[int(i)] for i in indices]
    counts = Counter(map(int, indices))
    return {'rows': len(rows), 'unique_indices': len(counts), 'max_repetition': max(counts.values()),
            'target_group_counts': {GROUPS[g]: sum(group_of(r['target_azimuth']) == g for r in rows) for g in range(6)},
            'field_counts': dict(Counter(r['field'] for r in rows)),
            'noise_counts': dict(Counter(r['noise_id'] for r in rows)),
            'SNR_counts': dict(Counter(str(r['snr_db']) for r in rows)),
            'speaker_counts': dict(Counter(str(r['speaker_id']) for r in rows)),
            'source_counts': dict(Counter(r['source_id'] for r in rows)),
            'stratum_counts': dict(Counter(json.dumps(stratum(r)) for r in rows)),
            'target_histogram_10deg': {str(lo): sum(lo <= r['target_azimuth'] < lo + 10 for r in rows)
                                       for lo in range(-45, 75, 10)}}


def prepare():
    import torch
    from g35_multifocus_r1_sampling import (unit_checks, original_schedule, focused_schedule,
                                           index_sha, GROUPS, group_of)
    assert not OUT.exists(), 'Independent directory exists; no overwrite or implicit resume'
    OUT.mkdir()
    status('preflight', check='source_and_sampling')
    assert shutil.disk_usage(ROOT).free >= 12 * 1024 ** 3
    assert not subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
        '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip(), 'Another compute job exists'
    frozen = meta(ROOT / 'artifacts/g15_rigid_training_v1/PIPELINE_READY.json')['files']
    assert all(sha(ROOT / p) == h for p, h in frozen.items())
    expected_data = meta(ROOT / 'artifacts/g15_rigid_training_v1/data_file_hashes.json')
    for name in ('train_catalog.json', 'train_x.npy', 'train_y.npy', 'clean_x.npy'):
        p = DATA / name
        assert sha(p) == expected_data[str(p.relative_to(ROOT))]
    done = meta(ROOT / 'artifacts/g35_azimuth_scan_v1/DATA_COMPLETE.json')
    assert done['status'] == 'passed' and sha(G35 / 'manifest.jsonl') == done['manifest_sha256']
    cases = read(G35 / 'manifest.jsonl')
    assert len(cases) == 1064 and sum(r['condition'] == 'clean' for r in cases) == 152
    for r in cases:
        assert sha(r['noisy_path']) == r['input_sha256'] and sha(r['reference_path']) == r['reference_sha256']
    catalog = meta(DATA / 'train_catalog.json')
    assert len(catalog) == 2048 and all(r['geometry_policy'] == 'rigid_rotation' for r in catalog)
    assert all(r['index'] == i and r['legacy_paired_index'] == i for i, r in enumerate(catalog))
    for r in catalog:
        assert -45 <= r['target_azimuth'] < 75
    xmap = np.load(DATA / 'train_x.npy', mmap_mode='r')
    ymap = np.load(DATA / 'train_y.npy', mmap_mode='r')
    assert xmap.shape == (2048, 6, 16128) and ymap.shape == (2048, 16128)
    assert xmap.dtype == ymap.dtype == np.dtype('float32') and not xmap.flags.writeable
    from mcse.geometry import rotate_azimuth
    from mcse.spatial_noise import direction
    dot_error = 0.
    for r in catalog:
        for u, actual in zip(r['base_noise_directions'], r['actual_noise_directions']):
            wanted = rotate_azimuth(u, r['target_azimuth'] - 20)
            assert np.array_equal(wanted, np.array(actual))
            dot_error = max(dot_error, abs(np.dot(actual, direction(r['target_azimuth'], 10)) - np.dot(u, direction(20, 10))))
    assert dot_error <= 1e-12
    protected_paths = set(frozen) | {str(p.relative_to(ROOT)) for p in DATA.glob('*.npy')}
    protected_paths |= {'data/g15_v1/train_catalog.json', 'data/g35_azimuth_scan_v1/manifest.jsonl',
        'src/mcse/runtime.py', 'src/mcse/evaluation.py', 'scripts/mcnet_infer.py',
        'artifacts/g35_azimuth_scan_v1/signals.jsonl',
        'third_party/McNet/model_checkpoints/offline/epoch494_criteria18.78_sdr18.78.ckpt'}
    protected_paths |= {str(p.relative_to(ROOT)) for p in (ROOT / 'docs').glob('*.md') if p.name != 'CODEX_PROJECT_MEMORY.md'}
    selected = meta(ROOT / 'artifacts/g35_admission_v1/selected_registry.json')
    for key, record in selected.items():
        if key != 'mcnet':
            assert sha(ROOT / record['path']) == record['checkpoint_sha256']
            protected_paths.add(record['path'])
    protected = {p: sha(ROOT / p) for p in sorted(protected_paths)}
    dump(OUT / 'protected_before.json', protected, True)
    shutil.copyfile(ROOT / 'docs/CODEX_PROJECT_MEMORY.md', OUT / 'CODEX_PROJECT_MEMORY_before_round1.md')
    plan = {'authorization': 'User approved exactly A/B, 3 seeds each, 3000 updates; G35 19-angle signal/clean evaluation; no ASR/search',
        'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'centers': [-40, 0, 10, 30, 70], 'half_width': 5, 'intervals': 'left-closed/right-open; disjoint; complement is background',
        'B_group_steps': dict(zip(GROUPS, [300] * 5 + [1500])), 'A_sampler': 'exact original G15 index algorithm',
        'categorical_matching': 'Per-step A/B same field, noise_id, SNR, and point baseline noise angle; multi8 angle label ignored as nonphysical',
        'allocation': 'Integer network transport minimizing squared departure from baseline conditional class proportions; no neural loss change',
        'group_field_quota': 'Largest remainder of A observed point fraction; exact global field counts; no forced ideal 50/50 for A',
        'initialization': 'Fresh official checkpoint plus same-seed initial LoRA for each A/B pair; no continuation',
        'seeds': list(SEEDS), 'steps_per_run': 3000, 'runs': 6, 'batch_size': 1, 'gradient_accumulation': 1,
        'learning_rate': .0003, 'optimizer': 'Adam defaults', 'gradient_clip_norm': 5., 'identity_weight': 0.,
        'loss': 'unchanged mcse.adaptation.mask_loss compressed cIRM MSE', 'rank': 4, 'alpha': 4, 'parameters': 3584,
        'checkpoints': 'save each500; uniformly register step3000, no metric selection; no intermediate validation forwards',
        'train_max_seconds_per_run': 1800, 'evaluation_max_seconds': 5400, 'smoke_max_seconds': 600,
        'new_artifact_limit_bytes': 6 * 1024 ** 3, 'catalog_sha256': sha(DATA / 'train_catalog.json'),
        'G35_manifest_sha256': sha(G35 / 'manifest.jsonl'),
        'code_hashes': {p: sha(ROOT / p) for p in CODE_FILES},
        'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pesq', 'pystoi', 'torch')},
        'GPU_before': gpu_snapshot(), 'no_retries_of_training_or_metrics': True}
    assert plan['versions'] == {'numpy': '1.26.4', 'scipy': '1.11.4', 'pesq': '0.0.4', 'pystoi': '0.4.1', 'torch': '2.5.1+cu121'}
    dump(OUT / 'execution_manifest.json', plan, True)
    sampler_checks = unit_checks()
    (OUT / 'sampling').mkdir()
    registry = {r['seed']: r for r in meta(ROOT / 'artifacts/g15_rigid_training_v1/registered_checkpoints.json')['systems']}
    schedule_audits = []
    for seed in SEEDS:
        a, state = original_schedule(catalog, seed)
        prior = ROOT / registry[seed]['path']
        assert sha(prior) == registry[seed]['checkpoint_sha256']
        saved = torch.load(prior, map_location='cpu', weights_only=True)
        assert saved['numpy_sampler_state'] == state and saved['step'] == 3000
        old_result = meta(prior.parent / 'result.json')
        assert sum(catalog[int(i)]['field'] == 'point' for i in a) == old_result['sampler_counts']['point']
        b, groups, details = focused_schedule(catalog, a, seed)
        for arm, indices in [('A', a), ('B', b)]:
            np.save(OUT / 'sampling' / f'{arm}_s{seed}_indices.npy', indices, allow_pickle=False)
            description = schedule_description(catalog, indices)
            dump(OUT / 'sampling' / f'{arm}_s{seed}.json', {'index_sha256_i64le': index_sha(indices),
                'description': description, 'original_rng_final': state if arm == 'A' else None,
                'allocation': details if arm == 'B' else None}, True)
        schedule_audits.append({'seed': seed, 'A': schedule_description(catalog, a),
                               'B': schedule_description(catalog, b), 'strata_exact_per_step': True})
    dump(OUT / 'sampler_checks.json', {'status': 'passed', 'unit': sampler_checks,
         'dot_error': dot_error, 'seeds': schedule_audits, 'old_sampler_final_states_match': True}, True)
    status('preflight', check='GPU_smoke')
    smoke_start = time.monotonic(); backend = configure(torch)
    runtime = load_fresh(SEEDS[0]); base_before = digest(runtime.model, 'base')
    runtime.model.train(); torch.cuda.reset_peak_memory_stats()
    from mcse.adaptation import mask_loss
    x = torch.from_numpy(np.array(xmap[0], copy=True))[None].cuda()
    target = torch.from_numpy(np.array(ymap[0], copy=True))[None].cuda()
    pred, data = runtime.model(x); loss = mask_loss(runtime.model, x, target, pred, data)
    assert torch.isfinite(loss); loss.backward()
    grads = [p.grad for p in runtime.model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) for g in grads)
    assert all(p.grad is None for p in runtime.model.parameters() if not p.requires_grad)
    runtime.model.zero_grad(set_to_none=True); runtime.model.eval()
    from mcse.runtime import enhance_array
    audio, sr = sf.read(cases[0]['noisy_path'], dtype='float32', always_2d=True)
    y, _ = enhance_array(runtime, audio)
    assert sr == 16000 and y.shape == (len(audio),) and np.isfinite(y).all()
    assert digest(runtime.model, 'base') == base_before
    guard(smoke_start, 600)
    smoke = {'status': 'passed', 'backend': backend, 'cirm_loss': float(loss.detach()),
        'training_forward_calls': 1, 'backward_calls': 1, 'optimizer_updates': 0, 'enhancement_calls': 1,
        'seconds': time.monotonic() - smoke_start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'GPU': gpu_snapshot(),
        'base_unchanged': True, 'input_channels': 6, 'output_channels': 1}
    dump(OUT / 'preflight.json', smoke, True)
    del runtime, x, target, pred, data, loss, grads, y, audio
    gc.collect(); torch.cuda.empty_cache()
    status('ready_for_authorized_training', preflight_passed=True)


def train_one(arm, seed):
    import torch
    from mcse.adaptation import mask_loss, adapter_state
    from g35_multifocus_r1_sampling import index_sha, group_of, GROUPS
    plan = code_guard(); assert meta(OUT / 'preflight.json')['status'] == 'passed'
    folder = OUT / 'train' / f'{arm}_s{seed}'; folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); backend = configure(torch)
    catalog = meta(DATA / 'train_catalog.json')
    indices = np.load(OUT / 'sampling' / f'{arm}_s{seed}_indices.npy', allow_pickle=False)
    schedule = meta(OUT / 'sampling' / f'{arm}_s{seed}.json')
    assert index_sha(indices) == schedule['index_sha256_i64le']
    xmap = np.load(DATA / 'train_x.npy', mmap_mode='r'); ymap = np.load(DATA / 'train_y.npy', mmap_mode='r')
    runtime = load_fresh(seed); model = runtime.model; model.train()
    base_before = digest(model, 'base')
    initial = {'base_digest': base_before, 'adapter_digest': digest(model, 'adapter'),
        'CPU_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        'CUDA_rng_sha256': hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
    if arm == 'B':
        assert initial == meta(OUT / 'train' / f'A_s{seed}' / 'initial_state.json'), 'A/B initial states differ'
    dump(folder / 'initial_state.json', initial, True)
    torch.save({'adapter': adapter_state(model), 'seed': seed, 'initial': initial}, folder / 'initial_adapter.pt')
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.0003)
    dump(folder / 'config.json', {'arm': arm, 'seed': seed, 'steps': 3000, 'schedule_sha256': index_sha(indices),
        'initial': initial, 'backend': backend, 'optimizer_defaults': optimizer.defaults,
        'loss_source_sha256': sha(ROOT / 'src/mcse/adaptation.py'), 'execution_manifest_sha256': sha(OUT / 'execution_manifest.json')}, True)
    torch.cuda.reset_peak_memory_stats(); resources = []
    with (folder / 'steps.jsonl').open('x', encoding='utf-8') as log, (folder / 'samples.jsonl').open('x', encoding='utf-8') as samplelog:
        for step, idx in enumerate(indices, 1):
            guard(start, plan['train_max_seconds_per_run']); r = catalog[int(idx)]
            xx = np.array(xmap[int(idx)], copy=True); ss = np.array(ymap[int(idx)], copy=True)
            assert np.isfinite(xx).all() and np.isfinite(ss).all()
            x = torch.from_numpy(xx)[None].cuda(); target = torch.from_numpy(ss)[None].cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=False):
                pred, data = model(x)
                loss = mask_loss(model, x, target, pred, data)
            assert torch.isfinite(loss), f'Nonfinite loss {arm}/{seed}/{step}'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            optimizer.step()
            record = {'step': step, 'cirm_loss': float(loss.detach()), 'grad_norm_before_clip': float(norm),
                      'elapsed_seconds': time.monotonic() - start}
            log.write(json.dumps(record, allow_nan=False) + '\n')
            sample = {'step': step, 'catalog_index': int(idx), 'target_group': GROUPS[group_of(r['target_azimuth'])],
                      **{k: r[k] for k in ('target_azimuth', 'source_id', 'speaker_id', 'source_crop_start',
                                           'noise_id', 'noise_intervals', 'snr_db', 'field', 'angle', 'common_scale')}}
            samplelog.write(json.dumps(sample, allow_nan=False) + '\n')
            if step % 25 == 0:
                log.flush(); samplelog.flush()
                status('training', arm=arm, seed=seed, **record)
            if step % 500 == 0:
                assert digest(model, 'base') == base_before
                checkpoint = {'adapter': adapter_state(model), 'optimizer': optimizer.state_dict(),
                    'step': step, 'seed': seed, 'arm': arm, 'initial': initial,
                    'base_checkpoint_sha256': runtime.metadata['base_checkpoint_sha256'],
                    'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state(),
                    'python_rng_state': random.getstate(), 'numpy_global_state': [np.random.get_state()[0],
                         np.random.get_state()[1].tolist(), *np.random.get_state()[2:]],
                    'sampling_state': {'schedule_sha256_i64le': index_sha(indices), 'next_position': step,
                                       'generator_details': schedule}}
                torch.save(checkpoint, folder / f'adapter_{step:05d}.pt')
                resources.append({'step': step, 'elapsed_seconds': time.monotonic() - start,
                                  'GPU': gpu_snapshot(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
                dump(folder / 'resources.json', resources)
                print(f'TRAIN {arm} seed={seed} step={step}/3000 loss={record["cirm_loss"]:.6f}', flush=True)
    actual = read(folder / 'samples.jsonl')
    assert len(actual) == 3000 and [r['catalog_index'] for r in actual] == indices.tolist()
    if arm == 'B':
        baseline = read(OUT / 'train' / f'A_s{seed}' / 'samples.jsonl')
        assert all(all(a[k] == b[k] for k in ('field', 'noise_id', 'snr_db')) and
                   (a['field'] != 'point' or a['angle'] == b['angle']) for a, b in zip(baseline, actual))
        assert Counter(r['target_group'] for r in actual) == Counter(dict(zip(GROUPS, [300] * 5 + [1500])))
    result = {'status': 'passed', 'arm': arm, 'seed': seed, 'completed_updates': 3000,
        'registered_step': 3000, 'checkpoint': str((folder / 'adapter_03000.pt').relative_to(ROOT)),
        'checkpoint_sha256': sha(folder / 'adapter_03000.pt'), 'base_digest_before': base_before,
        'base_digest_after': digest(model, 'base'), 'initial': initial, 'seconds': time.monotonic() - start,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'actual_sampling': schedule_description(catalog, indices), 'per_step_strata_match_A': arm == 'B',
        'samples_sha256': sha(folder / 'samples.jsonl'), 'steps_sha256': sha(folder / 'steps.jsonl')}
    assert result['base_digest_after'] == base_before
    dump(folder / 'result.json', result, True)
    del checkpoint, runtime, model, optimizer, x, target, pred, data, loss
    gc.collect(); torch.cuda.empty_cache()
    return result


_metric_evaluator = None


def score_output(row):
    global _metric_evaluator
    from mcse.evaluation import MetricEvaluator, waveform_diagnostics
    if _metric_evaluator is None:
        _metric_evaluator = MetricEvaluator()
    reference, sr = sf.read(row['reference_path'], dtype='float32')
    output, osr = sf.read(row['output_path'], dtype='float32')
    assert sr == osr == 16000 and output.shape == reference.shape
    row['metrics'] = _metric_evaluator.evaluate(reference, output)
    row['waveform_diagnostics'] = waveform_diagnostics(reference, output)
    row['status'] = 'ok' if all(v['status'] == 'ok' for v in row['metrics']['status'].values()) else 'metric_failure'
    if row['condition'] == 'clean':
        s = reference.astype(np.float64); y = output.astype(np.float64)
        row['clean_fidelity'] = {'NMSE': float(np.sum((y - s) ** 2) / np.sum(s * s)),
                                'RMS_gain': float(np.sqrt(np.sum(y * y) / np.sum(s * s)))}
    return row


def evaluate():
    import torch
    from mcse.adaptation import load_adapter
    from mcse.runtime import enhance_array
    plan = code_guard(); configure(torch)
    assert not (OUT / 'signals.jsonl').exists(), 'No implicit evaluation retry/resume'
    assert meta(OUT / 'TRAIN_COMPLETE.json')['updates'] == 18000
    assert sha(G35 / 'manifest.jsonl') == plan['G35_manifest_sha256']
    cases = read(G35 / 'manifest.jsonl')
    start = time.monotonic(); scored = 0; resources = []
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool, (OUT / 'signals.jsonl').open('x', encoding='utf-8') as log:
        pending = set()

        def collect(block):
            nonlocal pending, scored
            if not pending:
                return
            ready, pending = wait(pending, timeout=60 if block else 0, return_when=FIRST_COMPLETED)
            guard(start, plan['evaluation_max_seconds'])
            for future in ready:
                row = future.result()
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); log.flush()
                scored += 1
                if row['status'] != 'ok':
                    raise RuntimeError('Metric failure saved; no replacement or tolerance relaxation')

        for arm in ('A', 'B'):
            for seed in SEEDS:
                guard(start, plan['evaluation_max_seconds'])
                began = time.monotonic(); system = f'{arm}_s{seed}'
                result = meta(OUT / 'train' / system / 'result.json')
                checkpoint = ROOT / result['checkpoint']
                assert result['status'] == 'passed' and sha(checkpoint) == result['checkpoint_sha256']
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                runtime = load_fresh(seed)
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                assert saved['step'] == 3000 and saved['arm'] == arm and saved['seed'] == seed
                load_adapter(runtime.model, saved['adapter']); del saved
                runtime.model.eval().requires_grad_(False)
                runtime.metadata.update(system_id=system, adapter_path=result['checkpoint'],
                    adapter_sha256=result['checkpoint_sha256'], registered_step=3000)
                before = digest(runtime.model)
                folder = OUT / 'audio' / system; folder.mkdir(parents=True, exist_ok=False)
                for number, case in enumerate(cases, 1):
                    guard(start, plan['evaluation_max_seconds'])
                    assert sha(case['noisy_path']) == case['input_sha256']
                    assert sha(case['reference_path']) == case['reference_sha256']
                    x, sr = sf.read(case['noisy_path'], dtype='float32', always_2d=True)
                    assert sr == 16000 and x.shape[1] == 6
                    y, elapsed = enhance_array(runtime, x)
                    assert y.shape == (len(x),) and np.isfinite(y).all()
                    path = folder / (case['sample_id'] + '.wav')
                    with path.open('xb') as f:
                        sf.write(f, y, 16000, format='WAV', subtype='FLOAT')
                    saved_y, saved_sr = sf.read(path, dtype='float32')
                    assert saved_sr == 16000 and np.array_equal(y, saved_y)
                    pending.add(pool.submit(score_output, {**case, 'system': system, 'arm': arm, 'seed': seed,
                        'output_path': str(path), 'output_sha256': sha(path), 'inference_seconds': elapsed,
                        'checkpoint_sha256': result['checkpoint_sha256']}))
                    while len(pending) >= 8:
                        collect(True)
                    if number % 100 == 0:
                        status('evaluation', system=system, input_done=number, scored=scored,
                               elapsed_seconds=time.monotonic() - start)
                while pending:
                    collect(True)
                assert digest(runtime.model) == before, 'Inference changed model state'
                resources.append({'system': system, 'outputs': 1064, 'seconds_with_metrics': time.monotonic() - began,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'GPU': gpu_snapshot(), 'runtime': runtime.metadata, 'state_unchanged': True})
                dump(OUT / 'evaluation_resources.json', resources)
                print(f'EVAL {system} finished; scored={scored}/6384', flush=True)
                del runtime, x, y, saved_y
                gc.collect(); torch.cuda.empty_cache()
    assert scored == 6384
    dump(OUT / 'EVALUATION_COMPLETE.json', {'status': 'passed', 'outputs': 6384, 'signal_rows': scored,
        'noisy_rows': 5472, 'clean_rows': 912, 'systems': 6, 'seconds': time.monotonic() - start,
        'signals_sha256': sha(OUT / 'signals.jsonl'), 'ASR_run': False}, True)


def delivery_audit():
    start = time.monotonic(); plan = code_guard()
    protected = meta(OUT / 'protected_before.json')
    assert all(sha(ROOT / p) == h for p, h in protected.items()), 'Historical protected file changed'
    cases = {r['sample_id']: r for r in read(G35 / 'manifest.jsonl')}
    rows = read(OUT / 'signals.jsonl')
    systems = [f'{arm}_s{seed}' for arm in ('A', 'B') for seed in SEEDS]
    assert len(rows) == 6384 and len({(r['system'], r['sample_id']) for r in rows}) == 6384
    assert Counter(r['system'] for r in rows) == Counter({s: 1064 for s in systems})
    for row in rows:
        assert row['status'] == 'ok' and sha(row['output_path']) == row['output_sha256']
        case = cases[row['sample_id']]
        for key in ('input_sha256', 'reference_sha256', 'target_azimuth', 'common_scale', 'noise_gain'):
            assert row[key] == case[key]
        assert all(np.isfinite(row['metrics'][key]) for key in METRICS)
    for case in cases.values():
        assert sha(case['noisy_path']) == case['input_sha256'] and sha(case['reference_path']) == case['reference_sha256']
    for system in systems:
        folder = OUT / 'train' / system
        steps = read(folder / 'steps.jsonl'); samples = read(folder / 'samples.jsonl')
        assert [r['step'] for r in steps] == list(range(1, 3001)) and len(samples) == 3000
        for angle in range(-90, 91, 10):
            cell = [r for r in rows if r['system'] == system and r['target_azimuth'] == angle]
            assert len(cell) == 56 and sum(r['condition'] == 'clean' for r in cell) == 8
    size = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert size <= plan['new_artifact_limit_bytes']
    audit = {'status': 'passed', 'seconds': time.monotonic() - start, 'runs': 6, 'optimizer_updates': 18000,
        'evaluation_outputs': 6384, 'signals_rows': len(rows), 'all_19_angles_all_6_models': True,
        'all_input_reference_output_hashes_verified': True, 'protected_files': len(protected),
        'protected_files_unchanged': True, 'output_bytes_before_summary': size,
        'ASR_executed': False, 'parameter_search': False, 'G34_resumed': False}
    dump(OUT / 'DELIVERY_AUDIT.json', audit, True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run'])
    parser.parse_args()
    started = time.monotonic()
    try:
        prepare()
        results = []
        for seed in SEEDS:
            for arm in ('A', 'B'):
                results.append(train_one(arm, seed))
        assert sum(r['completed_updates'] for r in results) == 18000
        dump(OUT / 'TRAIN_COMPLETE.json', {'status': 'passed', 'runs': results, 'updates': 18000}, True)
        evaluate()
        audit = delivery_audit()
        from g35_multifocus_r1_summary import summarize
        summarize()
        code_guard()
        status('complete', optimizer_updates=18000, evaluation_outputs=6384,
               seconds=time.monotonic() - started, ASR_run=False)
        dump(OUT / 'ROUND1_COMPLETE.json', {'status': 'passed', 'seconds': time.monotonic() - started,
            'audit': audit, 'report': 'docs/G35_五重点方位LoRA首轮匹配训练与19角开发评价报告.md'}, True)
        print('ROUND1 COMPLETE; no ASR, no second ratio, no further training', flush=True)
    except BaseException:
        if OUT.exists():
            failure = {'status': 'failed', 'seconds': time.monotonic() - started,
                       'traceback': traceback.format_exc(), 'no_automatic_retry': True}
            dump(OUT / 'FAILED.json', failure, exclusive=not (OUT / 'FAILED.json').exists())
            status('failed', **failure)
        raise


if __name__ == '__main__':
    main()
