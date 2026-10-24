"""Shared one-command evaluation for independent model repositories.

Artifacts and receipts live outside Git. Model repositories provide inference,
immutable artifact declarations and an optional model-specific export hook.
"""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

from motion_challenge.data.dataset import audit_dataset, fetch_dataset
from motion_challenge.resources import lafan_paths
from motion_challenge.runtime.progress import Progress, copy_download
from motion_challenge.validation import external_output_path, sha256_file
from motion_challenge.evaluation.verify import verify
from motion_challenge.evaluation.paired import combine


def artifact(path, expected_hash, url=None, offline=False, progress=None):
    path = Path(path)
    progress = progress or Progress()
    if path.is_file():
        if sha256_file(path) != expected_hash:
            raise ValueError(f'checkpoint checksum mismatch: {path}')
        progress.emit('checkpoint_verified', force=True, file=str(path), bytes=path.stat().st_size, cached=True)
        return path
    if offline or not url:
        raise FileNotFoundError(f'checkpoint missing: {path}; supply --weights or enable download')
    if not url.startswith('https://'):
        raise ValueError('checkpoint downloads require HTTPS')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                copy_download(response, stream, path.name, progress)
        if sha256_file(temporary) != expected_hash:
            raise ValueError('downloaded checkpoint checksum mismatch')
        temporary.replace(path)
        progress.emit('checkpoint_verified', force=True, file=str(path), bytes=path.stat().st_size, cached=False)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_snapshot(root):
    """Bind adapter source and dependency lock to the result without copying data."""
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'], text=True).split('\0')
    files = {name: sha256_file(root/name) for name in names
             if name and (root/name).is_file() and name != 'motion-challenge'}
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    core = root/'motion-challenge'
    return {'adapter_commit': git('rev-parse', 'HEAD'),
            'adapter_dirty': bool(git('status', '--porcelain', '--untracked-files=normal')),
            'tracked_files_sha256': files,
            'core_commit': subprocess.check_output(['git', '-C', str(core), 'rev-parse', 'HEAD'], text=True).strip(),
            'core_dirty': bool(subprocess.check_output(['git', '-C', str(core), 'status', '--porcelain'], text=True).strip())}


def resolve_artifacts(runtime, root, overrides=(), offline=False, progress=None):
    """Resolve additional model components without model-specific runner logic."""
    declarations = runtime.get('artifacts', {})
    selected = {}
    for override in overrides:
        name, separator, path = override.partition('=')
        if not separator or not path or name not in declarations or name in selected:
            raise ValueError('artifact overrides require unique declared NAME=PATH values')
        selected[name] = Path(path).expanduser().resolve()
    resolved = {}
    for name, item in declarations.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
            raise ValueError('invalid artifact name or SHA256')
        local = root/item['path']
        path = selected.get(name, local if local.is_file() else
            Path.home()/'.cache/motion-challenge/checkpoints'/item['sha256']/Path(item['path']).name)
        path = artifact(path, item['sha256'], item.get('url'), offline, progress)
        resolved[name] = {**item, 'path': str(path.resolve())}
    return resolved


def main(adapter_root=None):
    p = argparse.ArgumentParser(description='Evaluate a checkpoint with the shared benchmark core.')
    p.add_argument('--suite', default='lafan40', choices=['lafan40'])
    p.add_argument('--data', type=Path, default=Path(os.environ.get('MOTION_CHALLENGE_DATA', '~/.cache/motion-challenge/lafan')))
    p.add_argument('--weights', type=Path)
    p.add_argument('--artifact', action='append', default=[], metavar='NAME=PATH',
                   help='Use a cached additional model component; repeat for multi-file policies')
    p.add_argument('--output', type=Path, help='New directory outside any Git repository')
    p.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument('--backend', choices=['batch'], default='batch', help=argparse.SUPPRESS)
    p.add_argument('--termination', choices=['both', 'early', 'none'], default='both',
                   help='Compute both metric scopes from one full rollout (default), or select a diagnostic mode')
    p.add_argument('--threads', type=int, help='MuJoCo CPU threads; defaults to batch slots')
    p.add_argument('--progress-interval', type=float, default=5.0)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--max-clips', type=int)
    p.add_argument('--max-duration', type=float)
    a = p.parse_args()
    if a.jobs < 1 or (a.max_clips is not None and a.max_clips < 1):
        p.error('jobs and max-clips must be positive')
    if a.progress_interval <= 0 or (a.threads is not None and a.threads < 1):
        p.error('progress interval and threads must be positive')
    root = Path(adapter_root or Path.cwd()).resolve()
    original = root/'runtime.json'
    runtime = json.loads(original.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = external_output_path((a.output or Path.home()/'.local/share/motion-challenge/runs'/runtime['id']/stamp).expanduser())
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be new or empty; existing evidence is never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    progress = Progress(output/'progress.json', a.progress_interval)
    a.data = external_output_path(a.data.expanduser())
    paths = lafan_paths('early')
    manifest = json.loads(paths['suite'].read_text())
    if not a.offline:
        print(json.dumps(fetch_dataset(manifest, a.data, progress)), flush=True)
    audit = audit_dataset(manifest, a.data)
    progress.emit('dataset_audited', force=True, completed=audit['clip_count'], total=len(manifest['files']))
    if 'weights_sha256' in runtime:
        local = root/runtime.get('weights', 'checkpoints/policy.onnx')
        weights = (a.weights.expanduser().resolve() if a.weights else local if local.is_file()
                   else Path.home()/'.cache/motion-challenge/checkpoints'/runtime['weights_sha256']/'policy.onnx')
        if not weights.exists() and not a.offline and runtime.get('prepare_checkpoint'):
            sys.path.insert(0, str(root))
            module, function = runtime['prepare_checkpoint'].split(':')
            getattr(importlib.import_module(module), function)(weights)
        runtime['weights'] = str(artifact(weights, runtime['weights_sha256'], runtime.get('weights_url'), a.offline, progress).resolve())
    if runtime.get('artifacts') or a.artifact:
        runtime['artifacts'] = resolve_artifacts(runtime, root, a.artifact, a.offline, progress)
    runtime['adapter_root'] = str(root)
    before = source_snapshot(root)
    if a.termination != 'both':
        run_mode(root, a, runtime, output, a.termination, before, stamp)
        return
    run_pair(root, a, runtime, output, before, stamp)


def run_pair(root, args, runtime, output, source, stamp):
    """Roll out once; verification computes both metric scopes from that trace."""
    record = {'schema': 'motion-challenge/paired-evaluation-v1', 'status': 'running',
              'execution': 'single_full_rollout', 'started_utc': stamp, 'source': source,
              'checkpoint_sha256': runtime.get('weights_sha256'), 'modes': {}}
    summary_path = output/'summary.json'
    def save():
        temporary = summary_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(summary_path)
    save()
    destination = output/'rollout'
    destination.mkdir()
    try:
        receipt = run_mode(root, args, runtime, destination, 'none', source, stamp, derive_prefix=True)
        record['status'] = receipt['status']
        full_path = destination/'result.json'
        prefix_path = destination/'prefix-result.json'
        for mode, path in [('early', prefix_path), ('none', full_path)]:
            result = json.loads(path.read_text())
            record['modes'][mode] = {'status': receipt['status'], 'receipt': 'rollout/receipt.json',
                                     'result': 'rollout/'+path.name,
                                     'success_count': result['success_count'], 'macro_completion': result['macro_completion'],
                                     'metrics': result['macro_metrics_over_observed_prefix']}
        if receipt['status'] == 'trace_verified':
            combined = combine(json.loads(prefix_path.read_text()), json.loads(full_path.read_text()), source, runtime['weights_sha256'])
            combined['execution'] = 'single_full_rollout'
            combined['components'] = {mode: {
                'result': item['result'], 'result_sha256': sha256_file(output/item['result']),
                'receipt': item['receipt'], 'receipt_sha256': sha256_file(output/item['receipt'])
            } for mode, item in record['modes'].items()}
            (output/'result.json').write_text(json.dumps(combined, indent=2)+'\n')
            record['result'] = 'result.json'
    except Exception as error:
        record.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({'status': record['status'], 'summary': str(summary_path)}), flush=True)
    return record


def run_mode(root, a, runtime, output, mode, before, stamp, derive_prefix=False):
    paths = lafan_paths(mode)
    manifest = json.loads(paths['suite'].read_text())
    progress = Progress(output/'progress.json', a.progress_interval)
    resolved = output/'runtime.json'
    resolved.write_text(json.dumps(runtime, indent=2)+'\n')
    receipt = {'submitter': 'local-run', 'source': before,
               'source_runtime_sha256': sha256_file(root/'runtime.json'), 'resolved_runtime_sha256': sha256_file(resolved),
               'suite': a.suite, 'started_utc': stamp, 'author_inference_review': 'not_granted_by_runner',
               'output': str(output), 'status': 'running', 'backend': a.backend,
               'termination_mode': 'early_termination' if mode == 'early' else 'no_termination',
               'batch_size': a.jobs}
    receipt_path = output/'receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    selected = manifest['files'][:a.max_clips] if a.max_clips else manifest['files']
    count = min(a.jobs, len(selected))
    common = ['--controller', str(resolved), '--data', str(a.data)]
    for key, value in paths.items():
        common += ['--'+key, str(value)]

    def evaluate(item):
        index, clips = item
        result = output/f'shard-{index:02d}.json'
        command = [sys.executable, '-m', 'motion_challenge.run', *common, '--output', str(result),
                   '--backend', a.backend, '--batch-size', str(count),
                   '--progress-interval', str(a.progress_interval)]
        if a.threads is not None:
            command += ['--threads', str(a.threads)]
        for clip in clips:
            command += ['--clip-id', clip['path']]
        if a.max_duration is not None:
            command += ['--max-duration', str(a.max_duration)]
        with (output/f'shard-{index:02d}.log').open('w') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        print(f'Finished shard {index+1}/{count}', flush=True)
        return result

    run_started = time.monotonic()
    try:
        results = [evaluate((0, selected))]
        rollout_finished = time.monotonic()
        progress.emit('verification', force=True, status='started')
        partial = a.max_clips is not None or a.max_duration is not None
        # Both complete and smoke records use the same verified path.
        shutil.copyfile(results[0], output/'result.json')
        options = ({'prefix_result_path': output/'prefix-result.json',
                    'prefix_protocol_path': lafan_paths('early')['protocol']} if derive_prefix else {})
        check = verify(output/'result.json', paths['suite'], paths['robot'], paths['protocol'], a.data, resolved, jobs=count, **options)
        verification_finished = time.monotonic()
        receipt['timings_s'] = {'rollout': rollout_finished-run_started,
                               'verification': verification_finished-rollout_finished}
        receipt.update(status='verified_smoke' if partial else 'trace_verified', verification=check, full_suite=not partial)
        after = source_snapshot(root)
        receipt['source_unchanged_during_run'] = before == after
        if before != after:
            raise ValueError('source changed during evaluation; rerun from a fixed checkout')
    except Exception as error:
        receipt.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt.setdefault('timings_s', {})['total'] = time.monotonic()-run_started
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    progress.emit('complete', force=True, status=receipt['status'], receipt=str(receipt_path))
    print(json.dumps({'status': receipt['status'], 'receipt': str(receipt_path)}, indent=2))

    return receipt
