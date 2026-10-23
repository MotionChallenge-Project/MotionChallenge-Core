"""Shared one-command evaluation for independent model repositories.

Artifacts and receipts live outside Git. Model repositories provide inference,
immutable artifact declarations and an optional model-specific export hook.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

from .dataset import audit_dataset, fetch_dataset
from .merge import merge
from .resources import lafan_paths
from .validation import external_output_path, sha256_file
from .verify import verify


def artifact(path, expected_hash, url=None, offline=False):
    path = Path(path)
    if path.is_file():
        if sha256_file(path) != expected_hash:
            raise ValueError(f'checkpoint checksum mismatch: {path}')
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
                while block := response.read(1024*1024):
                    stream.write(block)
        if sha256_file(temporary) != expected_hash:
            raise ValueError('downloaded checkpoint checksum mismatch')
        temporary.replace(path)
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


def main(adapter_root=None):
    p = argparse.ArgumentParser(description='Evaluate a checkpoint with the shared benchmark core.')
    p.add_argument('--suite', default='lafan40', choices=['lafan40'])
    p.add_argument('--data', type=Path, default=Path(os.environ.get('MOTION_CHALLENGE_DATA', '~/.cache/motion-challenge/lafan')))
    p.add_argument('--weights', type=Path)
    p.add_argument('--output', type=Path, help='New directory outside any Git repository')
    p.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument('--offline', action='store_true')
    p.add_argument('--max-clips', type=int)
    p.add_argument('--max-duration', type=float)
    a = p.parse_args()
    if a.jobs < 1 or (a.max_clips is not None and a.max_clips < 1):
        p.error('jobs and max-clips must be positive')
    root = Path(adapter_root or Path.cwd()).resolve()
    original = root/'runtime.json'
    runtime = json.loads(original.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = external_output_path((a.output or Path.home()/'.local/share/motion-challenge/runs'/runtime['id']/stamp).expanduser())
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be new or empty; existing evidence is never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    a.data = external_output_path(a.data.expanduser())
    paths = lafan_paths()
    manifest = json.loads(paths['suite'].read_text())
    if not a.offline:
        print(json.dumps(fetch_dataset(manifest, a.data)), flush=True)
    audit_dataset(manifest, a.data)
    if 'weights_sha256' in runtime:
        local = root/runtime.get('weights', 'checkpoints/policy.onnx')
        weights = (a.weights.expanduser().resolve() if a.weights else local if local.is_file()
                   else Path.home()/'.cache/motion-challenge/checkpoints'/runtime['weights_sha256']/'policy.onnx')
        if not weights.exists() and not a.offline and runtime.get('prepare_checkpoint'):
            sys.path.insert(0, str(root))
            module, function = runtime['prepare_checkpoint'].split(':')
            getattr(importlib.import_module(module), function)(weights)
        runtime['weights'] = str(artifact(weights, runtime['weights_sha256'], runtime.get('weights_url'), a.offline).resolve())
    runtime['adapter_root'] = str(root)
    resolved = output/'runtime.json'
    resolved.write_text(json.dumps(runtime, indent=2)+'\n')
    before = source_snapshot(root)
    receipt = {'submitter': 'local-run', 'source': before,
               'source_runtime_sha256': sha256_file(original), 'resolved_runtime_sha256': sha256_file(resolved),
               'suite': a.suite, 'started_utc': stamp, 'author_inference_review': 'not_granted_by_runner',
               'output': str(output), 'status': 'running'}
    receipt_path = output/'receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    selected = manifest['files'][:a.max_clips] if a.max_clips else manifest['files']
    count = min(a.jobs, len(selected))
    shards = [selected[i::count] for i in range(count)]
    common = ['--controller', str(resolved), '--data', str(a.data)]
    for key, value in paths.items():
        common += ['--'+key, str(value)]

    def evaluate(item):
        index, clips = item
        result = output/f'shard-{index:02d}.json'
        command = [sys.executable, '-m', 'motion_challenge.run', *common, '--output', str(result)]
        for clip in clips:
            command += ['--clip-id', clip['path']]
        if a.max_duration is not None:
            command += ['--max-duration', str(a.max_duration)]
        with (output/f'shard-{index:02d}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Finished shard {index+1}/{count}', flush=True)
        return result

    try:
        with ThreadPoolExecutor(max_workers=count) as executor:
            results = list(executor.map(evaluate, enumerate(shards)))
        partial = a.max_clips is not None or a.max_duration is not None
        if partial:
            checks = [verify(r, paths['suite'], paths['robot'], paths['protocol'], a.data, resolved) for r in results]
            receipt.update(status='verified_smoke', verification=checks, full_suite=False)
        else:
            check = merge(results, output/'result.json', paths['suite'], paths['robot'], paths['protocol'], a.data, resolved)
            receipt.update(status='trace_verified', verification=check, full_suite=True)
        after = source_snapshot(root)
        receipt['source_unchanged_during_run'] = before == after
        if before != after:
            raise ValueError('source changed during evaluation; rerun from a fixed checkout')
    except Exception as error:
        receipt.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'status': receipt['status'], 'receipt': str(receipt_path)}, indent=2))
