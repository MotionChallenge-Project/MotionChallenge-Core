"""Development evaluator. Core owns data, clocks, physics, failures and metrics.

Outputs are explicitly unverified development records until deployment isolation,
upstream inference parity and the complete protocol are independently reviewed.
"""
import argparse
from dataclasses import asdict
import gzip
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import threading

from .progress import Progress

from .dataset import load_csv_motion
from .physics import World, tracking_sample
from .sdk import ControlCommand
from .validation import confined_path, external_output_path, sha256_file


class Worker:
    def __init__(self, config_path, logfile, timeout=120, python_executable=None):
        self.log = open(logfile, 'w')
        self.timeout = timeout
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        self.process = subprocess.Popen([python_executable or sys.executable, '-m', 'motion_challenge.worker', str(config_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self._read()
        except Exception:
            self.close()
            raise

    def _read(self):
        if not self.selector.select(self.timeout):
            self.close()
            raise TimeoutError('adapter timed out; see worker log')
        line = self.process.stdout.readline(16*1024*1024)
        if not line:
            raise RuntimeError(f'adapter worker exited ({self.process.poll()}); see worker log')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def call(self, op, state, reference, seed=0):
        self.process.stdin.write(json.dumps({'op': op, 'state': asdict(state),
            'reference': [asdict(f) for f in reference], 'seed': seed}, allow_nan=False)+'\n')
        self.process.stdin.flush()
        result = self._read()
        return ControlCommand(**result['command']) if op == 'act' else None

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.selector.close()
        self.log.close()


def evaluate_clip(world, motion, worker, runtime, protocol, max_duration=None, trace=None, on_step=None):
    state = world.reset(motion)
    offsets = runtime['reference_offsets_s']
    def window(t):
        return tuple(world.reference(t+x) for x in offsets)
    worker.call('reset', state, window(0), seed=0)
    command = worker.call('act', state, window(0))
    period = runtime['control_period_s']
    metric_dt = protocol['metric_period_s']
    end = motion.duration_s if max_duration is None else min(max_duration, motion.duration_s)
    control_i, metric_i = 1, 1
    samples, failure, last_valid = [], [], 0.0
    first_failure_time = None
    terminate_on_failure = protocol.get('terminate_on_failure', True)
    if not isinstance(terminate_on_failure, bool):
        raise ValueError('terminate_on_failure must be boolean')
    started = time.monotonic()
    while world.data.time < end-1e-9:
        next_t = min(control_i*period, metric_i*metric_dt, end)
        world.advance(command, next_t)
        state = world.state()
        if on_step:
            on_step(float(world.data.time), end)
        if next_t >= metric_i*metric_dt-1e-9 or next_t >= end-1e-9:
            current_failure, values = tracking_sample(state, world.reference(next_t), protocol)
            if current_failure and first_failure_time is None:
                failure = current_failure
                first_failure_time = float(world.data.time)
            samples.append({'time_s': next_t, **values})
            if trace is not None:
                trace.write(json.dumps({'time_s':next_t,
                    'joint_position_rad':state.joint_position_rad,
                    'root_position_world_m':state.root_position_world_m,
                    'root_orientation_xyzw':state.root_orientation_xyzw}, allow_nan=False)+'\n')
            metric_i += 1
            if failure and terminate_on_failure:
                break
            if first_failure_time is None:
                last_valid = next_t
        if next_t >= end-1e-9:
            break
        if next_t >= control_i*period-1e-9:
            command = worker.call('act', state, window(next_t))
            control_i += 1
    means = {k: math.fsum(s[k] for s in samples)/len(samples)
             for k in samples[0] if k != 'time_s'} if samples else {}
    return {'success': not failure and last_valid >= motion.duration_s-1e-9,
        'duration_s': motion.duration_s, 'observed_time_s': float(world.data.time),
        'completion': last_valid/motion.duration_s,
        'failure_time_s': first_failure_time,
        'failure_reasons': failure, 'sample_count': len(samples),
        'metric_means_over_observed_prefix': means, 'last_sample': samples[-1] if samples else None,
        'wall_time_s': time.monotonic()-started,
        'truncated_smoke': max_duration is not None and end < motion.duration_s}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--controller', type=Path, required=True)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--robot', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-duration', type=float)
    p.add_argument('--max-clips', type=int)
    p.add_argument('--clip-id', action='append', help='Exact suite clip ID for independently verifiable sharding; repeatable')
    p.add_argument('--backend', choices=['batch', 'serial', 'batch-client'], default='batch')
    p.add_argument('--batch-address', help=argparse.SUPPRESS)
    p.add_argument('--batch-slot', type=int, help=argparse.SUPPRESS)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int)
    p.add_argument('--progress-interval', type=float, default=5.0)
    p.add_argument('--worker-python', help='Python executable in the independently locked adapter environment')
    a = p.parse_args()
    if a.batch_size < 1 or (a.threads is not None and a.threads < 1) or a.progress_interval <= 0:
        p.error('batch size, threads and progress interval must be positive')
    a.output = external_output_path(a.output)
    runtime = json.loads(a.controller.read_text())
    period = runtime['control_period_s']
    offsets = runtime['reference_offsets_s']
    if not math.isfinite(period) or period <= 0 or not offsets or any(not math.isfinite(x) for x in offsets):
        raise ValueError('invalid controller clock/window')
    if offsets != sorted(set(offsets)) or min(offsets) < -runtime['reference_history_s'] or max(offsets) > runtime['reference_preview_s']:
        raise ValueError('reference window is outside the declared access contract')
    if a.max_duration is not None and (not math.isfinite(a.max_duration) or a.max_duration <= 0):
        raise ValueError('smoke duration must be positive')
    manifest = json.loads(a.suite.read_text())
    if a.max_clips is not None and a.max_clips < 1:
        raise ValueError('max-clips must be positive')
    if a.clip_id:
        if a.max_clips is not None or len(set(a.clip_id)) != len(a.clip_id):
            raise ValueError('clip IDs must be unique and cannot combine with max-clips')
        if not set(a.clip_id) <= {f['path'] for f in manifest['files']}:
            raise ValueError('clip ID is not in the frozen suite')
    protocol = json.loads(a.protocol.read_text())
    engine = __import__('mujoco').__version__
    if protocol.get('mujoco_version', engine) != engine:
        raise ValueError('installed MuJoCo does not match the frozen protocol')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    results = {'kind': 'development_evaluation', 'official': False,
        'protocol_id': protocol['id'],
        'termination_mode': 'early_termination' if protocol.get('terminate_on_failure', True) else 'no_termination',
        'metric_scope': 'observed_prefix' if protocol.get('terminate_on_failure', True) else 'full_motion',
        'controller_id': runtime['id'], 'controller_config_sha256': sha256_file(a.controller),
        'suite_sha256': sha256_file(a.suite), 'protocol_sha256': sha256_file(a.protocol),
        'robot_profile_sha256': sha256_file(a.robot), 'mujoco_version': __import__('mujoco').__version__,
        'isolation': 'separate_process_not_security_sandbox', 'clips': [], 'status': 'running',
        'backend': a.backend, 'batch_size': a.batch_size if a.backend == 'batch' else 1}
    selected = ([f for f in manifest['files'] if f['path'] in a.clip_id] if a.clip_id
                else manifest['files'][:a.max_clips] if a.max_clips else manifest['files'])
    if a.backend == 'batch':
        from .batch_dispatch import evaluate_batch
        evaluate_batch(a, selected, results, protocol, len(selected) == len(manifest['files']) and a.max_duration is None)
        return
    progress = Progress(a.output.with_suffix('.progress.json'), a.progress_interval)
    jobs = iter(enumerate(selected))
    lock = threading.RLock()
    active = {}
    order = {clip['path']: i for i, clip in enumerate(selected)}
    def report(force=False):
        with lock:
            progress.emit('evaluation', force=force, completed=len(results['clips']), total=len(selected),
                          passed=sum(c['success'] for c in results['clips']), active=dict(active),
                          backend=a.backend)
    def evaluate_slot(slot, world):
        worker = Worker(a.controller.resolve(), a.output.with_suffix(f'.worker{slot:02d}.log'),
                        python_executable=a.worker_python)
        try:
            while True:
                with lock:
                    item = next(jobs, None)
                    if item is None:
                        break
                    index, clip = item
                    active[slot] = {'clip': clip['path'], 'time_s': 0.0}
                    report(True)
                path = confined_path(a.data, clip['path'])
                if sha256_file(path) != clip['sha256']:
                    raise ValueError('dataset integrity mismatch')
                motion = load_csv_motion(path, manifest['joint_names'], manifest['source_fps'])
                trace_path = a.output.with_name(a.output.stem+f'.clip{index:03d}.poses.jsonl.gz')
                def on_step(t, end):
                    with lock:
                        active[slot] = {'clip': clip['path'], 'time_s': t, 'duration_s': end}
                        report()
                with gzip.open(trace_path, 'wt') as trace:
                    result = evaluate_clip(world, motion, worker, runtime, protocol, a.max_duration, trace, on_step)
                result['trace'] = {'path': trace_path.name, 'sha256': sha256_file(trace_path)}
                with lock:
                    results['clips'].append({'id': clip['path'], **result})
                    results['clips'].sort(key=lambda c: order[c['id']])
                    active.pop(slot, None)
                    temporary = a.output.with_suffix('.tmp')
                    temporary.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
                    temporary.replace(a.output)
                    report(True)
        finally:
            worker.close()
    try:
        if a.backend == 'batch-client':
            from .batch_dispatch import RemoteWorld
            world = RemoteWorld(a.robot, a.batch_address, a.batch_slot)
            try:
                evaluate_slot(0, world)
            finally:
                world.close()
        else:
            evaluate_slot(0, World(a.robot))
        results['status'] = 'completed'
        results['full_suite'] = len(selected) == len(manifest['files']) and a.max_duration is None
        results['success_count'] = sum(x['success'] for x in results['clips'])
        results['macro_completion'] = math.fsum(x['completion'] for x in results['clips'])/len(selected)
        if protocol.get('position_alignment_metrics'):
            metric_keys = results['clips'][0]['metric_means_over_observed_prefix']
            results['macro_metrics_over_observed_prefix'] = {
                key: math.fsum(c['metric_means_over_observed_prefix'][key]
                               for c in results['clips'])/len(selected)
                for key in metric_keys}
    except Exception as error:
        results['status'] = 'infrastructure_error'
        results['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        a.output.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
