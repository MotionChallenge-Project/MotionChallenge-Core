"""Development evaluator. Core owns data, clocks, physics, failures and metrics.

Outputs are unverified development records until source provenance,
upstream inference parity and the complete protocol are independently reviewed.
"""
import argparse
import json
import math
from pathlib import Path
import time

from motion_challenge.evaluation.accumulator import MetricAccumulator
from motion_challenge.runtime.observations import controller_state, validate_access

from motion_challenge.simulation.physics import tracking_sample
from motion_challenge.validation import external_output_path, sha256_file


def clip_steps(world, motion, runtime, protocol, max_duration=None, trace=None, on_step=None):
    access = validate_access(protocol, runtime)
    state = yield ('reset', motion)
    offsets = runtime['reference_offsets_s']
    def window(t):
        return tuple(world.reference(t+x) for x in offsets)
    yield ('policy_reset', controller_state(state, access), window(0), 0)
    command = yield ('act', controller_state(state, access), window(0))
    period = runtime['control_period_s']
    metric_dt = protocol['metric_period_s']
    end = motion.duration_s if max_duration is None else min(max_duration, motion.duration_s)
    control_i, metric_i = 1, 1
    samples, failure, last_valid = MetricAccumulator(), [], 0.0
    first_failure_time = None
    metric_prefix = None
    terminate_on_failure = protocol.get('terminate_on_failure', True)
    if not isinstance(terminate_on_failure, bool):
        raise ValueError('terminate_on_failure must be boolean')
    started = time.monotonic()
    while state.time_s < end-1e-9:
        next_t = min(control_i*period, metric_i*metric_dt, end)
        state = yield ('advance', command, next_t)
        if on_step:
            on_step(float(state.time_s), end)
        if next_t >= metric_i*metric_dt-1e-9 or next_t >= end-1e-9:
            current_failure, values = yield ('score', state, world.reference(next_t))
            if current_failure and first_failure_time is None:
                failure = current_failure
                first_failure_time = float(state.time_s)
            tick = time.perf_counter()
            samples.add(values)
            if current_failure and metric_prefix is None:
                metric_prefix = dict(sample_count=samples.count,
                    observed_time_s=float(state.time_s),metric_means_over_observed_prefix=samples.means(),
                    last_sample={'time_s':next_t,**samples.last})
            world.trace_work['metric_accumulation_s'] += time.perf_counter()-tick
            if trace is not None:
                tick = time.perf_counter()
                line = json.dumps({'time_s':next_t,
                    'joint_position_rad':state.joint_position_rad,
                    'root_position_world_m':state.root_position_world_m,
                    'root_orientation_xyzw':state.root_orientation_xyzw}, allow_nan=False)+'\n'
                world.trace_work['json_encoding_s'] += time.perf_counter()-tick
                tick = time.perf_counter()
                trace.write(line)
                world.trace_work['buffer_submit_s'] += time.perf_counter()-tick
                world.trace_work['samples'] += 1
            metric_i += 1
            if failure and terminate_on_failure:
                break
            if first_failure_time is None:
                last_valid = next_t
        if next_t >= end-1e-9:
            break
        if next_t >= control_i*period-1e-9:
            command = yield ('act', controller_state(state, access), window(next_t))
            control_i += 1
    means = samples.means()
    return {'success': not failure and last_valid >= motion.duration_s-1e-9,
        'duration_s': motion.duration_s, 'observed_time_s': float(state.time_s),
        'completion': last_valid/motion.duration_s,
        'failure_time_s': first_failure_time,
        'failure_reasons': failure, 'sample_count': samples.count,
        'metric_means_over_observed_prefix': means, 'last_sample': {'time_s': next_t, **samples.last} if samples.count else None,
        'metric_prefix': metric_prefix or dict(sample_count=samples.count,
            observed_time_s=float(state.time_s),metric_means_over_observed_prefix=means,
            last_sample={'time_s':next_t,**samples.last} if samples.count else None),
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
    p.add_argument('--backend', choices=['batch'], default='batch', help=argparse.SUPPRESS)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int)
    p.add_argument('--progress-interval', type=float, default=5.0)
    p.add_argument('--no-traces', action='store_true', help='Keep online metrics only; no independent trajectory verification')
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
    access = validate_access(protocol, runtime)
    engine = __import__('mujoco').__version__
    if protocol.get('mujoco_version', engine) != engine:
        raise ValueError('installed MuJoCo does not match the frozen protocol')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    results = {'kind': 'development_evaluation', 'official': False,
        'protocol_id': protocol['id'],
        'state_access': access,
        'termination_mode': 'early_termination' if protocol.get('terminate_on_failure', True) else 'no_termination',
        'metric_scope': 'observed_prefix' if protocol.get('terminate_on_failure', True) else 'full_motion',
        'controller_id': runtime['id'], 'controller_config_sha256': sha256_file(a.controller),
        'suite_sha256': sha256_file(a.suite), 'protocol_sha256': sha256_file(a.protocol),
        'robot_profile_sha256': sha256_file(a.robot), 'mujoco_version': __import__('mujoco').__version__,
        'isolation': 'in_process_sdk_contract_not_security_sandbox', 'clips': [], 'status': 'running',
        'backend': a.backend, 'batch_size': a.batch_size, 'traces_recorded': not a.no_traces}
    selected = ([f for f in manifest['files'] if f['path'] in a.clip_id] if a.clip_id
                else manifest['files'][:a.max_clips] if a.max_clips else manifest['files'])
    from motion_challenge.simulation.batch_dispatch import evaluate_batch
    evaluate_batch(a, selected, results, protocol, len(selected) == len(manifest['files']) and a.max_duration is None)


if __name__ == '__main__':
    main()
