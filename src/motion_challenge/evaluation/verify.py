"""Independent recomputation from recorded poses and the frozen core inputs.

Trace consistency is not a signature, sandbox attestation or proof of physical
execution. Official acceptance requires an independent rerun as well.
"""
import argparse
from motion_challenge.evaluation.accumulator import MetricAccumulator
import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gzip
import json
import math
from pathlib import Path

import mujoco

from motion_challenge.data.dataset import load_csv_motion
from motion_challenge.simulation.physics import World, forward_kinematics, tracking_sample
from motion_challenge.runtime.progress import Progress
from motion_challenge.sdk import ReferenceFrame
from motion_challenge.validation import confined_path, sha256_file


def failure_time_matches(recorded, metric_time, failed):
    if not failed:
        return recorded is None
    return (isinstance(recorded, (int, float)) and math.isfinite(recorded)
            and math.isclose(recorded, metric_time, rel_tol=0.0, abs_tol=1e-8))


def _verify_clip(task):
    clip, item, manifest, robot_path, protocol, data_root, result_directory, derive_prefix = task
    world = World(robot_path)
    terminate_on_failure = protocol.get('terminate_on_failure', True)
    progress = Progress()
    source = confined_path(data_root, item['path'])
    if sha256_file(source) != item['sha256']:
        raise ValueError('source dataset hash mismatch')
    motion = load_csv_motion(source, manifest['joint_names'], manifest['source_fps'])
    world.reset(motion)
    trace = confined_path(result_directory, clip['trace']['path'])
    if sha256_file(trace) != clip['trace']['sha256']:
        raise ValueError('pose trace hash mismatch')
    samples, last_valid, failure, last_time = MetricAccumulator(), 0.0, [], 0.0
    first_failure_time = None
    prefix_count, prefix_means, prefix_last = None, None, None
    limit = clip['trace'].get('sample_limit')
    if limit is not None and (not terminate_on_failure or type(limit) is not int or limit < 1):
        raise ValueError('invalid trace prefix sample limit')
    with gzip.open(trace, 'rt') as stream:
        for line in stream:
            if limit is not None and samples.count >= limit:
                break
            if len(line) > 100000 or (failure and terminate_on_failure):
                raise ValueError('invalid pose trace or continued after failure')
            p = json.loads(line)
            t = p['time_s']
            pose = p['joint_position_rad']+p['root_position_world_m']+p['root_orientation_xyzw']
            if len(pose) != len(world.joints)+7 or not all(math.isfinite(x) for x in pose):
                raise ValueError('invalid recorded pose')
            if not math.isclose(sum(x*x for x in p['root_orientation_xyzw']),1.0,abs_tol=1e-5):
                raise ValueError('invalid recorded quaternion')
            if not last_time < t <= motion.duration_s+1e-9:
                raise ValueError('pose timestamps are out of order/range')
            expected = min((samples.count+1)*protocol['metric_period_s'], motion.duration_s)
            if not clip['truncated_smoke'] and not math.isclose(t,expected,abs_tol=1e-8):
                raise ValueError('missing or shifted metric timestamp')
            frame = ReferenceFrame(t,t,tuple(p['joint_position_rad']),
                tuple(p['root_position_world_m']),tuple(p['root_orientation_xyzw']))
            world.data.qpos[:] = world._qpos(frame)
            world.data.qvel[:] = 0
            world.data.time = t
            forward_kinematics(world.model, world.data)
            current_failure, metrics = tracking_sample(world.state(),world.reference(t),protocol)
            if current_failure and first_failure_time is None:
                failure = current_failure
                first_failure_time = t
                prefix_count = samples.count+1
            samples.add(metrics)
            if prefix_count == samples.count:
                prefix_means, prefix_last = samples.means(), samples.last
            if samples.count % 100 == 0:
                progress.emit('verification_clip', clip=clip['id'],
                              samples=samples.count, total_samples=clip['sample_count'], time_s=t)
            if not failure: last_valid = t
            last_time = t
    if not samples.count or samples.count != clip['sample_count']:
        raise ValueError('trace sample count mismatch')
    success = not failure and last_valid >= motion.duration_s-1e-9
    if not failure and not clip['truncated_smoke'] and not success:
        raise ValueError('non-failed full episode ended before final timestamp')
    if not terminate_on_failure and not clip['truncated_smoke'] and last_time < motion.duration_s-1e-9:
        raise ValueError('no-termination episode ended before final timestamp')
    if success != clip['success'] or failure != clip['failure_reasons']:
        raise ValueError('success/failure recomputation mismatch')
    if not math.isclose(last_valid/motion.duration_s,clip['completion'],abs_tol=1e-9):
        raise ValueError('completion recomputation mismatch')
    if not math.isclose(last_time,clip['observed_time_s'],abs_tol=1e-8):
        raise ValueError('observed duration mismatch')
    if not failure_time_matches(clip['failure_time_s'], first_failure_time, bool(failure)):
        raise ValueError('failure timestamp mismatch')
    if set(clip['metric_means_over_observed_prefix']) != set(samples.partials):
        raise ValueError('missing or extra protocol metrics')
    for k, recorded in clip['metric_means_over_observed_prefix'].items():
        actual = samples.means()[k]
        if not math.isclose(actual,recorded,abs_tol=1e-9,rel_tol=1e-7):
            raise ValueError(f'{k} recomputation mismatch')
    derived = None
    if derive_prefix:
        count = prefix_count or samples.count
        terminal_time = first_failure_time if failure else last_time
        item = {**clip, 'sample_count': count,
                'observed_time_s': clip['failure_time_s'] if failure else clip['observed_time_s'],
                'metric_means_over_observed_prefix': prefix_means or samples.means(),
                'last_sample': {'time_s': terminal_time, **(prefix_last or samples.last)},
                'trace': {**clip['trace'], 'sample_limit': count}}
        item.pop('wall_time_s', None)  # No second physical rollout took place.
        derived = item
    return {'samples': samples.count, 'prefix': derived}


def verify(result_path, suite_path, robot_path, protocol_path, data_root, controller_path,
           prefix_result_path=None, prefix_protocol_path=None, jobs=1):
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    if result.get('traces_recorded') is False:
        raise ValueError('no recorded trajectory to verify')
    if result.get('status') != 'completed':
        raise ValueError('run did not complete')
    for key, path in [('suite_sha256',suite_path),('robot_profile_sha256',robot_path),
                      ('protocol_sha256',protocol_path),('controller_config_sha256',controller_path)]:
        if sha256_file(path) != result[key]:
            raise ValueError(f'{key} mismatch')
    protocol = json.loads(Path(protocol_path).read_text())
    from motion_challenge.runtime.observations import validate_access
    access = validate_access(protocol, json.loads(Path(controller_path).read_text()))
    if result.get('state_access') != access:
        raise ValueError('recorded observation access differs from protocol')
    terminate_on_failure = protocol.get('terminate_on_failure', True)
    if not isinstance(terminate_on_failure, bool):
        raise ValueError('terminate_on_failure must be boolean')
    mode = 'early_termination' if terminate_on_failure else 'no_termination'
    if result.get('termination_mode', mode) != mode:
        raise ValueError('termination mode differs from protocol')
    if result.get('mujoco_version') != mujoco.__version__:
        raise ValueError('verification must use the recorded MuJoCo version')
    if protocol.get('mujoco_version', mujoco.__version__) != mujoco.__version__:
        raise ValueError('MuJoCo version differs from protocol')
    prefix = None
    if prefix_result_path is not None:
        if terminate_on_failure or prefix_protocol_path is None:
            raise ValueError('prefix derivation requires a full-motion trace and prefix protocol')
        prefix_result_path = Path(prefix_result_path)
        if prefix_result_path.parent.resolve() != result_path.parent.resolve() or prefix_result_path == result_path:
            raise ValueError('derived prefix must be a separate record beside the full trace')
        prefix_protocol = json.loads(Path(prefix_protocol_path).read_text())
        def physical_rules(spec):
            spec = copy.deepcopy(spec)
            for name in ('id', 'warning', 'terminate_on_failure'):
                spec.pop(name, None)
            spec.get('position_alignment_definition', {}).pop('aggregation', None)
            return spec
        if not prefix_protocol.get('terminate_on_failure', True) or physical_rules(protocol) != physical_rules(prefix_protocol):
            raise ValueError('prefix and full-motion protocols have different physical or scoring rules')
        prefix = {**result, 'protocol_id': prefix_protocol['id'],
                  'protocol_sha256': sha256_file(prefix_protocol_path), 'termination_mode': 'early_termination',
                  'metric_scope': 'observed_prefix', 'clips': [],
                  'derivation': {'method': 'first-failure prefix of one verified full rollout',
                                 'full_result_sha256': sha256_file(result_path)}}
    manifest = json.loads(Path(suite_path).read_text())
    files = {f['path']:f for f in manifest['files']}
    ids = [c['id'] for c in result['clips']]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= files.keys():
        raise ValueError('unknown or duplicate clips')
    if result['full_suite'] and set(ids) != set(files):
        raise ValueError('full-suite record is missing clips')
    if type(jobs) is not int or jobs < 1:
        raise ValueError('verification jobs must be positive')
    progress = Progress()
    checked_samples = 0
    motion_layout = {key: manifest[key] for key in ('joint_names', 'source_fps')}
    tasks = [(clip, files[clip['id']], motion_layout, robot_path, protocol, data_root,
              result_path.parent, prefix is not None) for clip in result['clips']]
    from motion_challenge.simulation.vector_physics import cpu_count
    workers = min(jobs, len(tasks), cpu_count())
    def collect(records):
        nonlocal checked_samples
        for index, record in enumerate(records):
            checked_samples += record['samples']
            if prefix is not None:
                prefix['clips'].append(record['prefix'])
            progress.emit('verification', force=True, completed=index+1, total=len(ids), samples=checked_samples)
    if workers > 1:
        # Each process independently checks a complete clip. Ordered collection
        # preserves deterministic aggregation, including the first-failure prefix.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            collect(pool.map(_verify_clip, tasks))
    else:
        collect(map(_verify_clip, tasks))
    if sum(c['success'] for c in result['clips']) != result['success_count']:
        raise ValueError('success count mismatch')
    mean_completion=math.fsum(c['completion'] for c in result['clips'])/len(ids)
    if not math.isclose(mean_completion,result['macro_completion'],abs_tol=1e-9):
        raise ValueError('aggregate completion mismatch')
    if protocol.get('position_alignment_metrics'):
        keys = result['clips'][0]['metric_means_over_observed_prefix']
        recorded = result.get('macro_metrics_over_observed_prefix', {})
        if set(recorded) != set(keys):
            raise ValueError('missing or extra aggregate protocol metrics')
        for key in keys:
            mean = math.fsum(c['metric_means_over_observed_prefix'][key]
                             for c in result['clips'])/len(ids)
            if not math.isclose(mean,recorded[key],abs_tol=1e-9,rel_tol=1e-7):
                raise ValueError(f'{key} aggregate recomputation mismatch')
    check = {'verification':'trace_consistent','clips':len(ids),
            'samples':checked_samples,'full_suite':result['full_suite'],
            'scope':'hashes, canonical FK, timing, failure rules and metrics; not execution attestation'}
    if prefix is not None:
        prefix['macro_metrics_over_observed_prefix'] = {
            k: math.fsum(c['metric_means_over_observed_prefix'][k] for c in prefix['clips'])/len(ids)
            for k in prefix['clips'][0]['metric_means_over_observed_prefix']}
        prefix_result_path.write_text(json.dumps(prefix, indent=2, allow_nan=False)+'\n')
        check['derived_prefix'] = {'result': prefix_result_path.name,
                                   'result_sha256': sha256_file(prefix_result_path),
                                   'samples': sum(c['sample_count'] for c in prefix['clips'])}
    return check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('result','suite','robot','protocol','data','controller'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--jobs', type=int, default=1)
    a=p.parse_args()
    print(json.dumps(verify(a.result,a.suite,a.robot,a.protocol,a.data,a.controller,jobs=a.jobs),indent=2))


if __name__ == '__main__':
    main()
