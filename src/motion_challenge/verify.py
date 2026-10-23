"""Independent recomputation from recorded poses and the frozen core inputs.

Trace consistency is not a signature, sandbox attestation or proof of physical
execution. Official acceptance requires an independent rerun as well.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

import mujoco

from .dataset import load_csv_motion
from .physics import World, tracking_sample
from .sdk import ReferenceFrame
from .validation import confined_path, sha256_file


def failure_time_matches(recorded, metric_time, failed):
    if not failed:
        return recorded is None
    return (isinstance(recorded, (int, float)) and math.isfinite(recorded)
            and math.isclose(recorded, metric_time, rel_tol=0.0, abs_tol=1e-8))


def verify(result_path, suite_path, robot_path, protocol_path, data_root, controller_path):
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    if result.get('status') != 'completed':
        raise ValueError('run did not complete')
    for key, path in [('suite_sha256',suite_path),('robot_profile_sha256',robot_path),
                      ('protocol_sha256',protocol_path),('controller_config_sha256',controller_path)]:
        if sha256_file(path) != result[key]:
            raise ValueError(f'{key} mismatch')
    protocol = json.loads(Path(protocol_path).read_text())
    manifest = json.loads(Path(suite_path).read_text())
    files = {f['path']:f for f in manifest['files']}
    ids = [c['id'] for c in result['clips']]
    if len(ids) != len(set(ids)) or not set(ids) <= files.keys():
        raise ValueError('unknown or duplicate clips')
    if result['full_suite'] and set(ids) != set(files):
        raise ValueError('full-suite record is missing clips')
    world = World(robot_path)
    checked_samples = 0
    for clip in result['clips']:
        item = files[clip['id']]
        source = confined_path(data_root, item['path'])
        if sha256_file(source) != item['sha256']:
            raise ValueError('source dataset hash mismatch')
        motion = load_csv_motion(source, manifest['joint_names'], manifest['source_fps'])
        world.reset(motion)
        trace = confined_path(result_path.parent, clip['trace']['path'])
        if sha256_file(trace) != clip['trace']['sha256']:
            raise ValueError('pose trace hash mismatch')
        samples, last_valid, failure, last_time = [], 0.0, [], 0.0
        with gzip.open(trace, 'rt') as stream:
            for line in stream:
                if len(line) > 100000 or failure:
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
                expected = min((len(samples)+1)*protocol['metric_period_s'], motion.duration_s)
                if not clip['truncated_smoke'] and not math.isclose(t,expected,abs_tol=1e-8):
                    raise ValueError('missing or shifted metric timestamp')
                frame = ReferenceFrame(t,t,tuple(p['joint_position_rad']),
                    tuple(p['root_position_world_m']),tuple(p['root_orientation_xyzw']))
                world.data.qpos[:] = world._qpos(frame)
                world.data.qvel[:] = 0
                world.data.time = t
                mujoco.mj_forward(world.model,world.data)
                failure, metrics = tracking_sample(world.state(),world.reference(t),protocol)
                samples.append(metrics)
                if not failure: last_valid = t
                last_time = t
        if not samples or len(samples) != clip['sample_count']:
            raise ValueError('trace sample count mismatch')
        success = not failure and last_valid >= motion.duration_s-1e-9
        if not failure and not clip['truncated_smoke'] and not success:
            raise ValueError('non-failed full episode ended before final timestamp')
        if success != clip['success'] or failure != clip['failure_reasons']:
            raise ValueError('success/failure recomputation mismatch')
        if not math.isclose(last_valid/motion.duration_s,clip['completion'],abs_tol=1e-9):
            raise ValueError('completion recomputation mismatch')
        if not math.isclose(last_time,clip['observed_time_s'],abs_tol=1e-8):
            raise ValueError('observed duration mismatch')
        if not failure_time_matches(clip['failure_time_s'], last_time, bool(failure)):
            raise ValueError('failure timestamp mismatch')
        if set(clip['metric_means_over_observed_prefix']) != set(samples[0]):
            raise ValueError('missing or extra protocol metrics')
        for k, recorded in clip['metric_means_over_observed_prefix'].items():
            actual = math.fsum(s[k] for s in samples)/len(samples)
            if not math.isclose(actual,recorded,abs_tol=1e-9,rel_tol=1e-7):
                raise ValueError(f'{k} recomputation mismatch')
        checked_samples += len(samples)
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
    return {'verification':'trace_consistent','official':False,'clips':len(ids),
            'samples':checked_samples,'full_suite':result['full_suite'],
            'scope':'hashes, canonical FK, timing, failure rules and metrics; not execution attestation'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('result','suite','robot','protocol','data','controller'):
        p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(verify(a.result,a.suite,a.robot,a.protocol,a.data,a.controller),indent=2))


if __name__ == '__main__':
    main()
