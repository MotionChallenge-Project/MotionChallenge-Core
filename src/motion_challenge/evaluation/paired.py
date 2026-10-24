"""One result for the two metric scopes of a fixed evaluation."""
import math


def combine(early, full, source, checkpoint_sha256):
    for key in ('controller_id', 'controller_config_sha256', 'suite_sha256',
                'robot_profile_sha256', 'mujoco_version', 'backend'):
        if early[key] != full[key]:
            raise ValueError(f'Paired evaluation has different {key}')
    a = {c['id']: c for c in early['clips']}
    b = {c['id']: c for c in full['clips']}
    if not a or a.keys() != b.keys() or len(a) != len(early['clips']) or len(b) != len(full['clips']):
        raise ValueError('Paired evaluation has different or duplicate clips')
    clips = []
    for key, prefix in a.items():
        whole = b[key]
        for field in ('success', 'failure_reasons'):
            if prefix[field] != whole[field]:
                raise ValueError(f'{key}: paired {field} differs')
        for field in ('duration_s', 'completion', 'failure_time_s'):
            x, y = prefix[field], whole[field]
            if (x is None) != (y is None) or (x is not None and not math.isclose(x, y, rel_tol=0, abs_tol=1e-9)):
                raise ValueError(f'{key}: paired {field} differs')
        clips.append({**{k: prefix[k] for k in ('id', 'duration_s', 'success', 'completion', 'failure_time_s', 'failure_reasons')},
                      'metrics': {'until_first_failure': prefix['metric_means_over_observed_prefix'],
                                  'full_motion': whole['metric_means_over_observed_prefix']},
                      'sample_count': {'until_first_failure': prefix['sample_count'], 'full_motion': whole['sample_count']}})
    return {'schema': 'motion-challenge/unified-result-v1', 'status': 'trace_verified',
            'source': source, 'checkpoint_sha256': checkpoint_sha256,
            **{k: early[k] for k in ('controller_id', 'suite_sha256', 'robot_profile_sha256', 'mujoco_version', 'backend')},
            'success_count': sum(c['success'] for c in clips), 'total': len(clips),
            'macro_completion': math.fsum(c['completion'] for c in clips)/len(clips),
            'metrics': {'until_first_failure': early['macro_metrics_over_observed_prefix'],
                        'full_motion': full['macro_metrics_over_observed_prefix']},
            'clips': clips}
