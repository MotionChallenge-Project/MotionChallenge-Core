"""Verify and combine complete motion shards under one frozen evaluation contract."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil

from motion_challenge.validation import confined_path, external_output_path, sha256_file
from motion_challenge.evaluation.verify import verify


CONTRACT_KEYS = ('controller_id', 'controller_config_sha256', 'suite_sha256',
                 'protocol_sha256', 'robot_profile_sha256', 'mujoco_version')


def validate_shards(records, expected_ids):
    if not records or not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('nonempty unique suite and shards required')
    baseline = records[0]
    clips = {}
    for record in records:
        if record.get('status') != 'completed':
            raise ValueError('unfinished or infrastructure-failed shard')
        if any(record.get(key) != baseline.get(key) for key in CONTRACT_KEYS):
            raise ValueError('shards have different evaluation contracts')
        for clip in record['clips']:
            if clip.get('truncated_smoke'):
                raise ValueError('a smoke truncation cannot enter a full ranking')
            if clip['id'] in clips:
                raise ValueError('duplicate motion across shards')
            clips[clip['id']] = clip
    if set(clips) != set(expected_ids):
        raise ValueError('missing or extra motion across shards')
    return [clips[key] for key in expected_ids]


def merge(paths, output, suite, robot, protocol, data, controller):
    paths = [Path(p).resolve() for p in paths]
    output = external_output_path(Path(output))
    manifest = json.loads(Path(suite).read_text())
    records = [json.loads(p.read_text()) for p in paths]
    ordered = validate_shards(records, [f['path'] for f in manifest['files']])
    receipts = []
    for path in paths:
        receipt = verify(path, suite, robot, protocol, data, controller)
        receipts.append({'name':path.name, 'sha256':sha256_file(path), **receipt})
    result = {k:copy.deepcopy(v) for k,v in records[0].items()
              if k not in ('clips','error','success_count','macro_completion',
                           'macro_metrics_over_observed_prefix','full_suite')}
    origins = {c['id']:p for p,r in zip(paths,records) for c in r['clips']}
    output.parent.mkdir(parents=True,exist_ok=True)
    result['clips'] = []
    for clip in ordered:
        clip = copy.deepcopy(clip)
        source = confined_path(origins[clip['id']].parent,clip['trace']['path'])
        target = output.parent/(clip['trace']['sha256']+'.poses.jsonl.gz')
        if source != target:
            if target.exists():
                if sha256_file(target) != clip['trace']['sha256']:
                    raise ValueError('existing trace copy has unexpected bytes')
            else:shutil.copyfile(source,target)
        clip['trace']['path'] = target.name
        result['clips'].append(clip)
    result['status'],result['full_suite'] = 'completed',True
    result['success_count'] = sum(c['success'] for c in ordered)
    result['macro_completion'] = math.fsum(c['completion'] for c in ordered)/len(ordered)
    if 'macro_metrics_over_observed_prefix' in records[0]:
        result['macro_metrics_over_observed_prefix'] = {
            k:math.fsum(c['metric_means_over_observed_prefix'][k] for c in ordered)/len(ordered)
            for k in ordered[0]['metric_means_over_observed_prefix']}
    result['verified_shards'] = receipts
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    receipt = verify(output,suite,robot,protocol,data,controller)
    return {**receipt,'success_count':result['success_count'],
            'macro_completion':result['macro_completion']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','suite','robot','protocol','data','controller'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('shards',nargs='+',type=Path)
    a=p.parse_args()
    print(json.dumps(merge(a.shards,a.output,a.suite,a.robot,a.protocol,a.data,a.controller),indent=2))


if __name__=='__main__':main()
