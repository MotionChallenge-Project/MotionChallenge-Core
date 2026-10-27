"""Recompute scoring from a complete trajectory without running policy or physics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import errno
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
from motion_challenge.evaluation.paired import combine
from motion_challenge.evaluation.verify import _score_clip
from motion_challenge.resources import lafan_paths, resource_root
from motion_challenge.runtime.observations import validate_access
from motion_challenge.runtime.progress import Progress
from motion_challenge.validation import confined_path, sha256_file


def unchanged_rollout_rules(original, updated):
    """Only failure/scoring metadata may change when reusing a trajectory."""
    def rules(spec):
        spec = copy.deepcopy(spec)
        for key in ('id', 'failure', 'primary_metric'):
            spec.pop(key, None)
        return spec
    if original.get('terminate_on_failure', True) or updated.get('terminate_on_failure', True):
        raise ValueError('rescoring requires complete no-termination trajectories')
    if rules(original) != rules(updated):
        raise ValueError('trajectory reuse requires unchanged physics, observations, references and metric clock')


def core_source():
    root = resource_root()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return {'core_commit': git('rev-parse', 'HEAD'),
            'core_dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}


def rescore(run, output, suite, robot, original_protocol, protocol, prefix_protocol, data, jobs=1):
    run, output = Path(run), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty; original evidence is never overwritten')
    if type(jobs) is not int or jobs < 1:
        raise ValueError('jobs must be a positive integer')
    read = lambda p: json.loads(Path(p).read_text())
    original = read(run/'result.json'); full = read(run/'rollout/result.json')
    prior_receipt = read(run/'rollout/receipt.json')
    runtime = run/'rollout/runtime.json'
    source = original['source']; scoring_source = core_source()
    if original['status']!='trace_verified' or prior_receipt['status']!='trace_verified':
        raise ValueError('original trajectory must already be independently verified')
    if source['adapter_dirty'] or source['core_dirty']:raise ValueError('original source must be clean')
    if prior_receipt['source']!=source or not prior_receipt['source_unchanged_during_run']:
        raise ValueError('original source evidence differs')
    if not full['full_suite'] or full.get('traces_recorded') is False or full['termination_mode']!='no_termination':
        raise ValueError('complete no-termination traces are required')
    for key,path in [('suite_sha256',suite),('robot_profile_sha256',robot),
                     ('protocol_sha256',original_protocol),('controller_config_sha256',runtime)]:
        if full[key]!=sha256_file(path):raise ValueError(key+' mismatch')
    old, new, prefix_spec = read(original_protocol),read(protocol),read(prefix_protocol)
    unchanged_rollout_rules(old,new)
    check=copy.deepcopy(prefix_spec);check['id']=new['id'];check['terminate_on_failure']=False
    if 'warning' in new:check['warning']=new['warning']
    check.get('position_alignment_definition',{}).update(new.get('position_alignment_definition',{}))
    if check!=new:raise ValueError('prefix and full protocols disagree')
    if full['mujoco_version']!=mujoco.__version__ or new['mujoco_version']!=mujoco.__version__:
        raise ValueError('MuJoCo version differs')
    validate_access(new,read(runtime))
    manifest=read(suite);files={f['path']:f for f in manifest['files']}
    ids=[c['id'] for c in full['clips']]
    if set(ids)!=set(files) or len(ids)!=len(files):raise ValueError('full motion suite is incomplete')
    if any(c['truncated_smoke'] or c['trace'].get('sample_limit') is not None for c in full['clips']):
        raise ValueError('partial traces cannot be rescored')
    output.mkdir(parents=True,exist_ok=True);dest=output/'rollout';dest.mkdir()
    start=time.monotonic();started=datetime.now(timezone.utc).isoformat()
    progress=Progress(output/'progress.json')
    layout={k:manifest[k] for k in ('joint_names','source_fps')}
    tasks=[(c,files[c['id']],layout,robot,new,data,run/'rollout',True) for c in full['clips']]
    complete,prefixes=[],[]
    def collect(records):
        for record in records:
            complete.append(record['full']);prefixes.append(record['prefix'])
            progress.emit('rescoring',force=True,completed=len(complete),total=len(tasks),
                          samples=sum(c['sample_count'] for c in complete))
    if jobs>1:
        with ProcessPoolExecutor(max_workers=min(jobs,len(tasks)),mp_context=multiprocessing.get_context('spawn')) as pool:
            collect(pool.map(_score_clip,tasks))
    else:collect(map(_score_clip,tasks))
    max_full_delta = 0.0
    for before,after in zip(full['clips'],complete):
        for key in ('mpjpe_world_m','mpjpe_xy_aligned_m','mpjpe_xy_yaw_aligned_m'):
            if not math.isclose(before['metric_means_over_observed_prefix'][key],after['metric_means_over_observed_prefix'][key],abs_tol=1e-9,rel_tol=1e-7):
                raise ValueError('unchanged full-motion MPBPE differs: '+before['id']+' '+key)
            max_full_delta=max(max_full_delta,abs(before['metric_means_over_observed_prefix'][key]-after['metric_means_over_observed_prefix'][key]))
        src=confined_path(run/'rollout',before['trace']['path']);target=dest/Path(before['trace']['path']).name
        try:os.link(src,target)
        except OSError as e:
            if e.errno!=errno.EXDEV:raise
            shutil.copyfile(src,target)
        after['trace']['path']=target.name
    for clip,whole in zip(prefixes,complete):clip['trace']['path']=whole['trace']['path']
    def result(clips,spec,path,mode):
        return {**full,'clips':clips,'protocol_id':spec['id'],'protocol_sha256':sha256_file(path),
                'termination_mode':mode,'success_count':sum(c['success'] for c in clips),
                'macro_completion':math.fsum(c['completion'] for c in clips)/len(clips),
                'macro_metrics_over_observed_prefix':{k:math.fsum(c['metric_means_over_observed_prefix'][k] for c in clips)/len(clips)
                    for k in clips[0]['metric_means_over_observed_prefix']}}
    whole=result(complete,new,protocol,'no_termination');prefix=result(prefixes,prefix_spec,prefix_protocol,'early_termination')
    # Original execution measurements do not describe rescoring work.
    for record in (whole,prefix):
        for key in ('timings_s','native_batch_calls','native_fk','reference_work','trace_work','adapter_transport'):
            record.pop(key,None)
    combined=combine(prefix,whole,source,original['checkpoint_sha256'])
    derivation={'method':'rescore complete recorded poses; no policy calls or physics steps',
                'original_result_sha256':sha256_file(run/'result.json'),
                'original_rollout_sha256':sha256_file(run/'rollout/result.json'),
                'original_receipt_sha256':sha256_file(run/'rollout/receipt.json'),
                'original_protocol_sha256':sha256_file(original_protocol),
                'full_motion_max_abs_delta_m':max_full_delta,
                'trace_sha256':{c['id']:c['trace']['sha256'] for c in complete}}
    combined.update(execution='rescored_full_trajectory',scoring_source=scoring_source,derivation=derivation)
    if scoring_source!=core_source():raise ValueError('scoring source changed during recomputation')
    finished=datetime.now(timezone.utc).isoformat()
    receipt={'status':'trace_verified','source':source,'scoring_source':scoring_source,
             'source_unchanged_during_run':prior_receipt['source_unchanged_during_run'],
             'full_suite':True,'derivation':derivation,
             'verification':{'verification':'trace_consistent','full_suite':True,'clips':len(complete),
                             'samples':sum(c['sample_count'] for c in complete),'scope':'trace hashes, complete timestamps, canonical FK and new scoring rules'},
             'timings_s':{'rescoring':time.monotonic()-start},'finished_utc':finished}
    summary={'status':'trace_verified','execution':'rescored_full_trajectory','source':source,
             'scoring_source':scoring_source,'started_utc':started,'finished_utc':finished,
             'timings_s':receipt['timings_s'],'derivation':derivation}
    for path,value in [(dest/'result.json',whole),(dest/'prefix-result.json',prefix),(dest/'receipt.json',receipt),
                       (output/'result.json',combined),(output/'summary.json',summary)]:
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(runtime,dest/'runtime.json')
    return combined


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','output','original-protocol','data'):p.add_argument('--'+name,type=Path,required=True)
    paths=lafan_paths('none')
    for name in ('suite','robot','protocol'):p.add_argument('--'+name,type=Path,default=paths[name])
    p.add_argument('--prefix-protocol',type=Path,default=lafan_paths('early')['protocol'])
    p.add_argument('--jobs',type=int,default=8)
    a=p.parse_args();r=rescore(a.run,a.output,a.suite,a.robot,a.original_protocol,a.protocol,a.prefix_protocol,a.data,a.jobs)
    print(json.dumps({'status':r['status'],'success_count':r['success_count'],'total':r['total']}))


if __name__=='__main__':main()
