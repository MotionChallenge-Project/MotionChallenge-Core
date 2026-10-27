"""One array control loop; per-clip Python work happens only at load/finish."""
import heapq
import json
import math
import time
import numpy as np
from motion_challenge.data.dataset import load_csv_motion
from motion_challenge.evaluation.accumulator import BatchAccumulator
from motion_challenge.evaluation.batch_metrics import tracking_arrays, FAILURE_NAMES
from motion_challenge.evaluation.array_traces import ArrayTraceWriter
from motion_challenge.runtime.policy import PolicyBatch, array_slice
from motion_challenge.runtime.progress import Progress
from motion_challenge.simulation.vector_physics import VectorPhysics
from motion_challenge.simulation.array_reference import ArrayReferences
from motion_challenge.validation import confined_path, sha256_file


def motion_frames(path):
    with path.open('rb') as stream:
        return sum(1 for line in stream if line.strip())


def evaluate_batch(args, selected, results, protocol, full_suite):
    if not selected: raise ValueError('empty motion selection')
    started = time.monotonic()
    size = min(args.batch_size,len(selected))
    runtime = json.loads(args.controller.read_text())
    manifest = json.loads(args.suite.read_text())
    cfg = protocol['failure']
    score_names = (cfg['anchor_body'],*protocol['metric_bodies'])
    requested_bodies = runtime.get('reference_bodies')
    if requested_bodies is not None and (not runtime.get('array_batch_entrypoint') or
            not isinstance(requested_bodies,list) or any(not isinstance(n,str) for n in requested_bodies)):
        raise ValueError('reference_bodies requires an array adapter and a list of body names')
    pool = VectorPhysics(args.robot,size,args.threads,set(score_names),requested_bodies)
    offsets = np.asarray(runtime['reference_offsets_s'])
    refs = ArrayReferences(pool,size,len(offsets)+1,
        (runtime['control_period_s'],protocol['metric_period_s']),offsets)
    actual_columns = [pool.body_names.index(n) for n in score_names]
    ref_columns = [refs.body_names.index(n) for n in score_names]
    writer = None if getattr(args,'no_traces',False) else ArrayTraceWriter()
    progress = Progress(args.output.with_suffix('.progress.json'),args.progress_interval)
    worker = None
    queue,items,finished,prefix = [],{},{},{}
    alive = np.zeros(size,dtype=bool)
    duration,end,starts,last_valid,last_sample = (np.zeros(size) for _ in range(5))
    failure_time = np.full(size,np.nan)
    reasons = FAILURE_NAMES
    failures = np.zeros((size,len(reasons)),dtype=bool)
    control_i,metric_i = np.ones(size,dtype=np.int64),np.ones(size,dtype=np.int64)
    commands = np.empty((size,5,len(pool.world.joint_names)))
    sums = BatchAccumulator(size)
    period,metric_dt = runtime['control_period_s'],protocol['metric_period_s']
    terminate = protocol.get('terminate_on_failure',True)
    if not isinstance(terminate,bool): raise ValueError('terminate_on_failure must be boolean')
    timings = dict(preflight=0.,dataset_load=0.,metrics=0.,reference_trace=0.,physics=0.,policy=0.,trace_finalize=0.)
    results.update(batch_size=size,batch_threads=pool.batch.nthread,
        execution='vector_loop_in_process_adapter',policy_execution='direct_call',
        scheduling='longest_first_dynamic_refill')

    def report(force=False):
        if not force and time.monotonic()-progress.last.get(('evaluation',None),-float('inf'))<progress.interval: return
        ids = np.flatnonzero(alive)
        progress.emit('evaluation',force=force,completed=len(finished),total=len(selected),
            passed=sum(c['success'] for c in finished.values()),active_count=len(ids),active_preview_limit=16,
            active={str(i):dict(clip=items[i]['clip']['path'],time_s=float(pool.time[i]),duration_s=duration[i]) for i in ids[:16]},
            backend='batch',timings_s={**timings,'elapsed':time.monotonic()-started},
            reference_work=pool.world.reference_work,native_fk=pool.fk.timings,
            trace_work={} if writer is None else writer.timings,native_batch_calls=pool.native_timings,
            policy_phases={} if worker is None else worker.telemetry)

    def reference(ids,times):
        tick = time.monotonic(); value = refs.get(ids,times)
        timings['reference_trace'] += time.monotonic()-tick
        return value

    def policy(op,ids,state,reference,seeds=None):
        nonlocal worker
        tick = time.monotonic()
        if worker is None:
            worker = PolicyBatch(args.controller,size,state,reference,args.output.with_suffix('.adapter.log'))
            results['inference_batch_size'] = worker.capacity
            results['policy_interface'] = ('arrays' if runtime.get('array_batch_entrypoint') else
                'packets' if runtime.get('batch_entrypoint') else 'scalar')
        value = worker.call_arrays(op,ids,state,reference,seeds)
        timings['policy'] += time.monotonic()-tick
        return value

    def refill(free):
        tick = time.monotonic(); loaded = []
        for slot in free:
            if not queue: break
            _,index,clip = heapq.heappop(queue)
            path = confined_path(args.data,clip['path'])
            if sha256_file(path)!=clip['sha256']: raise ValueError('dataset integrity mismatch')
            motion = load_csv_motion(path,manifest['joint_names'],manifest['source_fps'])
            refs.load(slot,motion)
            duration[slot]=motion.duration_s
            end[slot]=motion.duration_s if args.max_duration is None else min(args.max_duration,motion.duration_s)
            trace_path = args.output.with_name(args.output.stem+f'.clip{index:05d}.poses.jsonl.gz')
            if writer is not None: writer.open(int(slot),trace_path)
            items[slot] = dict(index=index,clip=clip,path=trace_path)
            loaded.append(slot)
        timings['dataset_load'] += time.monotonic()-tick
        if not loaded: return
        ids = np.asarray(loaded,dtype=np.int32)
        tick = time.monotonic()
        pool.reset_arrays(ids,refs.reset_position[ids],refs.initial_velocity(ids))
        state,_,_ = pool.read_arrays(ids)
        timings['physics'] += time.monotonic()-tick
        alive[ids]=True;control_i[ids]=1;metric_i[ids]=1
        failure_time[ids]=np.nan;failures[ids]=False;last_valid[ids]=0.;last_sample[ids]=0.
        sums.reset(ids)
        for slot in ids: prefix.pop(slot,None)
        window = reference(ids,np.broadcast_to(offsets,(len(ids),len(offsets))))
        policy('reset',ids,state,window,np.zeros(len(ids),dtype=int))
        commands[ids] = policy('act',ids,state,window)
        starts[ids] = time.monotonic()

    def finish(ids):
        tick = time.monotonic()
        if writer is not None: writer.finish(tuple(map(int,ids)))
        for slot in ids:
            count,means,last = sums.summary(slot)
            item = items.pop(slot)
            last = {'time_s':float(last_sample[slot]),**last} if last is not None else None
            result = dict(success=bool(np.isnan(failure_time[slot]) and last_valid[slot]>=duration[slot]-1e-9),
                duration_s=float(duration[slot]),observed_time_s=float(pool.time[slot]),
                completion=float(last_valid[slot]/duration[slot]),
                failure_time_s=None if np.isnan(failure_time[slot]) else float(failure_time[slot]),
                failure_reasons=[r for r,failed in zip(reasons,failures[slot]) if failed],sample_count=count,
                metric_means_over_observed_prefix=means,last_sample=last,
                metric_prefix=prefix.get(slot,dict(sample_count=count,observed_time_s=float(pool.time[slot]),
                    metric_means_over_observed_prefix=means,last_sample=last)),
                wall_time_s=time.monotonic()-starts[slot],truncated_smoke=bool(args.max_duration is not None and end[slot]<duration[slot]))
            if writer is not None: result['trace']=dict(path=item['path'].name,sha256=sha256_file(item['path']))
            finished[item['index']]=dict(id=item['clip']['path'],**result)
        alive[ids]=False
        timings['trace_finalize'] += time.monotonic()-tick

    try:
        for index,clip in enumerate(selected):
            path = confined_path(args.data,clip['path'])
            if sha256_file(path)!=clip['sha256']: raise ValueError('dataset integrity mismatch')
            heapq.heappush(queue,(-motion_frames(path),index,clip))
        timings['preflight']=time.monotonic()-started
        refill(np.arange(size))
        while alive.any():
            ids = np.flatnonzero(alive)
            targets = np.minimum(np.minimum(control_i[ids]*period,metric_i[ids]*metric_dt),end[ids])
            tick = time.monotonic()
            pool.advance(ids,commands[ids],targets)
            state,root_pos,bodies = pool.read_arrays(ids)
            timings['physics'] += time.monotonic()-tick
            score = (targets>=metric_i[ids]*metric_dt-1e-9)|(targets>=end[ids]-1e-9)
            score_ids = ids[score]
            if len(score_ids):
                desired = reference(score_ids,targets[score,None])
                tick = time.monotonic()
                db = np.concatenate((desired.body_position_world_m[:,0,ref_columns],
                                     desired.body_orientation_xyzw[:,0,ref_columns]),axis=-1)
                failed,values = tracking_arrays(bodies[score][:,actual_columns],db,
                    state.joint_position_rad[score],desired.joint_position_rad[:,0],root_pos[score],
                    desired.root_position_world_m[:,0],state.root_orientation_xyzw[score],desired.root_orientation_xyzw[:,0],protocol)
                sums.add(score_ids,values)
                first = failed.any(axis=1)&np.isnan(failure_time[score_ids])
                first_ids = score_ids[first]
                failures[first_ids] = failed[first]
                failure_time[first_ids] = state.time_s[score][first]
                last_sample[score_ids] = targets[score]
                for slot in first_ids:
                    count,means,last = sums.summary(slot)
                    prefix[slot] = dict(sample_count=count,observed_time_s=float(pool.time[slot]),
                        metric_means_over_observed_prefix=means,last_sample={'time_s':float(last_sample[slot]),**last})
                valid = np.isnan(failure_time[score_ids])
                last_valid[score_ids[valid]] = targets[score][valid]
                metric_i[score_ids] += 1
                timings['metrics'] += time.monotonic()-tick
                if writer is not None:
                    writer.write(score_ids,targets[score],state.joint_position_rad[score],root_pos[score],state.root_orientation_xyzw[score])
            done = (targets>=end[ids]-1e-9)|((~np.isnan(failure_time[ids]))&terminate)
            act = (~done)&(targets>=control_i[ids]*period-1e-9)
            if act.any():
                act_ids = ids[act]
                window = reference(act_ids,targets[act,None]+offsets)
                commands[act_ids] = policy('act',act_ids,array_slice(state,act),window)
                control_i[act_ids] += 1
            if done.any():
                free = ids[done];finish(free);refill(free)
            report()
        report(True)
        results['clips']=[finished[i] for i in range(len(selected))]
        results.update(status='completed',full_suite=full_suite,success_count=sum(c['success'] for c in results['clips']),
            macro_completion=math.fsum(c['completion'] for c in results['clips'])/len(selected))
        if protocol.get('position_alignment_metrics'):
            results['macro_metrics_over_observed_prefix']={
                key:math.fsum(c['metric_means_over_observed_prefix'][key] for c in results['clips'])/len(selected)
                for key in results['clips'][0]['metric_means_over_observed_prefix']}
    except BaseException as error:
        results.update(status='infrastructure_error',error=f'{type(error).__name__}: {error}',clips=[finished[i] for i in sorted(finished)])
        raise
    finally:
        try:
            if worker is not None: worker.close()
            if writer is not None: writer.close()
        finally:
            timings['total']=time.monotonic()-started
            timings['dispatch']=timings['total']-sum(v for k,v in timings.items() if k!='total')
            results['timings_s']=timings
            results['native_batch_calls']=pool.native_timings
            results['native_fk']=pool.fk.timings
            results['reference_work']=pool.world.reference_work
            results['trace_work']={} if writer is None else writer.timings
            if worker is not None:
                results['adapter_transport']=dict(mechanism='in_process_direct_call',
                    input_buffer_bytes=worker.input_buffer_bytes if worker.array_mode else 0,phases=worker.telemetry)
            args.output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
