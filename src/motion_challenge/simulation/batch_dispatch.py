"""A bounded batch loop with longest-first dynamic refill and direct policy calls."""
import heapq
from itertools import islice
import json
import math
from pathlib import Path
import time

from motion_challenge.data.dataset import load_csv_motion
from motion_challenge.evaluation.run import clip_steps
from motion_challenge.evaluation.batch_metrics import tracking_batch
from motion_challenge.evaluation.traces import TraceWriter
from motion_challenge.runtime.progress import Progress
from motion_challenge.runtime.policy import PolicyBatch
from motion_challenge.simulation.vector_physics import VectorPhysics
from motion_challenge.validation import confined_path, sha256_file


def motion_frames(path):
    """Bounded-memory preflight for CSV suites without frame-count metadata."""
    with path.open('rb') as stream:
        return sum(1 for line in stream if line.strip())


def evaluate_batch(args, selected, results, protocol, full_suite):
    if not selected:
        raise ValueError('empty motion selection')
    started = time.monotonic()
    size = min(args.batch_size,len(selected))
    runtime = json.loads(args.controller.read_text())
    manifest = json.loads(args.suite.read_text())
    scoring_bodies = set(protocol['metric_bodies']) | {protocol['failure']['anchor_body']}
    scoring_bodies.update(protocol['failure']['end_effector_bodies'])
    pool = VectorPhysics(args.robot,size,args.threads,scoring_bodies)
    writer = None if getattr(args,'no_traces',False) else TraceWriter()
    progress = Progress(args.output.with_suffix('.progress.json'),args.progress_interval)
    worker = None
    queue,active,requests,finished = [],{},{},{}
    timings = dict(preflight=0.,dataset_load=0.,metrics=0.,reference_trace=0.,physics=0.,policy=0.,trace_finalize=0.)
    results.update(batch_size=size,batch_threads=pool.batch.nthread,
        execution='vector_loop_in_process_adapter',policy_execution='direct_call',
        scheduling='longest_first_dynamic_refill')

    def report(force=False):
        if not force and time.monotonic()-progress.last.get(('evaluation',None),-float('inf')) < progress.interval:
            return
        progress.emit('evaluation',force=force,completed=len(finished),total=len(selected),
            passed=sum(c['success'] for c in finished.values()),
            active_count=len(active),active_preview_limit=16,
            active={str(i):dict(clip=x['clip']['path'],time_s=float(pool.time[i]),duration_s=x['duration'])
                    for i,x in islice(active.items(),16)},backend='batch',
            timings_s={**timings,'elapsed':time.monotonic()-started},
            reference_work=pool.world.reference_work,
            trace_work=pool.world.trace_work,
            native_batch_calls=pool.native_timings,
            policy_phases={} if worker is None else worker.telemetry)

    def refill(slot):
        if not queue:
            return
        load_started = time.monotonic()
        _,index,clip = heapq.heappop(queue)
        path = confined_path(args.data,clip['path'])
        # Recheck at use time, so preflight cannot hide a changed file.
        if sha256_file(path)!=clip['sha256']:
            raise ValueError('dataset integrity mismatch')
        motion = load_csv_motion(path,manifest['joint_names'],manifest['source_fps'])
        timings['dataset_load'] += time.monotonic()-load_started
        world = pool.reference_world(motion,max(8,2*len(runtime['reference_offsets_s'])+4))
        trace_path = args.output.with_name(args.output.stem+f'.clip{index:05d}.poses.jsonl.gz')
        trace = writer.open(trace_path) if writer is not None else None
        episode = clip_steps(world,motion,runtime,protocol,args.max_duration,trace)
        active[slot] = dict(index=index,clip=clip,world=world,episode=episode,trace=trace,
                            path=trace_path,duration=motion.duration_s)
        requests[slot] = next(episode)

    def resume(slot,response):
        item = active[slot]
        try:
            requests[slot] = item['episode'].send(response)
        except StopIteration as done:
            requests.pop(slot,None)
            t = time.monotonic()
            if item['trace'] is not None:
                item['trace'].close()
            timings['trace_finalize'] += time.monotonic()-t
            result = done.value
            if item['trace'] is not None:
                result['trace'] = dict(path=item['path'].name,sha256=sha256_file(item['path']))
            finished[item['index']] = dict(id=item['clip']['path'],**result)
            del active[slot]
            report()
            refill(slot)

    try:
        for index,clip in enumerate(selected):
            path = confined_path(args.data,clip['path'])
            if sha256_file(path)!=clip['sha256']:
                raise ValueError('dataset integrity mismatch')
            frames = motion_frames(path)
            heapq.heappush(queue,(-frames,index,clip))
        timings['preflight'] = time.monotonic()-started
        for slot in range(size):
            refill(slot)
        while active:
            for op in ('reset','policy_reset','act','advance','score'):
                ids = sorted(i for i,r in requests.items() if r[0]==op)
                if not ids:
                    continue
                t = time.monotonic()
                if op=='reset':
                    pool.reset(ids,[active[i]['world'] for i in ids])
                    responses = pool.states(ids)
                    timings['physics'] += time.monotonic()-t
                elif op=='advance':
                    pool.advance(ids,[requests[i][1] for i in ids],[requests[i][2] for i in ids])
                    responses = pool.states(ids)
                    timings['physics'] += time.monotonic()-t
                elif op=='score':
                    responses = tracking_batch([requests[i][1] for i in ids],
                                               [requests[i][2] for i in ids],protocol)
                    timings['metrics'] += time.monotonic()-t
                else:
                    states = [requests[i][1] for i in ids]
                    refs = [requests[i][2] for i in ids]
                    if worker is None:
                        worker = PolicyBatch(args.controller,size,states[0],refs[0],
                            args.output.with_suffix('.adapter.log'))
                    results['inference_batch_size'] = worker.capacity
                    results['policy_interface'] = ('arrays' if runtime.get('array_batch_entrypoint') else
                        'packets' if runtime.get('batch_entrypoint') else 'scalar')
                    if op=='policy_reset':
                        worker.call('reset',ids,states,refs,[requests[i][3] for i in ids])
                        responses = [None]*len(ids)
                    else:
                        responses = worker.call('act',ids,states,refs)
                    timings['policy'] += time.monotonic()-t
                t = time.monotonic()
                trace_time = timings['trace_finalize']+timings['dataset_load']
                for slot,response in zip(ids,responses):
                    resume(slot,response)
                timings['reference_trace'] += time.monotonic()-t-(timings['trace_finalize']+timings['dataset_load']-trace_time)
            report()
        report(True)
        results['clips'] = [finished[i] for i in range(len(selected))]
        results.update(status='completed',full_suite=full_suite,
            success_count=sum(c['success'] for c in results['clips']),
            macro_completion=math.fsum(c['completion'] for c in results['clips'])/len(selected))
        if protocol.get('position_alignment_metrics'):
            results['macro_metrics_over_observed_prefix'] = {
                key:math.fsum(c['metric_means_over_observed_prefix'][key] for c in results['clips'])/len(selected)
                for key in results['clips'][0]['metric_means_over_observed_prefix']}
    except BaseException as error:
        results.update(status='infrastructure_error',error=f'{type(error).__name__}: {error}',
                       clips=[finished[i] for i in sorted(finished)])
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
            for item in active.values():
                item['episode'].close()
            if writer is not None:
                writer.close()
        finally:
            timings['total'] = time.monotonic()-started
            results['timings_s'] = timings
            results['native_batch_calls'] = pool.native_timings
            results['reference_work'] = pool.world.reference_work
            results['trace_work'] = pool.world.trace_work
            if worker is not None:
                results['adapter_transport'] = dict(
                    mechanism='in_process_direct_call',
                    input_buffer_bytes=worker.buffer.size_bytes if worker.buffer is not None else 0,
                    phases=worker.telemetry)
            args.output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
