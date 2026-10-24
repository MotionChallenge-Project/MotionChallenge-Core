"""Scalar physics oracle for regression tests; never a public evaluation path."""
import argparse
import gzip
import json
from pathlib import Path
from motion_challenge.evaluation.run import clip_steps
from motion_challenge.data.dataset import load_csv_motion
from motion_challenge.simulation.physics import World, tracking_sample
from motion_challenge.runtime.loading import load_controller
from motion_challenge.validation import sha256_file

def evaluate_clip(world, motion, worker, runtime, protocol, max_duration=None, trace=None, on_step=None):
    """Single-environment oracle using the same episode state machine as batch."""
    episode = clip_steps(world, motion, runtime, protocol, max_duration, trace, on_step)
    response = None
    while True:
        try:
            request = episode.send(response)
        except StopIteration as done:
            return done.value
        op, *args = request
        if op == 'reset':
            response = world.reset(*args)
        elif op == 'advance':
            world.advance(*args)
            response = world.state()
        elif op == 'score':
            response = tracking_sample(*args, protocol)
        else:
            response = worker.call('reset' if op == 'policy_reset' else op, *args)


class Worker:
    def __init__(self,path): self.policy=load_controller(path)
    def call(self,op,state,reference,seed=0):
        if op=='reset':return self.policy.reset(state,reference,seed)
        return self.policy.act(state,reference)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('controller','suite','protocol','robot','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args,_=parser.parse_known_args()
    manifest=json.loads(args.suite.read_text());runtime=json.loads(args.controller.read_text())
    protocol=json.loads(args.protocol.read_text());worker=Worker(args.controller);world=World(args.robot)
    clips=[]
    try:
        for index,item in enumerate(manifest['files']):
            motion=load_csv_motion(args.data/item['path'],manifest['joint_names'],manifest['source_fps'])
            path=args.output.with_name(args.output.stem+f'.{index}.gz')
            with gzip.open(path,'wt') as trace:
                result=evaluate_clip(world,motion,worker,runtime,protocol,trace=trace)
            clips.append(dict(id=item['path'],**result,trace=dict(path=path.name,sha256=sha256_file(path))))
        args.output.write_text(json.dumps(dict(clips=clips)))
    finally:worker.policy.close()
