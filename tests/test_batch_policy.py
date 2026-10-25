"""Optional model batching must preserve per-slot state and core-owned physics."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from motion_challenge.data.bodies import BodyArray
from motion_challenge.sdk import ProprioceptiveStatePacket, ReferenceFrame

ROOT = Path(__file__).resolve().parents[1]


class BatchPolicyTests(unittest.TestCase):
    def test_arbitrary_slot_counts_partial_resets_and_invalid_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root/'adapter.py').write_text('''from motion_challenge.sdk import ControlCommand, ControlBatch
import numpy as np
class Single:
    def reset(self, state, reference, seed): self.count = 0
    def act(self, state, reference):
        self.count += 1
        n = len(state.joint_names)
        return ControlCommand((self.count*.001,)*n,(0.,)*n,(20.,)*n,(1.,)*n,(0.,)*n)
    def close(self): pass
class Batched:
    def __init__(self, config): self.policies = {}; self.fail = config.get('fail',False)
    def reset_batch(self, slots, states, references, seeds):
        for slot,state,ref,seed in zip(slots,states,references,seeds):
            self.policies[slot] = Single()
            self.policies[slot].reset(state,ref,seed)
    def act_batch(self, slots, states, references):
        if self.fail: return ()
        return tuple(self.policies[slot].act(state,ref) for slot,state,ref in zip(slots,states,references))
    def close(self): pass
class Arrays:
    def __init__(self, config): self.count = {}; self.fail = config.get('fail',False)
    def reset_batch(self, slots, states, references, seeds):
        assert not hasattr(states,'root_position_world_m')
        assert not states.joint_position_rad.flags.writeable
        assert references.root_position_world_m.shape == (len(slots),2,3)
        for slot in slots: self.count[slot] = 0
    def act_batch(self, slots, states, references):
        for slot in slots: self.count[slot] += 1
        q = np.array([self.count[s]*.001 for s in slots])[:,None]*np.ones_like(states.joint_position_rad)
        if self.fail: q[:] = np.nan
        return ControlBatch(q,np.zeros_like(q),np.full_like(q,20.),np.ones_like(q),np.zeros_like(q))
    def close(self): pass
def create_parallel(root, config):
    from motion_challenge.adapters.parallel import ParallelControllers
    return ParallelControllers(root, config, create)
def create_array(root, config): return Arrays(config)
def create(root, config): return Single()
def create_batch(root, config): return Batched(config)
''')
            manifest = json.loads((ROOT/'suites/lafan/dataset.json').read_text())
            manifest['files'] = []
            row = ','.join(str(x) for x in ([0,0,.8,0,0,0,1]+[0]*29))+'\n'
            for i in range(9):
                path = root/f'clip{i}.csv'
                path.write_text(row*(2+i%4))
                manifest['files'].append({'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            (root/'suite.json').write_text(json.dumps(manifest))
            cfg = root/'runtime.json'
            runtime = {'state_access':'ideal-proprioceptive-no-position-v1', 'id':'test','adapter_root':str(root),'entrypoint':'adapter:create',
                       'control_period_s':.013,'reference_offsets_s':[0.,.013],
                       'reference_history_s':0,'reference_preview_s':.013}
            env = dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            base = [sys.executable,'-m','motion_challenge.run','--controller',str(cfg),
                    '--suite',str(root/'suite.json'),'--data',str(root),
                    '--robot',str(ROOT/'robots/g1-29dof-cpu-v0.json'),
                    '--protocol',str(ROOT/'suites/lafan/protocol.mujoco317-ideal-proprioception-none-v1.json')]
            previous = None
            for name,jobs,limit in [('fallback',2,None),('single_batch',1,1),('microbatch',3,2),('wide',7,5),('parallel',5,3),('arrays',5,3),('no_bodies',5,3),('body_subset',5,3)]:
                if limit is not None:
                    runtime.update(batch_entrypoint='adapter:create_batch',inference_batch_size=limit)
                if name == 'parallel':
                    runtime['batch_entrypoint']='adapter:create_parallel'
                    runtime['inference_workers']=3
                if name == 'arrays':
                    runtime['array_batch_entrypoint']='adapter:create_array'
                if name == 'no_bodies':
                    runtime['reference_bodies']=[]
                if name == 'body_subset':
                    runtime['reference_bodies']=['torso_link','pelvis']
                cfg.write_text(json.dumps(runtime))
                output = root/(name+'.json')
                result = subprocess.run(base+['--batch-size',str(jobs),'--output',str(output)],env=env,
                                        capture_output=True,text=True,timeout=40)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                current = json.loads(output.read_text())
                transport = current['adapter_transport']
                self.assertEqual(transport['mechanism'],'in_process_direct_call')
                self.assertEqual(transport['input_buffer_bytes']>0,name in ('arrays','no_bodies','body_subset'))
                for phase in transport['phases'].values():
                    self.assertGreater(phase['requests'],0)
                    self.assertEqual(phase['ipc_s'],0.)
                    self.assertGreater(phase['adapter_s'],0.)
                    self.assertGreaterEqual(phase['input_assembly_s'],0.)
                if previous is not None:
                    for a,b in zip(previous['clips'],current['clips']):
                        for key in a:
                            if key not in ('wall_time_s','trace'):
                                self.assertEqual(a[key],b[key],(name,a['id'],key))
                        with gzip.open(root/a['trace']['path'],'rt') as f:before=f.read()
                        with gzip.open(root/b['trace']['path'],'rt') as f:after=f.read()
                        self.assertEqual(before,after)
                previous = current
            runtime['fail']=True;cfg.write_text(json.dumps(runtime))
            result = subprocess.run(base+['--batch-size','3','--output',str(root/'bad.json')],
                                    env=env,capture_output=True,text=True,timeout=40)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(json.loads((root/'bad.json').read_text())['status'],'infrastructure_error')
