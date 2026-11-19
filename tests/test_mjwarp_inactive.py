import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from motion_challenge.resources import resource_root
from motion_challenge.simulation.warp_physics import WarpPhysics, _pd, _record_overflow, _freeze_inactive
from motion_challenge.validation import sha256_file


DEVICE=os.environ.get('MOTIONCHALLENGE_TEST_DEVICE','cuda:0')


class OriginalActivePhysics(WarpPhysics):
    """Original advance and step equations, before unused-slot freezing."""
    def _advance(self):
        for _ in range(self.substeps):
            wp.launch(_pd,dim=(self.size,len(self.joint_names)),
                      inputs=[self.data.qpos,self.data.qvel,self.command_wp,self.qadr,self.vadr,self.limits],outputs=[self.data.ctrl])
            mjw.step(self.model,self.data)
            wp.launch(_record_overflow,dim=self.size,inputs=[self.data.overflow,self.data.nefc],outputs=[self.overflow_sticky,self.nefc_peak])
        mjw.kinematics(self.model,self.data);mjw.com_pos(self.model,self.data);mjw.com_vel(self.model,self.data)

    def step(self,commands,duration_s):
        self.command.copy_(commands);self.dt.copy_(duration_s/self.substeps)
        with wp.ScopedDevice(str(self.device)),self.stream():
            if self.graph is None:self._advance()
            else:wp.capture_launch(self.graph)
        self.time_s.add_(duration_s.double())


@unittest.skipUnless(torch.cuda.is_available(),'CUDA is required for MJWarp')
class InactiveWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        protocol=json.loads((resource_root()/'suites/lafan/protocol.mjwarp311-no-linear-velocity-v1.json').read_text())
        cls.protocol=protocol;cls.profile=resource_root()/protocol['robot_profile']

    def test_active_free_flight_matches_original_and_refill_reactivates(self):
        for capture in (False,True):
            baseline=OriginalActivePhysics(self.profile,3,DEVICE,capture=capture)
            actual=WarpPhysics(self.profile,3,DEVICE,capture=capture)
            # Avoid contact-order nondeterminism when requiring bitwise equality.
            q=actual.safe_qpos.expand(3,-1).clone();q[[0,2],2]=torch.tensor([5.,5.1],device=DEVICE)
            velocity=torch.zeros((3,actual.world.model.nv),device=DEVICE)
            slots=torch.arange(3,device=DEVICE)
            baseline.reset(slots,q,velocity);actual.reset(slots,q,velocity);actual.park([1])
            command=torch.zeros_like(actual.command);command[:,2]=45.;command[:,3]=2.
            for step in range(36):
                active=[0,2] if step<20 else [0,1,2]
                if step==20:
                    replacement=q[1:2].clone();replacement[:,2]=5.
                    baseline.reset([1],replacement,velocity[1:2]);actual.reset([1],replacement,velocity[1:2])
                    self.assertTrue(bool(actual.active[1]))
                command[:,0]=.08*torch.sin(torch.arange(29,device=DEVICE)+step*.03)
                clean=command.clone();poisoned=command.clone()
                if step<20:clean[1]=0.;poisoned[1]=float('nan')
                duration=torch.full((3,),.007 if step==35 else .02,device=DEVICE,dtype=torch.float64)
                passed=duration.clone()
                if step<20:passed[1]=1e-9
                baseline.step(clean,duration);actual.step(poisoned,passed)
                torch.testing.assert_close(actual.qpos[active],baseline.qpos[active],atol=0,rtol=0)
                torch.testing.assert_close(actual.qvel[active],baseline.qvel[active],atol=0,rtol=0)
                torch.testing.assert_close(actual.time_s[active],baseline.time_s[active],atol=0,rtol=0)
                if step<20:
                    torch.testing.assert_close(actual.qpos[1],actual.safe_qpos,atol=0,rtol=0)
                    self.assertTrue(bool((actual.qvel[1]==0).all()))
                    self.assertTrue(bool((actual.command[1]==0).all()))
                    self.assertEqual(float(actual.time_s[1]),0.)
                    self.assertAlmostEqual(float(actual.dt[1]),.005,places=8)
                    self.assertTrue(bool(torch.isfinite(actual.observations().root_orientation_xyzw).all()))
                actual.check()

    def test_freeze_preserves_every_active_field_in_contact(self):
        physics=WarpPhysics(self.profile,2,DEVICE,capture=True)
        q=physics.safe_qpos.expand(2,-1).clone();q[0,2]=.82
        physics.reset([0,1],q,torch.zeros((2,physics.world.model.nv),device=DEVICE));physics.park([1])
        commands=torch.zeros_like(physics.command);commands[0,2]=40.;commands[0,3]=2.
        for _ in range(20):physics.step(commands,torch.full((2,),.02,device=DEVICE,dtype=torch.float64))
        self.assertGreater(int(wp.to_torch(physics.data.nefc)[0]),0)
        fields=['qpos','qvel','qacc','qacc_warmstart','time']
        before={name:wp.to_torch(getattr(physics.data,name))[0].clone() for name in fields}
        with wp.ScopedDevice(DEVICE),physics.stream():
            wp.launch(_freeze_inactive,dim=(physics.size,max(physics.world.model.nq,physics.world.model.nv)),
                      inputs=[physics.active_wp,physics.safe_qpos_wp],
                      outputs=[physics.data.qpos,physics.data.qvel,physics.data.qacc,physics.data.qacc_warmstart,physics.data.time])
        for name,value in before.items():
            torch.testing.assert_close(wp.to_torch(getattr(physics.data,name))[0],value,atol=0,rtol=0)

    def test_active_finiteness_and_global_overflow_checks(self):
        physics=WarpPhysics(self.profile,2,DEVICE,capture=False);physics.park([1])
        physics.qvel[1,0]=float('nan');physics.check()
        physics.qvel[0,0]=float('nan')
        with self.assertRaisesRegex(FloatingPointError,'Non-finite'):physics.check()
        physics.qvel.zero_();wp.to_torch(physics.overflow_sticky)[1]=1
        with self.assertRaisesRegex(FloatingPointError,'capacity overflow'):physics.check()

    def test_runner_refill_and_nan_actions_in_unused_tail(self):
        from motion_challenge.evaluation.tensor_run import evaluate
        observations=[]
        class Adapter:
            def __init__(self,root,runtime,names,slots,device,weights):
                self.calls=torch.zeros(slots,device=device,dtype=torch.long)
            def reset(self,slots):self.calls[slots]=0
            def act(self,state,reference):
                unused=(state.time_s==0)&(self.calls>0)
                observations.append((len(self.calls),int(unused.sum())))
                self.calls+=1
                command=torch.zeros((len(self.calls),5,29),device=state.time_s.device)
                command[unused]=float('nan')
                return command
        module=types.ModuleType('_inactive_regression_adapter');module.Adapter=Adapter
        sys.modules[module.__name__]=module
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);prepared=root/'prepared';prepared.mkdir();weights=root/'weights';weights.mkdir();(weights/'identity').write_text('synthetic-test')
            (root/'policy.json').write_text('{}')
            (root/'admission.mjwarp.json').write_text(json.dumps({'status':'inference_verified'}))
            runtime={'state_access':self.protocol['state_access'],'tensor_entrypoint':module.__name__+':Adapter','control_period_s':.02,'reference_offsets_s':[0.],
                     'inference_config':'policy.json','inference_config_sha256':sha256_file(root/'policy.json'),'weights_sha256':'synthetic-test',
                     'observation_contract':{'admission_sha256':sha256_file(root/'admission.mjwarp.json')}}
            (root/'runtime.mjwarp.json').write_text(json.dumps(runtime))
            lengths=[2,4,6];poses=np.zeros((sum(lengths),36),np.float32);poses[:,2]=1.;poses[:,6]=1.
            np.save(prepared/'poses.npy',poses)
            (prepared/'dataset.json').write_text(json.dumps({'names':['short','medium','long'],'starts':[0,2,6],'lengths':lengths,'fps':50.,'poses_sha256':sha256_file(prepared/'poses.npy')}))
            with patch('motion_challenge.evaluation.tensor_run.prepare',return_value=prepared):
                result=evaluate(root,'lafan',root,prepared,root/'result',weights,slots=2,device=DEVICE)
            self.assertEqual(len(result['clips']),3)
            self.assertEqual([x['samples'] for x in result['clips']],[1,3,5])
            self.assertTrue(all(size==2 for size,_ in observations))
            self.assertTrue(any(nan_count==1 for _,nan_count in observations))
            self.assertTrue(all(np.isfinite(list(x['full'].values())).all() for x in result['clips']))


if __name__=='__main__':unittest.main()
