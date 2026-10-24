"""Keep optimized Core calculations tied to the scalar metric / Jacobian oracles."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from motion_challenge.evaluation.batch_metrics import tracking_batch
from motion_challenge.sdk import BodyState, StatePacket, ReferenceFrame
from motion_challenge.simulation.physics import World, forward_kinematics, tracking_sample

ROOT = Path(__file__).resolve().parents[1]


class BatchMetricTests(unittest.TestCase):
    def test_matches_independent_scalar_scores_and_failure_thresholds(self):
        protocol = json.loads((ROOT/'suites/lafan/protocol.mujoco317-ideal-proprioception-none-v1.json').read_text())
        names = tuple(dict.fromkeys([*protocol['metric_bodies'], protocol['failure']['anchor_body'],
                                    *protocol['failure']['end_effector_bodies']]))
        rng = np.random.default_rng(912)
        states, refs = [], []
        for i in range(257):
            p = rng.normal(size=(len(names),3));q = Rotation.random(len(names),random_state=rng).as_quat()
            bodies = tuple(BodyState(n,tuple(x),tuple(r),(0.,)*3,(0.,)*3) for n,x,r in zip(names,p,q))
            states.append(StatePacket(0.,('j',),(float(rng.normal()),),(0.,),tuple(p[0]),tuple(q[0]),(0.,)*3,(0.,)*3,bodies))
            ref = ReferenceFrame(0.,0.,(0.,),(0.,0.,0.),(0.,0.,0.,1.),bodies=bodies)
            # Identical bodies put height/gravity exactly below the thresholds;
            # alternate samples exercise large rotations, yaw ambiguity and failures.
            if i%2:
                ref = replace(ref,bodies=tuple(replace(b,position_world_m=tuple(rng.normal(size=3)),
                              orientation_xyzw=tuple(Rotation.random(random_state=rng).as_quat())) for b in bodies))
            refs.append(ref)
        for count in (1,3,40,257):
            fast = tracking_batch(states[:count],refs[:count],protocol)
            for state,ref,(failure,values) in zip(states,refs,fast):
                expected_failure,expected = tracking_sample(state,ref,protocol)
                self.assertEqual(failure,expected_failure)
                self.assertEqual(values.keys(),expected.keys())
                np.testing.assert_allclose(list(values.values()),[expected[k] for k in values],rtol=0,atol=4e-15)
        for key,body in [('anchor_height_m',protocol['failure']['anchor_body']),
                         ('end_effector_height_m',protocol['failure']['end_effector_bodies'][0])]:
            for epsilon in (-1e-10,0.,1e-10):
                ref = refs[0]
                state = replace(states[0], bodies=tuple(replace(b,position_world_m=(b.position_world_m[0],b.position_world_m[1],
                    b.position_world_m[2]+protocol['failure'][key]+epsilon)) if b.name==body else b for b in ref.bodies))
                self.assertEqual(tracking_batch([state],[ref],protocol)[0][0],tracking_sample(state,ref,protocol)[0])

    def test_body_origin_velocities_match_jacobians(self):
        world = World(ROOT/'robots/g1-29dof-cpu-v0.json')
        model,data = world.model,world.data
        rng = np.random.default_rng(33)
        jp,jr = np.empty((3,model.nv)),np.empty((3,model.nv))
        for _ in range(100):
            data.qpos[:3] = rng.normal(size=3)
            q = rng.normal(size=4);data.qpos[3:7] = q/np.linalg.norm(q)
            data.qpos[world.qadr] = rng.uniform(-2,2,len(world.qadr))
            data.qvel[:] = rng.normal(size=model.nv)
            forward_kinematics(model,data)
            for bid,body in zip(world.body_ids,world._bodies(data)):
                mujoco.mj_jacBody(model,data,jp,jr,bid)
                np.testing.assert_allclose(body.linear_velocity_world_m_s,jp@data.qvel,rtol=0,atol=8e-15)
                np.testing.assert_allclose(body.angular_velocity_world_rad_s,jr@data.qvel,rtol=0,atol=8e-15)
