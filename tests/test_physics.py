import math
from pathlib import Path
import unittest

import numpy as np
import mujoco

from motion_challenge.physics import World, forward_kinematics, tracking_sample
from motion_challenge.reference import Motion
from motion_challenge.sdk import ControlCommand


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]
        self.world=World(self.root/'robots/g1-29dof-cpu-v0.json')

    def motion(self,duration=1/30):
        q=(0.,)*29
        return Motion(self.world.joint_names,(0.,duration),(q,q),
                      ((0.,0.,.8),(0.,0.,.8)),((0.,0.,0.,1.),(0.,0.,0.,1.)))

    def test_exact_fractional_final_time_and_post_step_fk(self):
        motion=self.motion()
        self.world.reset(motion)
        self.world.advance(ControlCommand((0.,)*29,(0.,)*29,(0.,)*29,(0.,)*29,(0.,)*29),motion.duration_s)
        self.assertAlmostEqual(self.world.state().time_s,motion.duration_s,places=12)
        state=self.world.state()
        self.assertLess(state.root_position_world_m[2],.8)
        np.testing.assert_allclose(state.root_position_world_m,self.world.data.qpos[:3],atol=1e-12)
        self.assertEqual(self.world.model.opt.timestep,.002)

    def test_constant_reference_velocity_and_episode_reset(self):
        motion=self.motion()
        self.world.reset(motion)
        for t in (-.2,0.,.01,motion.duration_s,.2):
            f=self.world.reference(t)
            np.testing.assert_allclose(f.joint_velocity_rad_s,0,atol=1e-12)
            for b in f.bodies:
                np.testing.assert_allclose(b.linear_velocity_world_m_s,0,atol=1e-12)
        self.world.reset(motion)
        self.assertEqual(self.world.state().time_s,0.)

    def test_kinematic_packets_match_full_dynamics_at_random_poses(self):
        rng = np.random.default_rng(317)
        world = self.world
        for _ in range(80):
            # Includes ground penetration and high velocities: the packet must
            # not depend on the contact/constraint solve that is being skipped.
            q = world.model.qpos0.copy()
            q[:3] = rng.uniform(-1, 1, 3)
            quaternion = rng.normal(size=4)
            q[3:7] = quaternion / np.linalg.norm(quaternion)
            q[world.qadr] = rng.uniform(-2, 2, len(world.joints))
            world.data.qpos[:] = q
            world.data.qvel[:] = rng.normal(size=world.model.nv) * 4
            forward_kinematics(world.model, world.data)
            packet = world.state()
            mujoco.mj_forward(world.model, world.data)
            self.assertEqual(packet, world.state())

    def test_failure_uses_reference_gravity_z_not_yaw_angle(self):
        import json
        from dataclasses import replace
        protocol=json.loads((self.root/'suites/lafan/protocol.mujoco317-v1.json').read_text())
        state=self.world.reset(self.motion())
        reference=self.world.reference(0)
        modified=tuple(replace(b,orientation_xyzw=(0.,0.,1.,0.))
                       if b.name=='torso_link' else b for b in state.bodies)
        failures,_=tracking_sample(replace(state,bodies=modified),reference,protocol)
        self.assertNotIn('anchor_gravity_z',failures)
        modified=tuple(replace(b,orientation_xyzw=(1.,0.,0.,0.))
                       if b.name=='torso_link' else b for b in state.bodies)
        failures,_=tracking_sample(replace(state,bodies=modified),reference,protocol)
        self.assertIn('anchor_gravity_z',failures)

    def test_both_rollout_modes_share_failures_and_alignment_scores(self):
        import json
        from dataclasses import replace
        folder = self.root/'suites/lafan'
        old = json.loads((folder/'protocol.mujoco317-v1.json').read_text())
        new = json.loads((folder/'protocol.mujoco317-no-termination-v1.json').read_text())
        self.assertEqual(old['failure'],new['failure'])
        state = self.world.reset(self.motion())
        state = replace(state,bodies=tuple(replace(b,position_world_m=(
            b.position_world_m[0],b.position_world_m[1],b.position_world_m[2]-.4))
            for b in state.bodies))
        reference = self.world.reference(0)
        failure0,metrics0 = tracking_sample(state,reference,old)
        failure1,metrics1 = tracking_sample(state,reference,new)
        self.assertTrue(failure0)
        self.assertEqual(failure0,failure1)
        self.assertEqual(metrics0,metrics1)
        for name in ('mpjpe_world_m','mpjpe_xy_aligned_m','mpjpe_xy_yaw_aligned_m'):
            self.assertAlmostEqual(metrics1[name],.4)


if __name__=='__main__':unittest.main()
