"""Array arithmetic against independent scalar/native oracles and slot reuse."""
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from motion_challenge.data.reference import Motion
from motion_challenge.data.array_reference import MotionArena, quaternion_velocity
from motion_challenge.evaluation.accumulator import BatchAccumulator
from motion_challenge.simulation.vector_physics import VectorPhysics
from motion_challenge.simulation.array_reference import ArrayReferences

ROOT = Path(__file__).resolve().parents[1]


class ArrayNumericsTests(unittest.TestCase):
    def test_default_threads_scale_independently_of_slots(self):
        from motion_challenge.simulation.vector_physics import default_threads
        with patch('motion_challenge.simulation.vector_physics.cpu_count',return_value=56):
            self.assertEqual([default_threads(n) for n in (1,3,40,1024,2048,4096,10000)],[1,3,8,8,16,32,32])
        with patch('motion_challenge.simulation.vector_physics.cpu_count',return_value=2):
            self.assertEqual(default_threads(4096),2)

    def test_large_quaternion_velocity_matches_native(self):
        rng = np.random.default_rng(501)
        lo,hi = Rotation.random(8192,random_state=rng).as_quat().reshape(2,4096,4)
        hi[:4] = lo[:4];hi[4:8] = -lo[4:8]
        dt = rng.uniform(.001,.1,4096)
        actual = quaternion_velocity(lo,hi,dt)
        expected = np.empty_like(actual)
        for i in range(len(lo)):
            mujoco.mju_subQuat(expected[i],hi[i,[3,0,1,2]],lo[i,[3,0,1,2]])
        np.testing.assert_allclose(actual,expected/dt[:,None],rtol=3e-15,atol=3e-12)

    def test_arena_and_fk_cache_match_scalar_with_reordered_slots(self):
        rng = np.random.default_rng(52)
        pool = VectorPhysics(ROOT/'robots/g1-29dof-cpu-v0.json',5,2)
        refs = ArrayReferences(pool,5,5)
        for turn in range(3):
            motions = []
            for slot in range(5):
                n = 7+slot*3+turn
                t = np.r_[0.,np.cumsum(rng.uniform(.01,.04,n-1))]
                m = Motion(pool.world.joint_names,t,rng.uniform(-.3,.3,(n,29)),rng.normal(size=(n,3)),
                           Rotation.random(n,random_state=rng).as_quat())
                motions.append(m);refs.load(slot,m)
            ids = np.array([4,1,3,0,2])
            times = rng.uniform(-.1,.8,(5,5))
            for _ in range(2):
                result = refs.get(ids,times)
                for row,slot in enumerate(ids):
                    world = pool.reference_world(motions[slot],32)
                    for col,t in enumerate(times[row]):
                        r = world.reference(round(float(t),12))
                        np.testing.assert_allclose(result.joint_position_rad[row,col],r.joint_position_rad,rtol=0,atol=3e-14)
                        np.testing.assert_allclose(result.root_orientation_xyzw[row,col],r.root_orientation_xyzw,rtol=0,atol=3e-14)
                        np.testing.assert_allclose(result.joint_velocity_rad_s[row,col],r.joint_velocity_rad_s,rtol=0,atol=3e-12)
                        np.testing.assert_allclose(result.root_angular_velocity_world_rad_s[row,col],r.root_angular_velocity_world_rad_s,rtol=0,atol=3e-11)
                        np.testing.assert_allclose(result.body_position_world_m[row,col],r.bodies.values[:,:3],rtol=0,atol=3e-14)
                        np.testing.assert_allclose(result.body_linear_velocity_world_m_s[row,col],r.bodies.values[:,7:10],rtol=0,atol=5e-11)

    def test_batch_expansion_is_fsum_invariant_to_slot_order_and_reset(self):
        rng = np.random.default_rng(71)
        acc = BatchAccumulator(257)
        values = rng.normal(size=(300,257))*np.exp2(rng.integers(-50,50,(300,257)))
        for i in range(len(values)):
            ids = rng.permutation(257)
            acc.add(ids,{'value':values[i,ids],'constant':np.ones(257)})
        for slot in range(257):
            count,means,last = acc.summary(slot)
            self.assertEqual(count,300)
            self.assertEqual(means['value'],math.fsum(values[:,slot])/300)
            self.assertEqual(means['constant'],1.)
        acc.reset([3,90])
        acc.add(np.array([90,3]),{'value':np.array([2.,3.]),'constant':np.array([1.,1.])})
        self.assertEqual(acc.summary(3)[1]['value'],3.)
        self.assertEqual(acc.summary(90)[0],1)

    def test_physics_substeps_do_not_bind_body_outputs_or_run_forward(self):
        pool = VectorPhysics(ROOT/'robots/g1-29dof-cpu-v0.json',3,2)
        pool.qpos[:,2]=.8
        pool.advance([0,1,2],np.zeros((3,5,29)),[.02,.013,.007])
        self.assertEqual(set(pool.batch._bound),set())
        self.assertNotIn('forward',pool.native_timings)
        self.assertNotIn('jac',pool.native_timings)
        pool.read_arrays([2,0])
        self.assertEqual(set(pool.batch._bound),set())
