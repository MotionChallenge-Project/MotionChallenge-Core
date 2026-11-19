"""Current tensor motion sampling against independent CPU/native arithmetic."""
import unittest

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motion_challenge.data.array_reference import quaternion_velocity as numpy_quaternion_velocity
from motion_challenge.data.reference import Motion
from motion_challenge.resources import resource_root
from motion_challenge.simulation.physics import World
from motion_challenge.simulation.tensor_math import quaternion_velocity
from motion_challenge.simulation.tensor_reference import TensorMotionBank, TensorReference


class TensorReferenceTests(unittest.TestCase):
    def test_quaternion_velocity_matches_native_for_large_rotations_and_signs(self):
        rng = np.random.default_rng(501)
        low, high = Rotation.random(8192, random_state=rng).as_quat().reshape(2, 4096, 4)
        high[:4] = low[:4]
        high[4:8] = -low[4:8]
        dt = rng.uniform(.001, .1, 4096)
        expected = np.empty((4096, 3))
        for index in range(len(low)):
            mujoco.mju_subQuat(expected[index], high[index, [3, 0, 1, 2]], low[index, [3, 0, 1, 2]])
        expected /= dt[:, None]
        tensor = quaternion_velocity(torch.from_numpy(low), torch.from_numpy(high), torch.from_numpy(dt))
        np.testing.assert_allclose(tensor.numpy(), expected, rtol=3e-15, atol=3e-12)
        np.testing.assert_allclose(numpy_quaternion_velocity(low, high, dt), expected, rtol=3e-15, atol=3e-12)

    def bank(self):
        rng = np.random.default_rng(511)
        lengths = np.array([3, 4, 5])
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        fps = np.array([30., 60., 120.])
        poses = rng.normal(0, .1, (lengths.sum(), 36)).astype(np.float32)
        poses[:, 2] += .8
        poses[:, 3:7] = Rotation.random(len(poses), random_state=rng).as_quat()
        poses[::2, 3:7] *= -1
        bank = TensorMotionBank(poses, starts, lengths, fps, ('a', 'b', 'c'), 'cpu')
        return bank, poses, starts, lengths, fps

    def test_reordered_motion_sampling_and_clamped_endpoints(self):
        bank, poses, starts, lengths, fps = self.bank()
        ids = torch.tensor([2, 0, 1])
        requested = torch.tensor([[-.1, 0., .013, .3]] * 3, dtype=torch.float64)
        times, position, quaternion, joints = bank.sample(ids[:, None], requested)
        names = tuple(f'j{i}' for i in range(29))
        for row, clip in enumerate(ids.tolist()):
            source = poses[starts[clip]:starts[clip] + lengths[clip]]
            motion = Motion(names, np.arange(lengths[clip]) / fps[clip], source[:, 7:], source[:, :3], source[:, 3:7])
            expected_t, expected_j, expected_p, expected_q = motion.sample_arrays(requested[row].numpy())
            np.testing.assert_allclose(times[row].numpy(), expected_t, atol=1e-14, rtol=0)
            np.testing.assert_allclose(position[row].numpy(), expected_p, atol=1e-14, rtol=0)
            np.testing.assert_allclose(joints[row].numpy(), expected_j, atol=1e-14, rtol=0)
            actual_q = quaternion[row].numpy()
            np.testing.assert_allclose(abs(np.sum(actual_q * expected_q, axis=-1)), 1., atol=1e-14, rtol=0)

    def test_initial_velocity_matches_mujoco_and_motion_api_does_not_allocate_fk(self):
        bank, _, _, _, _ = self.bank()
        world = World(resource_root() / 'robots/g1-29dof-mjwarp-v1.json')
        reference = TensorReference(world, bank, 6, 'cpu')
        ids = torch.tensor([2, 0, 1])
        qpos, qvel = reference.initial(ids)
        zero = torch.zeros(3, dtype=torch.float64)
        interval = 1 / bank.fps[ids]
        _, p0, r0, j0 = bank.sample(ids, zero)
        _, p1, r1, j1 = bank.sample(ids, interval)
        for row in range(3):
            start, end = world.model.qpos0.copy(), world.model.qpos0.copy()
            for destination, p, q, j in [(start, p0, r0, j0), (end, p1, r1, j1)]:
                destination[:3] = p[row].numpy()
                destination[3:7] = q[row, [3, 0, 1, 2]].numpy()
                destination[world.qadr] = j[row].numpy()
            expected = np.empty(world.model.nv)
            mujoco.mj_differentiatePos(world.model, expected, float(interval[row]), start, end)
            np.testing.assert_array_equal(qpos[row].numpy(), start.astype(np.float32))
            np.testing.assert_allclose(qvel[row].numpy(), expected.astype(np.float32), atol=2e-5, rtol=2e-7)
        result = reference.motion(ids, torch.tensor([[0., .02]] * 3, dtype=torch.float64))
        self.assertEqual(result.joint_position_rad.shape, (3, 2, 29))
        self.assertFalse(hasattr(result, 'body_position_world_m'))
        self.assertIsNone(reference.data)
        self.assertIsNone(reference.model)


if __name__ == '__main__':
    unittest.main()
