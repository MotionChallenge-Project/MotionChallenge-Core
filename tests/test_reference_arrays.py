"""Grouped interpolation preserves scalar boundaries and quaternion arithmetic."""
import unittest
import numpy as np
from motion_challenge.data.reference import Motion


class ReferenceArrayTests(unittest.TestCase):
    def test_scalar_equivalence_nonuniform_times_signs_and_bounded_cache(self):
        rng = np.random.default_rng(314)
        times = np.r_[0.,np.cumsum(rng.uniform(.01,.04,500))]
        q = rng.normal(size=(len(times),4));q /= np.linalg.norm(q,axis=1)[:,None]
        motion = Motion(('a','b'),times,rng.normal(size=(len(times),2)),
                        rng.normal(size=(len(times),3)),q)
        requested = np.r_[[-1.,0.,times[-1],times[-1]+1], times,
                          rng.uniform(0,times[-1],1000)]
        rng.shuffle(requested)
        t,j,p,r = motion.sample_arrays(requested)
        for i,when in enumerate(requested):
            scalar = motion.sample(when)
            self.assertEqual(t[i],scalar.source_time_s)
            np.testing.assert_array_equal(j[i],scalar.joint_position_rad)
            np.testing.assert_array_equal(p[i],scalar.root_position_world_m)
            np.testing.assert_array_equal(r[i],scalar.root_orientation_xyzw)
        self.assertLessEqual(len(motion._rotation_segments),128)
