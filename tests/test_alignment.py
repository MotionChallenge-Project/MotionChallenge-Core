import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from motion_challenge.alignment import keybody_position_errors


class AlignmentTests(unittest.TestCase):
    points = np.array([[0., 0., .8], [.4, -.2, 1.1], [-.3, .1, .1]])
    root = np.array([0., 0., .8])
    identity = (0., 0., 0., 1.)

    def errors(self, a, ar, aq, r=None, rr=None, rq=None):
        return keybody_position_errors(a, self.points if r is None else r,
            ar, self.root if rr is None else rr, aq, self.identity if rq is None else rq)

    def test_horizontal_shift_only_is_removed_by_both_alignments(self):
        shift = np.array([3., 4., 0.])
        result = self.errors(self.points+shift, self.root+shift, self.identity)
        self.assertAlmostEqual(result['mpjpe_world_m'], 5.)
        self.assertAlmostEqual(result['mpjpe_xy_aligned_m'], 0.)
        self.assertAlmostEqual(result['mpjpe_xy_yaw_aligned_m'], 0.)

    def test_yaw_is_retained_by_xy_only_and_removed_by_xy_yaw(self):
        yaw = Rotation.from_rotvec([0., 0., math.pi/2])
        result = self.errors(yaw.apply(self.points), self.root, yaw.as_quat())
        expected = math.sqrt(2)*(math.hypot(.4, -.2)+math.hypot(-.3, .1))/3
        self.assertAlmostEqual(result['mpjpe_xy_aligned_m'], expected)
        self.assertAlmostEqual(result['mpjpe_xy_yaw_aligned_m'], 0.)
        self.assertAlmostEqual(result['root_yaw_correction_abs_rad'], math.pi/2)

    def test_height_is_preserved_by_all_three(self):
        shift = np.array([0., 0., -.2])
        result = self.errors(self.points+shift, self.root+shift, self.identity)
        for name in ('mpjpe_world_m','mpjpe_xy_aligned_m','mpjpe_xy_yaw_aligned_m'):
            self.assertAlmostEqual(result[name], .2)

    def test_tilt_is_not_removed(self):
        tilt = Rotation.from_rotvec([.4, 0., 0.])
        actual = tilt.apply(self.points-self.root)+self.root
        result = self.errors(actual, self.root, tilt.as_quat())
        self.assertGreater(result['mpjpe_xy_yaw_aligned_m'], .05)
        self.assertAlmostEqual(result['mpjpe_xy_yaw_aligned_m'],result['mpjpe_xy_aligned_m'])

    def test_vertical_forward_axis_does_not_create_euler_yaw_artifact(self):
        bent = Rotation.from_rotvec([0., math.pi/2, 0.])
        yaw = Rotation.from_rotvec([0., 0., .7])
        reference = bent.apply(self.points-self.root)+self.root
        actual = yaw.apply(reference)+[2., -1., 0.]
        result = self.errors(actual,self.root+[2.,-1.,0.],(yaw*bent).as_quat(),
                             r=reference,rq=bent.as_quat())
        self.assertAlmostEqual(result['mpjpe_xy_yaw_aligned_m'],0.)
        self.assertEqual(result['yaw_alignment_degenerate_fraction'],0.)

    def test_ambiguous_opposite_tilt_is_flagged_and_not_discarded(self):
        tilt = Rotation.from_rotvec([math.pi, 0., 0.])
        result = self.errors(tilt.apply(self.points-self.root)+self.root,self.root,tilt.as_quat())
        self.assertEqual(result['yaw_alignment_degenerate_fraction'],1.)
        self.assertGreater(result['mpjpe_xy_yaw_aligned_m'],.1)

    def test_inputs_unchanged_and_invalid_values_rejected(self):
        actual, reference = self.points.copy(),self.points.copy()
        self.errors(actual,self.root,self.identity,r=reference)
        np.testing.assert_array_equal(actual,self.points)
        np.testing.assert_array_equal(reference,self.points)
        actual[1,0] = float('nan')
        with self.assertRaises(ValueError):self.errors(actual,self.root,self.identity)


if __name__ == '__main__':unittest.main()
