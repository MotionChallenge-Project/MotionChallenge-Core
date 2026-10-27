"""Paper root-only failures, including boundary cases and independent scoring."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from motion_challenge.evaluation.batch_metrics import tracking_batch
from motion_challenge.resources import lafan_paths
from motion_challenge.sdk import BodyState, StatePacket, ReferenceFrame
from motion_challenge.simulation.physics import tracking_sample


class RootFailureTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(lafan_paths('none')['protocol'].read_text())
        self.bodies = tuple(BodyState(n, (0., 0., 0.), (0., 0., 0., 1.),
                                     (0.,)*3, (0.,)*3)
                            for n in self.protocol['metric_bodies'])
        self.state = StatePacket(0., ('j',), (0.,), (0.,), (0.,)*3,
                                 (0., 0., 0., 1.), (0.,)*3, (0.,)*3, self.bodies)
        self.ref = ReferenceFrame(0., 0., (0.,), (0.,)*3,
                                  (0., 0., 0., 1.), bodies=self.bodies)

    def check(self, state, expected):
        failures, metrics = tracking_sample(state, self.ref, self.protocol)
        batch_failures, batch_metrics = tracking_batch([state], [self.ref], self.protocol)[0]
        self.assertEqual(failures, expected)
        self.assertEqual(batch_failures, expected)
        self.assertEqual(metrics.keys(), batch_metrics.keys())
        np.testing.assert_allclose(list(metrics.values()), list(batch_metrics.values()), atol=2e-15, rtol=0)
        self.assertNotIn('end_effector_height_error_m', metrics)
        return metrics

    def test_default_protocol_is_root_only_v2(self):
        self.assertTrue(self.protocol['id'].endswith('v2'))
        self.assertEqual(self.protocol['failure']['anchor_body'], 'pelvis')
        self.assertNotIn('end_effector_bodies', self.protocol['failure'])
        early = json.loads(lafan_paths('early')['protocol'].read_text())
        self.assertEqual(early['failure'], self.protocol['failure'])

    def test_root_height_boundary(self):
        for height in (.249999, .25, .250001, -.25, -.250001):
            state = replace(self.state, bodies=tuple(
                replace(b, position_world_m=(0., 0., height)) if b.name=='pelvis' else b
                for b in self.bodies))
            self.check(state, ['anchor_height'] if abs(height)>.25 else [])

    def test_torso_and_end_effector_errors_do_not_fail(self):
        bodies = tuple(replace(b, position_world_m=(0., 0., 2.),
                               orientation_xyzw=(1., 0., 0., 0.))
                       if b.name!='pelvis' else b for b in self.bodies)
        metrics = self.check(replace(self.state, bodies=bodies), [])
        self.assertGreater(metrics['mpjpe_xy_aligned_m'], 1.)

    def test_root_tilt_and_yaw(self):
        for tilt in (.8-1e-7, .8+1e-7):
            q = tuple(Rotation.from_euler('x', np.arccos(1-tilt)).as_quat())
            state = replace(self.state, bodies=tuple(
                replace(b, orientation_xyzw=q) if b.name=='pelvis' else b for b in self.bodies))
            self.check(state, ['anchor_gravity_z'] if tilt>.8 else [])
        yaw = replace(self.state, bodies=tuple(
            replace(b, orientation_xyzw=(0., 0., 1., 0.)) if b.name=='pelvis' else b
            for b in self.bodies))
        self.check(yaw, [])


if __name__ == '__main__':
    unittest.main()
