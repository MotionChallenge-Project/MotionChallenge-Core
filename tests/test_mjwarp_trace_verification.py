"""Full-coverage, clock and integrity gates in the current CPU trace verifier."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from motion_challenge.evaluation.tensor_verify import KEYS, verify, initialize, chunk
from motion_challenge.resources import resource_root
from motion_challenge.simulation.physics import World
from motion_challenge.validation import sha256_file


class TraceVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prepared = self.root / 'prepared'
        self.prepared.mkdir()
        protocol_path = resource_root() / 'suites/lafan/protocol.mjwarp311-no-linear-velocity-v1.json'
        protocol = json.loads(protocol_path.read_text())
        world = World(resource_root() / protocol['robot_profile'])
        qpos = world.model.qpos0.copy()
        qpos[:3] = [0., 0., 1.]
        qpos[3:7] = [1., 0., 0., 0.]
        qpos[world.qadr] = 0
        pose = np.r_[qpos[:3], qpos[[4, 5, 6, 3]], qpos[world.qadr]].astype(np.float32)
        np.save(self.prepared / 'poses.npy', np.repeat(pose[None], 6, axis=0))
        self.meta = {'dataset': 'lafan', 'fps': 50., 'names': ['a', 'b'], 'starts': [0, 3], 'lengths': [3, 3],
                     'poses_sha256': sha256_file(self.prepared / 'poses.npy')}
        (self.prepared / 'dataset.json').write_text(json.dumps(self.meta))
        self.trace = {'clip_id': np.array([0, 0, 1, 1]), 'time_s': np.array([.02, .04, .02, .04]),
                      'qpos': np.repeat(qpos[None], 4, axis=0)}
        metrics = {key: 0. for key in KEYS}
        records = [dict(clip_id=index, name=name, duration_s=.04, samples=2, success=True, completion=1.,
                        full=metrics.copy(), prefix=metrics.copy()) for index, name in enumerate(self.meta['names'])]
        self.result = {'dataset': 'lafan', 'protocol_sha256': sha256_file(protocol_path),
                       'status': 'unverified_full_rollout', 'traces_recorded': True,
                       'prepared': str(self.prepared),
                       'dataset_identity': {key: value for key, value in self.meta.items() if key not in ('names', 'starts', 'lengths')},
                       'clips': records, 'summary': {'success_count': 2, 'completion': 1.,
                                                    'full': metrics.copy(), 'prefix': metrics.copy()}}
        self.write()

    def write(self):
        np.savez(self.root / 'trace-00000.npz', **self.trace)
        self.result['traces'] = [{'path': 'trace-00000.npz', 'sha256': sha256_file(self.root / 'trace-00000.npz'),
                                  'samples': len(self.trace['clip_id'])}]
        (self.root / 'result.json').write_text(json.dumps(self.result))

    def check(self):
        return verify(self.root / 'result.json', jobs=1)

    def test_full_trace_is_independently_recomputed(self):
        report = self.check()
        self.assertEqual(report['clips'], 2)
        self.assertEqual(report['verification'], 'independent_cpu_trace_recomputation_passed')
        self.assertLess(report['max_metric_abs_error'], 1e-12)

    def test_float32_reference_preserves_exact_strict_height_boundary(self):
        # Source samples from the independently diagnosed SEED boundary case.
        # At 3.64 s the unquantized height is .25000002384185793 m;
        # the stored float32 reference produces exactly .25 m.
        time = 3.64
        protocol = json.loads((resource_root() / 'suites/seed/protocol.mjwarp311-no-linear-velocity-v1.json').read_text())
        poses = np.repeat(np.load(self.prepared / 'poses.npy')[:1], 441, axis=0)
        poses[436, 2] = np.float32(0.7744037508964539)
        poses[437, 2] = np.float32(0.7744355201721191)
        raw_z = float(poses[436, 2]) + (time * 120 - 436) * float(poses[437, 2] - poses[436, 2])
        reference_z = np.float32(raw_z)
        actual_z = np.float32(0.5244291424751282)
        self.assertGreater(raw_z - float(actual_z), .25)
        self.assertEqual(float(reference_z) - float(actual_z), .25)
        np.save(self.prepared / 'poses.npy', poses)
        meta = {'fps': 120., 'names': ['boundary'], 'starts': [0], 'lengths': [441]}
        (self.prepared / 'dataset.json').write_text(json.dumps(meta))
        qpos = np.repeat(self.trace['qpos'][:1], 3, axis=0).astype(np.float32)
        qpos[:, 2] = [np.nextafter(actual_z, np.float32(np.inf)), actual_z,
                      np.nextafter(actual_z, np.float32(-np.inf))]
        path = self.root / 'boundary-trace.npz'
        np.savez(path, clip_id=np.zeros(3, dtype=np.int64), time_s=np.full(3, time), qpos=qpos)
        initialize(self.prepared, protocol)
        _, _, metrics, failed = chunk((path, sha256_file(path)))
        np.testing.assert_array_equal(failed, [False, False, True])
        self.assertEqual(metrics[1, KEYS.index('anchor_height_error_m')], .25)

    def test_duplicate_motion_record_is_rejected(self):
        self.result['clips'].append(copy.deepcopy(self.result['clips'][0]))
        self.write()
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            self.check()

    def test_missing_and_shifted_clock_are_rejected(self):
        for changed in ([.02, .03, .02, .04], [.02, .02, .02, .04]):
            with self.subTest(clock=changed):
                self.trace['time_s'] = np.asarray(changed)
                self.write()
                with self.assertRaisesRegex(ValueError, 'timestamp'):
                    self.check()

    def test_changed_reference_poses_are_rejected(self):
        poses = np.load(self.prepared / 'poses.npy')
        poses[0, 0] += .01
        np.save(self.prepared / 'poses.npy', poses)
        with self.assertRaisesRegex(ValueError, 'Reference poses changed'):
            self.check()

    def test_trace_hash_and_nonfinite_state_are_rejected(self):
        with (self.root / 'trace-00000.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Trace hash mismatch'):
            self.check()
        self.trace['qpos'][0, 0] = np.nan
        self.write()
        with self.assertRaisesRegex(ValueError, 'Invalid trace poses or time'):
            self.check()

    def test_summary_tampering_and_missing_trace_mode_are_rejected(self):
        self.result['summary']['success_count'] = 1
        self.write()
        with self.assertRaisesRegex(ValueError, 'Summary differs'):
            self.check()
        self.result['traces_recorded'] = False
        self.write()
        with self.assertRaisesRegex(ValueError, 'full rollout with traces'):
            self.check()


if __name__ == '__main__':
    unittest.main()
