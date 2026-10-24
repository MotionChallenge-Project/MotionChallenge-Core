import io
import json
from pathlib import Path
import unittest

import numpy as np

from motion_challenge.simulation.batch_physics import BatchPool
from motion_challenge.physics import World
from motion_challenge.data.reference import Motion
from motion_challenge.run import evaluate_clip
from motion_challenge.sdk import ControlCommand

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT/'robots/g1-29dof-cpu-v0.json'


def motion(world, duration=1/3, height=.78):
    q = tuple(world.model.qpos0[world.qadr])
    return Motion(world.joint_names, (0., duration), (q, q),
                  ((0., 0., height), (0., 0., height)), ((0., 0., 0., 1.),)*2)


def command(t):
    return ControlCommand(tuple(.12*np.sin(np.arange(29)+t*3)), (0.,)*29,
                          (35.,)*29, (1.1,)*29, (0.,)*29)


class BatchPhysicsTests(unittest.TestCase):
    def test_substeps_fractional_endpoint_reset_and_inactive_slots(self):
        for threads in (1, 3):
            pool = BatchPool(PROFILE, 3, threads)
            def trajectory(slot, world):
                serial = World(PROFILE)
                for episode in range(slot+1):
                    clip = motion(world, height=.78+.01*slot)
                    world.reset(clip)
                    serial.reset(clip)
                    for t in (.007, .02, .041, .11, .2, 1/3):
                        action = command(t+episode)
                        world.advance(action, t)
                        serial.advance(action, t)
                        np.testing.assert_array_equal(world.data.qpos, serial.data.qpos)
                        np.testing.assert_array_equal(world.data.qvel, serial.data.qvel)
                        np.testing.assert_array_equal(world.data.xpos, serial.data.xpos)
                        self.assertAlmostEqual(world.data.time, t, places=12)
                return slot
            self.assertEqual(pool.run(trajectory), [0, 1, 2])

    def test_serial_batch_full_evaluation_trace_and_failure_match(self):
        protocol = json.loads((ROOT/'suites/lafan/protocol.mujoco317-v1.json').read_text())
        runtime = {'control_period_s': .013, 'reference_offsets_s': [0., .013]}
        class FakeWorker:
            def call(self, op, state, reference, seed=0):
                if op == 'reset':
                    self.count = 0
                    return None
                self.count += 1
                return command(self.count*.013)
        def run(world):
            trace = io.StringIO()
            result = evaluate_clip(world, motion(world, 1.), FakeWorker(), runtime, protocol, trace=trace)
            result.pop('wall_time_s')
            return result, trace.getvalue()
        expected = run(World(PROFILE))
        self.assertEqual(BatchPool(PROFILE, 2, 2).run(lambda slot, world: run(world)), [expected]*2)

    def test_worker_exception_wakes_waiting_slots(self):
        pool = BatchPool(PROFILE, 2, 1)
        def fail(slot, world):
            if slot == 1:
                raise ValueError('adapter failed')
            world.reset(motion(world))
        with self.assertRaisesRegex(ValueError, 'adapter failed'):
            pool.run(fail)


if __name__ == '__main__':
    unittest.main()
