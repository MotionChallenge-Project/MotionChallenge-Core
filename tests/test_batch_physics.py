import io
from dataclasses import astuple
import json
from pathlib import Path
import unittest

import numpy as np

from motion_challenge.simulation.vector_physics import VectorPhysics
from vector_helpers import VectorWorld
from motion_challenge.physics import World
from motion_challenge.data.reference import Motion
from scalar_oracle import evaluate_clip
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
            pool = VectorPhysics(PROFILE, 3, threads)
            serials = [World(PROFILE) for _ in range(3)]
            for episode,ids in enumerate(([2,0,1],[1,2],[0])):
                worlds = [pool.reference_world(motion(pool.world,height=.78+.01*i),64) for i in ids]
                pool.reset(ids,worlds)
                for i,world in zip(ids,worlds): serials[i].reset(world.motion)
                for t in (.007, .02, .041, .11, .2, 1/3):
                    actions = [command(t+episode+i) for i in ids]
                    inactive = [i for i in range(3) if i not in ids]
                    before = pool.qpos[inactive].copy()
                    pool.advance(ids,[astuple(a) for a in actions],[t]*len(ids))
                    np.testing.assert_array_equal(pool.qpos[inactive],before)
                    for i,action,state in zip(ids,actions,pool.states(ids)):
                        serial = serials[i]
                        serial.advance(action,t)
                        np.testing.assert_array_equal(pool.qpos[i],serial.data.qpos)
                        np.testing.assert_array_equal(pool.qvel[i],serial.data.qvel)
                        np.testing.assert_array_equal(state.bodies.values[:,:3],serial.data.xpos[list(pool.body_ids)])
                        expected = serial.state()
                        self.assertEqual(state.root_linear_velocity_world_m_s,expected.root_linear_velocity_world_m_s)
                        self.assertEqual(state.root_angular_velocity_world_rad_s,expected.root_angular_velocity_world_rad_s)
                        self.assertAlmostEqual(pool.time[i],t,places=12)

    def test_serial_batch_full_evaluation_trace_and_failure_match(self):
        protocol = json.loads((ROOT/'suites/lafan/protocol.mujoco317-ideal-proprioception-early-v2.json').read_text())
        runtime = {'state_access':'ideal-proprioceptive-no-position-v1', 'control_period_s': .013, 'reference_offsets_s': [0., .013]}
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
        self.assertEqual(run(VectorWorld(PROFILE)),expected)



if __name__ == '__main__':
    unittest.main()
