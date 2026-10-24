"""A MuJoCo automatic reset must never turn a failed rollout into valid data."""
from contextlib import chdir
from pathlib import Path
import tempfile
import unittest
from dataclasses import astuple
import mujoco
import numpy as np
from motion_challenge.simulation.vector_physics import VectorPhysics
from motion_challenge.physics import World, command_values
from motion_challenge.sdk import ControlCommand

ROOT=Path(__file__).resolve().parents[1]
ZERO=ControlCommand((0.,)*29,(0.,)*29,(0.,)*29,(0.,)*29,(0.,)*29)

class NumericalResetTests(unittest.TestCase):
    def test_serial_automatic_reset_is_rejected(self):
        world=World(ROOT/'robots/g1-29dof-cpu-v0.json')
        world.data.time=.1;world.data.qvel[0]=1e20
        with tempfile.TemporaryDirectory() as folder, chdir(folder):
            with self.assertRaisesRegex(FloatingPointError,'reset or discontinuous'):
                world.advance(ZERO,.12)
        self.assertEqual(world.model.opt.timestep,.002)

    def test_batch_automatic_reset_is_rejected(self):
        pool=VectorPhysics(ROOT/'robots/g1-29dof-cpu-v0.json',1)
        pool.time[0]=.1;pool.qvel[0,0]=1e20
        with tempfile.TemporaryDirectory() as folder, chdir(folder):
            with self.assertRaisesRegex(FloatingPointError,'reset or discontinuous'):
                pool.advance([0],[astuple(ZERO)],[.12])
