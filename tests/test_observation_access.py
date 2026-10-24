from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from motion_challenge.adapters.kinematics import KinematicStateView
from motion_challenge.resources import lafan_paths
from motion_challenge.runtime.encoding import RequestEncoder
from motion_challenge.runtime.observations import PROPRIOCEPTIVE, controller_state, validate_access
from motion_challenge.runtime.worker import packet
from motion_challenge.sdk import StatePacket, ProprioceptiveStatePacket
from motion_challenge.simulation.physics import World, forward_kinematics


class ObservationAccessTests(unittest.TestCase):
    def fixture(self):
        world=World(lafan_paths()['robot'])
        world.data.qpos[:3]=(3.,-7.,1.2)
        world.data.qvel[:6]=(.1,.2,.3,.4,.5,.6)
        forward_kinematics(world.model,world.data)
        return world.state()

    def test_translation_and_body_truth_never_reach_transport(self):
        state=self.fixture()
        changed=replace(state,root_position_world_m=(999.,-888.,777.),
                        bodies=tuple(replace(b,position_world_m=(123.,456.,789.)) for b in state.bodies))
        a=controller_state(state,PROPRIOCEPTIVE); b=controller_state(changed,PROPRIOCEPTIVE)
        self.assertEqual(a,b)
        for operation in ('reset','act'):
            payload=json.loads(RequestEncoder().request(operation,a,()))
            self.assertEqual(set(payload['state']),set(asdict(a)))
            self.assertNotIn('bodies',payload['state'])
            self.assertNotIn('root_position_world_m',payload['state'])
            decoded=packet(StatePacket,payload['state'])
            self.assertIsInstance(decoded,ProprioceptiveStatePacket)
            with self.assertRaises(TypeError):
                packet(StatePacket,{**payload['state'],'root_position_world_m':[0,0,0]})

    def test_unknown_schema_and_access_mismatch_fail_closed(self):
        with self.assertRaises(ValueError): packet(StatePacket,{'schema':'unknown'})
        with self.assertRaises(ValueError): validate_access({'state_access':PROPRIOCEPTIVE},{})
        with self.assertRaises(ValueError): validate_access({'state_access':'unknown'},{})

    def test_fk_uses_only_permitted_measurements(self):
        state=self.fixture(); view=KinematicStateView(lafan_paths()['robot'])
        rebuilt=view.reconstruct(controller_state(state,PROPRIOCEPTIVE))
        for a,b in zip(state.bodies,rebuilt.bodies):
            np.testing.assert_allclose(np.asarray(a.position_world_m)-state.root_position_world_m,b.position_world_m,atol=1e-14)
            np.testing.assert_allclose(a.orientation_xyzw,b.orientation_xyzw,atol=1e-14)
            np.testing.assert_allclose(a.linear_velocity_world_m_s,b.linear_velocity_world_m_s,atol=1e-14)
        with self.assertRaises(TypeError): view.reconstruct(state)

    def test_new_protocol_preserves_physics_and_metrics(self):
        for mode in ('early','none'):
            old=json.loads(lafan_paths(mode)['protocol'].read_text())
            new=json.loads(lafan_paths(mode,'ideal-proprioception')['protocol'].read_text())
            self.assertNotEqual(old['id'],new.pop('id'))
            old.pop('id'); self.assertEqual(new.pop('state_access'),PROPRIOCEPTIVE)
            self.assertIn('forbidden',new.pop('observation_rules'))
            self.assertEqual(old,new)

    def test_undeclared_checkpoint_cannot_opt_in(self):
        from motion_challenge.runtime.entrypoint import select_track
        with self.assertRaisesRegex(ValueError,'no declared inference path'):
            select_track({'weights_sha256':'fixed'},'ideal-proprioception',Path('.'))

if __name__=='__main__':unittest.main()
