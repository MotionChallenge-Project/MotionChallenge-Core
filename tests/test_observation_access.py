from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from motion_challenge.adapters.kinematics import KinematicStateView
from motion_challenge.resources import lafan_paths
from motion_challenge.runtime.policy import array_slice
from motion_challenge.runtime.observations import PROPRIOCEPTIVE, controller_state, validate_access
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
        from motion_challenge.sdk import ReferenceFrame
        ref=ReferenceFrame(0.,0.,a.joint_position_rad,(0.,0.,0.),a.root_orientation_xyzw,
                           a.joint_velocity_rad_s,a.root_linear_velocity_world_m_s,a.root_angular_velocity_world_rad_s)
        from motion_challenge.sdk import StateBatch
        inputs=StateBatch(a.joint_names,np.array([a.time_s]),*[np.array([getattr(a,k)]) for k in
            ('joint_position_rad','joint_velocity_rad_s','root_orientation_xyzw',
             'root_linear_velocity_world_m_s','root_angular_velocity_world_rad_s')])
        inputs=array_slice(inputs,slice(None))
        np.testing.assert_array_equal(inputs.joint_position_rad[0],a.joint_position_rad)
        self.assertFalse(hasattr(inputs,'root_position_world_m'))
        with self.assertRaises(ValueError):array_slice(state,slice(None))

    def test_unknown_schema_and_access_mismatch_fail_closed(self):
        with self.assertRaises(ValueError): validate_access({'state_access':PROPRIOCEPTIVE},{})
        with self.assertRaises(ValueError): validate_access({'state_access':'unknown'},{})
        for obsolete in ({}, {'state_access':'privileged-state-v0'}):
            with self.assertRaises(ValueError): validate_access(obsolete, obsolete)
            with self.assertRaises(ValueError): controller_state(self.fixture(),obsolete.get('state_access'))

    def test_fk_uses_only_permitted_measurements(self):
        state=self.fixture(); view=KinematicStateView(lafan_paths()['robot'])
        rebuilt=view.reconstruct(controller_state(state,PROPRIOCEPTIVE))
        for a,b in zip(state.bodies,rebuilt.bodies):
            np.testing.assert_allclose(np.asarray(a.position_world_m)-state.root_position_world_m,b.position_world_m,atol=1e-14)
            np.testing.assert_allclose(a.orientation_xyzw,b.orientation_xyzw,atol=1e-14)
            np.testing.assert_allclose(a.linear_velocity_world_m_s,b.linear_velocity_world_m_s,atol=1e-14)
        with self.assertRaises(TypeError): view.reconstruct(state)

    def test_only_proprioceptive_protocols_are_available(self):
        for mode in ('early','none'):
            protocol=json.loads(lafan_paths(mode)['protocol'].read_text())
            self.assertEqual(protocol['state_access'],PROPRIOCEPTIVE)
            self.assertIn('forbidden',protocol['observation_rules'])
        with self.assertRaises(TypeError): lafan_paths('early','privileged')

    def test_undeclared_checkpoint_cannot_opt_in(self):
        from motion_challenge.runtime.entrypoint import resolve_observation_contract
        with self.assertRaisesRegex(ValueError,'no declared observation admission'):
            resolve_observation_contract({'weights_sha256':'fixed','state_access':PROPRIOCEPTIVE},Path('.'))

    def test_admission_binds_exact_weight_and_verified_status(self):
        from motion_challenge.runtime.entrypoint import resolve_observation_contract
        from motion_challenge.validation import sha256_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);record=root/'admission.json'
            runtime={'weights_sha256':'fixed','state_access':PROPRIOCEPTIVE,'entrypoint':'adapter:create'}
            for status,weight,accepted in [('inference_verified','fixed',True),('pending','fixed',False),('inference_verified','other',False)]:
                record.write_text(json.dumps({'checkpoint_sha256':weight,'status':status,'state_access':PROPRIOCEPTIVE}))
                runtime['observation_contract']={'admission_record':'admission.json','admission_sha256':sha256_file(record)}
                if accepted:
                    self.assertEqual(resolve_observation_contract(runtime,root)['entrypoint'],'adapter:create')
                else:
                    with self.assertRaises(ValueError): resolve_observation_contract(runtime,root)

if __name__=='__main__':unittest.main()
