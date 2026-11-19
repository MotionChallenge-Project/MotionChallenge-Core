from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from motion_challenge.runtime.observations import DEPLOYABLE, PROPRIOCEPTIVE, controller_state, controller_arrays, validate_access
from motion_challenge.sdk import StatePacket, StateBatch, ReferenceBatch, DeployableStateBatch
from motion_challenge.runtime.policy import PolicyBatch


class StrictObservationTests(unittest.TestCase):
    def state(self):
        return StatePacket(0.0, ("joint",), (0.1,), (0.2,), (1.,2.,3.), (0.,0.,0.,1.), (4.,5.,6.), (0.1,0.2,0.3))

    def test_forbidden_truth_cannot_change_packet(self):
        state=self.state()
        changed=replace(state, root_position_world_m=(100.,200.,300.), root_linear_velocity_world_m_s=(400.,500.,600.))
        a,b=controller_state(state,DEPLOYABLE),controller_state(changed,DEPLOYABLE)
        self.assertEqual(a,b)
        self.assertFalse(hasattr(a,"root_linear_velocity_world_m_s"))
        self.assertFalse(hasattr(a,"root_position_world_m"))
        self.assertEqual(controller_state(state,PROPRIOCEPTIVE).root_linear_velocity_world_m_s,(4.,5.,6.))
        with self.assertRaises(ValueError):
            validate_access({"state_access":DEPLOYABLE},{"state_access":PROPRIOCEPTIVE})

    def arrays(self):
        s=self.state()
        state=StateBatch(s.joint_names,np.array([0.]),*[np.array([getattr(s,k)]) for k in
            ("joint_position_rad","joint_velocity_rad_s","root_orientation_xyzw","root_linear_velocity_world_m_s","root_angular_velocity_world_rad_s")])
        refs=ReferenceBatch(np.zeros((1,1)),np.zeros((1,1)),np.zeros((1,1,1)),
            np.ones((1,1,3)),np.array([[[0.,0.,0.,1.]]]),np.zeros((1,1,1)),
            np.ones((1,1,3)),np.zeros((1,1,3)),(),
            np.zeros((1,1,0,3)),np.zeros((1,1,0,4)),np.zeros((1,1,0,3)),np.zeros((1,1,0,3)))
        return state,refs

    def test_policy_boundary_strips_before_array_and_packet_reset(self):
        from unittest.mock import patch
        for array_mode in (True,False):
            with self.subTest(array_mode=array_mode),tempfile.TemporaryDirectory() as d:
                state,refs=self.arrays()
                seen=[]
                class Adapter:
                    def reset_batch(self,slots,ss,rr,seeds):
                        seen.append(ss if array_mode else ss[0])
                    def close(self): pass
                p=Path(d)/"runtime.json"
                p.write_text(json.dumps({"state_access":DEPLOYABLE,("array_batch_entrypoint" if array_mode else "batch_entrypoint"):"unused:create"}))
                with patch("motion_challenge.runtime.policy.load_controller",return_value=Adapter()):
                    policy=PolicyBatch(p,1,state,refs,Path(d)/"log")
                    try: policy.call_arrays("reset",[0],state,refs,[0])
                    finally: policy.close()
                self.assertEqual(len(seen),1)
                self.assertFalse(hasattr(seen[0],"root_linear_velocity_world_m_s"))
                self.assertFalse(hasattr(seen[0],"root_position_world_m"))

    def test_reference_velocity_is_retained(self):
        state,refs=self.arrays()
        clean=controller_arrays(state,DEPLOYABLE)
        self.assertIsInstance(clean,DeployableStateBatch)
        self.assertFalse(hasattr(clean,"root_linear_velocity_world_m_s"))
        self.assertTrue(np.all(refs.root_linear_velocity_world_m_s==1))


if __name__ == "__main__":
    unittest.main()
