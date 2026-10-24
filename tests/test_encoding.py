"""Array views and the optional packet adapter boundary preserve allowed data."""
import unittest
import numpy as np
from motion_challenge.runtime.policy import array_slice, packet_inputs
from motion_challenge.sdk import StateBatch, ReferenceBatch
from motion_challenge.simulation.array_reference import reference_view

class EncodingTests(unittest.TestCase):
    def test_array_views_selected_bodies_and_packet_boundary(self):
        state=StateBatch(('joint1','joint2'),np.array([.02,.04]),np.ones((2,2)),np.zeros((2,2)),
                         np.tile([0.,0.,0.,1.],(2,1)),np.ones((2,3)),np.zeros((2,3)))
        data=np.arange(2*2*(19+39),dtype=float).reshape(2,2,58)
        reference=reference_view(data,2,('leg','root','arm'))
        ss=array_slice(state,slice(None));rr=array_slice(reference,slice(None),(1,0))
        self.assertEqual(rr.body_names,('root','leg'))
        self.assertFalse(hasattr(ss,'root_position_world_m'))
        self.assertFalse(ss.joint_position_rad.flags.writeable)
        self.assertFalse(rr.body_position_world_m.flags.writeable)
        np.testing.assert_array_equal(rr.body_position_world_m,reference.body_position_world_m[:,:,[1,0]])
        packets,refs=packet_inputs(ss,rr)
        for row in range(2):
            np.testing.assert_array_equal(packets[row].joint_position_rad,state.joint_position_rad[row])
            np.testing.assert_array_equal(refs[row][1].bodies.values[:,:3],rr.body_position_world_m[row,1])
            self.assertFalse(hasattr(packets[row],'root_position_world_m'))
