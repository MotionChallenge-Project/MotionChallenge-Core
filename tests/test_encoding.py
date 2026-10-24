"""Numeric transport equivalence, cache lifetime and array-view boundaries."""
import unittest
from dataclasses import replace
import numpy as np
from motion_challenge.runtime.policy import ArrayInputs
from motion_challenge.sdk import BodyState, ReferenceFrame, ProprioceptiveStatePacket
from motion_challenge.data.bodies import BodyArray, body_values

class EncodingTests(unittest.TestCase):
    def setUp(self):
        body=BodyState('body " unicode骨盆',(1.,-0.,3.),(0.,0.,0.,1.),(0.,0.,0.),(0.,0.,0.))
        self.state=ProprioceptiveStatePacket(.02,('joint1','joint2'),(1.,2.),(3.,4.),(0.,0.,0.,1.),(0.,0.,0.),(0.,0.,0.))
        self.reference=ReferenceFrame(.04,.04,(1.,2.),(1.,2.,3.),(0.,0.,0.,1.),(3.,4.),(1.,2.,3.),(4.,5.,6.),(body,))
        self.buffer=ArrayInputs(dict(capacity=2,slots=3,frames=2,joints=self.state.joint_names,bodies=[body.name]))
    def test_packets_and_array_views_are_identical(self):
        refs=(self.reference,replace(self.reference,requested_time_s=.06))
        for _ in range(3):
            self.buffer.put(0,self.state,refs)
            arrays,reference=self.buffer.get_arrays(1)
            self.assertFalse(arrays.joint_position_rad.flags.writeable)
            self.assertFalse(reference.body_position_world_m.flags.writeable)
            np.testing.assert_array_equal(reference.joint_velocity_rad_s[0,1],refs[1].joint_velocity_rad_s)
            self.assertFalse(hasattr(arrays,'root_position_world_m'))
    def test_cache_bounded_and_no_stale_motion_reuse(self):
        for i in range(40):
            ref=replace(self.reference,joint_position_rad=(float(i),2.))
            self.buffer.put(0,self.state,(ref,ref))
            self.assertEqual(self.buffer.get_arrays(1)[1].joint_position_rad[0,0,0],i)
        self.assertLessEqual(len(self.buffer.reference_cache),12)
    def test_array_path_does_not_materialize_body_packets(self):
        body = self.reference.bodies[0]
        values = np.array([(*body.position_world_m,*body.orientation_xyzw,
                           *body.linear_velocity_world_m_s,*body.angular_velocity_world_rad_s)])
        bodies = BodyArray((body.name,),values)
        ref = replace(self.reference,bodies=bodies)
        self.buffer.put(0,self.state,(ref,ref))
        np.testing.assert_array_equal(self.buffer.get_arrays(1)[1].body_position_world_m[0,0,0],body.position_world_m)
        np.testing.assert_array_equal(body_values(bodies,(body.name,)),values[:,:7])
        self.assertEqual(bodies._packets,{})
        self.assertEqual(bodies[0],body)
        self.assertIs(bodies[0],bodies[-1])
        self.assertFalse(bodies.values.flags.writeable)
        with self.assertRaises(IndexError): _ = bodies[-2]
    def test_unknown_contract_and_extra_truth_packet_are_rejected(self):
        with self.assertRaises(ValueError):
            self.buffer.put(0,replace(self.state,schema='unknown'),(self.reference,)*2)
        with self.assertRaises(ValueError):
            self.buffer.put(0,replace(self.state,joint_names=('bad','names')),(self.reference,)*2)
