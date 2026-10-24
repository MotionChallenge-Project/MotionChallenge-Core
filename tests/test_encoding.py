import json
import math
import unittest
from dataclasses import asdict, replace

from motion_challenge.runtime.encoding import RequestEncoder
from motion_challenge.runtime.worker import packet
from motion_challenge.sdk import BodyState, ReferenceFrame, ProprioceptiveStatePacket


class EncodingTests(unittest.TestCase):
    def setUp(self):
        body = BodyState('body " unicode骨盆', (1., -0., 3.), (0., 0., 0., 1.),
                         (0., 0., 0.), (0., 0., 0.))
        self.state = ProprioceptiveStatePacket(.02, ('joint1', 'joint2'), (1., 2.), (3., 4.),
                                 (0., 0., 0., 1.), (0., 0., 0.),
                                 (0., 0., 0.))
        self.reference = ReferenceFrame(.04, .04, (1., 2.), (1., 2., 3.),
                                        (0., 0., 0., 1.), bodies=(body,))

    def test_existing_wire_and_decoded_packets_are_identical(self):
        encoder = RequestEncoder()
        for op in ('reset', 'act', 'act'):
            refs = (self.reference, replace(self.reference, requested_time_s=.06))
            expected = json.dumps({'op': op, 'state': asdict(self.state),
                'reference': [asdict(f) for f in refs], 'seed': 23}, allow_nan=False) + '\n'
            actual = encoder.request(op, self.state, refs, 23)
            self.assertEqual(actual, expected)
            data = json.loads(actual)
            self.assertEqual(packet(ProprioceptiveStatePacket, data['state']),
                             packet(ProprioceptiveStatePacket, json.loads(expected)['state']))

    def test_cache_bounded_and_reset_does_not_reuse_clip_data(self):
        encoder = RequestEncoder(capacity=2)
        for i in range(4):
            frame = replace(self.reference, joint_position_rad=(float(i), 2.))
            data = json.loads(encoder.request('act', self.state, (frame,)))
            self.assertEqual(data['reference'][0]['joint_position_rad'][0], i)
        self.assertEqual(len(encoder.frames), 2)
        encoder.request('reset', self.state, (self.reference,))
        self.assertEqual(len(encoder.frames), 1)

    def test_rejects_nonfinite_values_in_state_and_reference(self):
        encoder = RequestEncoder()
        with self.assertRaises(ValueError):
            encoder.request('act', replace(self.state, time_s=math.nan), (self.reference,))
        with self.assertRaises(ValueError):
            encoder.request('act', self.state, (replace(self.reference, source_time_s=math.inf),))


if __name__ == '__main__':
    unittest.main()
