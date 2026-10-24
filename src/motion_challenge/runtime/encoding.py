"""Encode the existing JSON protocol without copying immutable packet trees."""
from collections import OrderedDict
from dataclasses import fields, is_dataclass
import json


def _packet_fields(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


def encode(value):
    return json.dumps(value, default=_packet_fields, allow_nan=False)


class RequestEncoder:
    """Keep a bounded cache of reference frames reused by overlapping windows.

    World owns immutable reference packets. Retaining each object alongside its
    encoded form prevents identity reuse; reset clears the preceding clip.
    """
    def __init__(self, capacity=128):
        if capacity < 1:
            raise ValueError('reference cache capacity must be positive')
        self.capacity = capacity
        self.frames = OrderedDict()

    def frame(self, frame):
        key = id(frame)
        if key not in self.frames:
            self.frames[key] = (frame, encode(frame))
            if len(self.frames) > self.capacity:
                self.frames.popitem(last=False)
        else:
            self.frames.move_to_end(key)
        return self.frames[key][1]

    def request(self, op, state, reference, seed=0):
        if op == 'reset':
            self.frames.clear()
        frames = ', '.join(self.frame(frame) for frame in reference)
        return ('{"op": ' + encode(op) + ', "state": ' + encode(state)
                + ', "reference": [' + frames + '], "seed": ' + encode(seed) + '}\n')
