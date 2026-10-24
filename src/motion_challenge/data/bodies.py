"""Numeric body snapshots; scalar adapters materialize named values on demand."""
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from motion_challenge.sdk import BodyState


@lru_cache(maxsize=64)
def _indices(names, selected):
    return np.array([names.index(name) for name in selected])


class BodyArray(Sequence):
    """Own a read-only [body, position/xyzw/linear/angular] snapshot."""
    def __init__(self, names, values):
        self.names = tuple(names)
        self.values = values
        self.values.flags.writeable = False
        self._packets = {}

    def __len__(self):
        return len(self.names)

    def __eq__(self, other):
        if isinstance(other, BodyArray):
            return self.names == other.names and np.array_equal(self.values, other.values)
        return tuple(self) == other

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self._packets:
            row = self.values[index]
            self._packets[index] = BodyState(self.names[index], tuple(row[:3]),
                tuple(row[3:7]), tuple(row[7:10]), tuple(row[10:13]))
        return self._packets[index]


def body_values(bodies, names):
    """Select numeric rows without building dictionaries or scalar body objects."""
    if isinstance(bodies, BodyArray):
        return bodies.values[_indices(bodies.names, tuple(names)), :7]
    by_name = {body.name: body for body in bodies}
    return np.array([(*by_name[name].position_world_m, *by_name[name].orientation_xyzw)
                     for name in names])
