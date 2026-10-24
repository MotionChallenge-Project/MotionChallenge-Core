"""Compatibility entrypoint for :mod:`motion_challenge.simulation.physics`."""
import sys

from .simulation import physics as _implementation

sys.modules[__name__] = _implementation
