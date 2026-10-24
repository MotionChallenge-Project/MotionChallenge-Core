"""Compatibility entrypoint for :mod:`motion_challenge.runtime.entrypoint`."""
import sys

from .runtime import entrypoint as _implementation

sys.modules[__name__] = _implementation
