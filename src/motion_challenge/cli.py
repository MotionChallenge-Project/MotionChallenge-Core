"""Compatibility entrypoint for :mod:`motion_challenge.runtime.cli`."""
import sys

from .runtime import cli as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
