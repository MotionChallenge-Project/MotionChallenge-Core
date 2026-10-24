"""Compatibility entrypoint for :mod:`motion_challenge.evaluation.run`."""
import sys

from .evaluation import run as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
