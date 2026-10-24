"""Compatibility entrypoint for :mod:`motion_challenge.evaluation.verify`."""
import sys

from .evaluation import verify as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
