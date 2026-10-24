"""Compatibility entrypoint for :mod:`motion_challenge.evaluation.merge`."""
import sys

from .evaluation import merge as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
