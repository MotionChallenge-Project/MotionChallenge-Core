"""Locate frozen suite/robot assets in an editable checkout or installed wheel."""
from pathlib import Path
import sysconfig


def resource_root():
    checkout=Path(__file__).resolve().parents[2]
    installed=Path(sysconfig.get_path('data'))/'share/MotionChallenge-Core'
    for root in (checkout,installed):
        if (root/'suites/lafan/dataset.json').is_file() and (root/'robots/g1-29dof-cpu-v0.json').is_file():
            return root
    raise FileNotFoundError('MotionChallenge package is missing its frozen suite/robot assets')


def lafan_paths(termination='early', track='privileged'):
    if termination not in ('early', 'none'):
        raise ValueError('termination must be early or none')
    if track not in ('privileged', 'ideal-proprioception'):
        raise ValueError('unknown observation track')
    root=resource_root()
    protocol = 'protocol.mujoco317-v1.json' if termination == 'early' else 'protocol.mujoco317-no-termination-v1.json'
    if track == 'ideal-proprioception':
        protocol = f'protocol.mujoco317-ideal-proprioception-{termination}-v1.json'
    return {'suite':root/'suites/lafan/dataset.json','robot':root/'robots/g1-29dof-cpu-v0.json',
            'protocol':root/'suites/lafan'/protocol}
