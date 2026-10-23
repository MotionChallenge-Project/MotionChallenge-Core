"""Locate frozen suite/robot assets in an editable checkout or installed wheel."""
from pathlib import Path
import sysconfig


def resource_root():
    checkout=Path(__file__).resolve().parents[2]
    installed=Path(sysconfig.get_path('data'))/'share/motion-challenge'
    for root in (checkout,installed):
        if (root/'suites/lafan/dataset.json').is_file() and (root/'robots/g1-29dof-cpu-v0.json').is_file():
            return root
    raise FileNotFoundError('Motion Challenge package is missing its frozen suite/robot assets')


def lafan_paths():
    root=resource_root()
    return {'suite':root/'suites/lafan/dataset.json','robot':root/'robots/g1-29dof-cpu-v0.json',
            'protocol':root/'suites/lafan/protocol.cpu-development-v1.json'}
