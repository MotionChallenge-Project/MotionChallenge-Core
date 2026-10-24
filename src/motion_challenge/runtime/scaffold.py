"""Create a separate SDK consumer project, with no copied evaluation implementation."""
import json
import os
import subprocess
from pathlib import Path


FIXTURE='''from motion_challenge.sdk import ControlCommand


class ReferencePD:
    """Interface fixture only. Replace with a trained model before ranking."""
    def reset(self, state, reference, seed):
        pass

    def act(self, state, reference):
        return ControlCommand(tuple(reference[0].joint_position_rad), (0.,)*29,
                              (40.,)*29, (2.,)*29, (0.,)*29)

    def close(self):
        pass


def create(root, config):
    return ReferencePD()
'''

SMOKE='''import argparse
from pathlib import Path
import subprocess
import sys
from motion_challenge.resources import lafan_paths

p=argparse.ArgumentParser()
p.add_argument('--data-cache',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--fetch',action='store_true')
a=p.parse_args();paths=lafan_paths()
common=['--controller',str(Path(__file__).parent/'runtime.json'),'--data',str(a.data_cache)]
for key,path in paths.items():common.extend(['--'+key,str(path)])
if a.fetch:
    subprocess.run([sys.executable,'-m','motion_challenge.cli','fetch-dataset','--manifest',str(paths['suite']),'--root',str(a.data_cache)],check=True)
subprocess.run([sys.executable,'-m','motion_challenge.run',*common,'--max-clips','1','--max-duration','2','--output',str(a.output)],check=True)
subprocess.run([sys.executable,'-m','motion_challenge.verify',*common,'--result',str(a.output)],check=True)
'''


def create_project(destination,core_path):
    destination=Path(destination).resolve();core_path=Path(core_path).resolve()
    if not (core_path/'pyproject.toml').is_file():raise ValueError('core-path must be the Motion Challenge source checkout')
    if destination==core_path or core_path in destination.parents:raise ValueError('create the consumer outside the core repository')
    if destination.exists() and any(destination.iterdir()):raise ValueError('destination must be absent or empty')
    destination.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','init','-b','main',str(destination)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(destination),'-c','protocol.file.allow=always','submodule','add',str(core_path),'MotionChallenge-Core'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(destination),'config','-f','.gitmodules','submodule.MotionChallenge-Core.url','https://github.com/MotionChallenge-Project/MotionChallenge-Core.git'],check=True)
    dependency='MotionChallenge-Core'
    (destination/'pyproject.toml').write_text('[project]\nname = "motion-challenge-consumer"\nversion = "0.0.1"\nrequires-python = ">=3.12,<3.13"\ndependencies = ["motion-challenge"]\n\n[tool.uv.sources]\nmotion-challenge = {path = '+json.dumps(dependency)+', editable = true}\n')
    (destination/'.python-version').write_text('3.12\n')
    (destination/'.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\n')
    (destination/'adapter.py').write_text(FIXTURE)
    (destination/'smoke.py').write_text(SMOKE)
    (destination/'evaluate.py').write_text('from pathlib import Path\nfrom motion_challenge.entrypoint import main\n\nif __name__ == "__main__":\n    main(Path(__file__).parent)\n')
    runtime={'id':'reference-pd-interface-fixture','entrypoint':'adapter:create','control_period_s':.02,
             'reference_offsets_s':[0.],'reference_history_s':0.,'reference_preview_s':0.,'state_access':'privileged-state-v0',
             'action_mode':'joint-pd-feedforward','official':False,'is_trained_checkpoint':False}
    (destination/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    return {'project':str(destination),'dependency':dependency,'next':'uv sync; uv run python smoke.py --data-cache EXTERNAL_CACHE --output EXTERNAL_RESULT --fetch'}
