"""Online-only output preserves both metric scopes without claiming trace evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from motion_challenge.runtime.entrypoint import run_pair
from motion_challenge.runtime.scaffold import FIXTURE
from motion_challenge.evaluation.verify import verify
from motion_challenge.validation import sha256_file

ROOT = Path(__file__).resolve().parents[1]


class ResultOnlyTests(unittest.TestCase):
    def test_full_command_matches_verified_metrics_without_trace_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'adapter.py').write_text(FIXTURE)
            manifest = json.loads((ROOT/'suites/lafan/dataset.json').read_text())
            manifest['files'] = []
            for i in range(2):
                p = root/f'{i}.csv'
                p.write_text((','.join(map(str,[0,0,.8,0,0,0,1]+[0]*29))+'\n')*(5+i))
                manifest['files'].append(dict(path=p.name,sha256=sha256_file(p)))
            suite = root/'dataset.json';suite.write_text(json.dumps(manifest))
            for mode in ('early','none'):
                spec = json.loads((ROOT/f'suites/lafan/protocol.mujoco317-ideal-proprioception-{mode}-v1.json').read_text())
                spec['failure']['anchor_height_m'] = 1e-6
                (root/f'{mode}.json').write_text(json.dumps(spec))
            runtime = dict(id='test',state_access='ideal-proprioceptive-no-position-v1',adapter_root=str(root),
                entrypoint='adapter:create',array_batch_entrypoint='adapter:create',control_period_s=.02,
                reference_offsets_s=[0.],reference_history_s=0.,reference_preview_s=0.)
            (root/'runtime.json').write_text(json.dumps(runtime))
            paths = lambda mode:dict(suite=suite,robot=ROOT/'robots/g1-29dof-cpu-v0.json',protocol=root/f'{mode}.json')
            records=[]
            for no_traces in (False,True):
                output=root/str(no_traces);output.mkdir()
                args=SimpleNamespace(suite='fixture',data=root,jobs=2,threads=1,max_clips=None,
                    max_duration=None,progress_interval=100.,backend='batch',no_traces=no_traces)
                with patch('motion_challenge.runtime.entrypoint.lafan_paths',side_effect=paths),\
                     patch('motion_challenge.runtime.entrypoint.source_snapshot',return_value={}),\
                     contextlib.redirect_stdout(io.StringIO()):
                    run_pair(root,args,runtime,output,{},'test')
                result=json.loads((output/'result.json').read_text());records.append(result)
                expected='completed_without_traces' if no_traces else 'trace_verified'
                self.assertEqual(result['status'],expected)
                self.assertEqual(bool(list(output.rglob('*.poses.jsonl.gz'))),not no_traces)
            for key in ('success_count','total','macro_completion','metrics','clips'):
                self.assertEqual(records[0][key],records[1][key],key)
            self.assertNotEqual(records[1]['clips'][0]['sample_count']['until_first_failure'],
                                records[1]['clips'][0]['sample_count']['full_motion'])
            with self.assertRaisesRegex(ValueError,'no recorded trajectory'):
                verify(root/'True/rollout/result.json',suite,paths('none')['robot'],paths('none')['protocol'],
                       root,root/'True/rollout/runtime.json')
