import copy
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from motion_challenge.batch_physics import BatchPool
from motion_challenge.dataset import load_csv_motion
from motion_challenge.physics import World
from motion_challenge.run import evaluate_clip
from motion_challenge.sdk import ControlCommand
from motion_challenge.validation import sha256_file
from motion_challenge.verify import verify

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT/'robots/g1-29dof-cpu-v0.json'


class Worker:
    def call(self, op, state, reference, seed=0):
        if op == 'act':
            return ControlCommand((0.,)*29, (0.,)*29, (20.,)*29, (1.,)*29, (0.,)*29)


class NoTerminationTests(unittest.TestCase):
    def test_full_trace_first_failure_and_batch_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'clip.csv'
            source.write_text((','.join(map(str,[0,0,.8,0,0,0,1]+[0]*29))+'\n')*31)
            manifest = json.loads((ROOT/'suites/lafan/dataset.json').read_text())
            manifest['files'] = [{'path':source.name,'sha256':sha256_file(source)}]
            suite = root/'suite.json'; suite.write_text(json.dumps(manifest))
            motion = load_csv_motion(source,manifest['joint_names'],manifest['source_fps'])
            protocol = json.loads((ROOT/'suites/lafan/protocol.mujoco317-no-termination-v1.json').read_text())
            protocol['failure']['anchor_height_m'] = 1e-6
            config = root/'protocol.json'; config.write_text(json.dumps(protocol))
            runtime = {'control_period_s':.02,'reference_offsets_s':[0.]}
            controller = root/'runtime.json'; controller.write_text(json.dumps(runtime))
            def run(world, spec):
                trace = io.StringIO()
                result = evaluate_clip(world,motion,Worker(),runtime,spec,trace=trace)
                result.pop('wall_time_s')
                return result,trace.getvalue()
            full, trace = run(World(PROFILE),protocol)
            early, early_trace = run(World(PROFILE),{**protocol,'terminate_on_failure':True})
            self.assertFalse(full['success'])
            self.assertAlmostEqual(full['observed_time_s'],motion.duration_s)
            self.assertEqual(full['failure_time_s'],early['failure_time_s'])
            self.assertEqual(full['completion'],early['completion'])
            self.assertTrue(trace.startswith(early_trace))
            self.assertGreater(full['sample_count'],early['sample_count'])
            self.assertEqual(BatchPool(PROFILE,2,2).run(lambda slot,world:run(world,protocol)),[(full,trace)]*2)
            # Recovery cannot erase the first failure or advance completion.
            ticks = iter([(['anchor_height'],{'error':1.})]+[([],{'error':0.})]*100)
            with patch('motion_challenge.run.tracking_sample',side_effect=lambda *args:next(ticks)):
                recovered, _ = run(World(PROFILE),protocol)
            self.assertFalse(recovered['success'])
            self.assertEqual(recovered['failure_reasons'],['anchor_height'])
            self.assertEqual(recovered['completion'],0.)
            self.assertAlmostEqual(recovered['failure_time_s'],.02)
            # Independent verifier accepts post-failure samples and rejects truncation.
            trace_path = root/'poses.jsonl.gz'
            with gzip.open(trace_path,'wt') as f:f.write(trace)
            full.update(id=source.name,trace={'path':trace_path.name,'sha256':sha256_file(trace_path)})
            result = dict(status='completed',full_suite=True,mujoco_version=__import__('mujoco').__version__,
                termination_mode='no_termination',clips=[full],success_count=0,macro_completion=full['completion'],
                macro_metrics_over_observed_prefix=full['metric_means_over_observed_prefix'],
                suite_sha256=sha256_file(suite),robot_profile_sha256=sha256_file(PROFILE),
                protocol_sha256=sha256_file(config),controller_config_sha256=sha256_file(controller))
            output = root/'result.json';output.write_text(json.dumps(result))
            self.assertTrue(verify(output,suite,PROFILE,config,root,controller)['full_suite'])
            for changed in ('failure_time','truncated_trace'):
                bad=copy.deepcopy(result)
                if changed=='failure_time':bad['clips'][0]['failure_time_s']=motion.duration_s
                else:
                    with gzip.open(trace_path,'wt') as f:f.write(early_trace)
                    bad['clips'][0].update(sample_count=early['sample_count'],trace={'path':trace_path.name,'sha256':sha256_file(trace_path)})
                output.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):verify(output,suite,PROFILE,config,root,controller)
