import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from motion_challenge.entrypoint import artifact, resolve_artifacts, run_pair


class ArtifactTests(unittest.TestCase):
    def test_component_override_is_hashed_and_bound_to_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'cached.onnx';path.write_bytes(b'encoder fixture')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            runtime={'artifacts':{'encoder':{'path':'checkpoints/encoder.onnx','sha256':digest,'url':'https://example.org/encoder'}}}
            with patch('urllib.request.urlopen', side_effect=AssertionError('must not download')):
                result=resolve_artifacts(runtime,root,[f'encoder={path}'],offline=True)
            self.assertEqual(result['encoder']['path'],str(path.resolve()))
            self.assertEqual(result['encoder']['sha256'],digest)
            self.assertEqual(runtime['artifacts']['encoder']['path'],'checkpoints/encoder.onnx')
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                resolve_artifacts(runtime,root,[f'encoder={path}'],offline=True)

    def test_unknown_duplicate_or_malformed_component_override_rejected(self):
        runtime={'artifacts':{'encoder':{}}}
        for values in (['missing=x'],['encoder'],['encoder='],['encoder=x','encoder=y']):
            with self.assertRaisesRegex(ValueError,'unique declared'):
                resolve_artifacts(runtime,Path('.'),values,offline=True)

    def test_local_checkpoint_reused_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.onnx'
            path.write_bytes(b'checkpoint fixture')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('urllib.request.urlopen', side_effect=AssertionError('must not download')):
                self.assertEqual(artifact(path, expected, 'https://example.org/model'), path)

    def test_wrong_checkpoint_rejected_instead_of_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.onnx'
            path.write_bytes(b'wrong checkpoint')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                artifact(path, '0'*64, 'https://example.org/model')
            self.assertEqual(path.read_bytes(), b'wrong checkpoint')

    def test_offline_missing_artifact_fails_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('urllib.request.urlopen', side_effect=AssertionError('must not download')):
                with self.assertRaises(FileNotFoundError):
                    artifact(Path(directory)/'absent', '0'*64, 'https://example.org/model', True)


class PairedRunTests(unittest.TestCase):
    def test_pair_keeps_separate_scores_and_runs_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            def mode(root,args,runtime,dest,name,source,stamp):
                self.assertEqual(runtime['weights_sha256'],'fixed')
                clip={'id':'motion','success':False,'duration_s':2.,'completion':.5,'failure_time_s':1.,'failure_reasons':['height'], 'sample_count':50 if name=='early' else 100,'metric_means_over_observed_prefix':{'error':1 if name=='early' else 10}}
                (dest/'result.json').write_text(json.dumps({'controller_id':'test','controller_config_sha256':'same','suite_sha256':'same','robot_profile_sha256':'same','mujoco_version':'3.17.0','backend':'batch','clips':[clip],'success_count':0,'macro_completion':.5,
                    'macro_metrics_over_observed_prefix':clip['metric_means_over_observed_prefix']}))
                (dest/'receipt.json').write_text(json.dumps({'status':'trace_verified'}))
                return {'status':'trace_verified'}
            with patch('motion_challenge.entrypoint.run_mode',side_effect=mode) as run:
                result=run_pair(output,SimpleNamespace(),{'weights_sha256':'fixed'},output,{},'start')
            self.assertEqual(run.call_count,2)
            self.assertEqual(result['status'],'trace_verified')
            self.assertEqual(result['modes']['early']['metrics']['error'],1)
            self.assertEqual(result['modes']['none']['metrics']['error'],10)
            self.assertEqual(result['modes']['none']['result'],'none/result.json')
            unified=json.loads((output/'result.json').read_text())
            self.assertEqual(unified['total'],1)
            self.assertEqual(unified['success_count'],0)
            self.assertEqual(unified['metrics']['full_motion']['error'],10)
            self.assertEqual(unified['metrics']['until_first_failure']['error'],1)

    def test_failure_preserves_other_mode_and_does_not_publish_failed_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            def mode(root,args,runtime,dest,name,source,stamp):
                (dest/'result.json').write_text(json.dumps({'success_count':1,'macro_completion':1,
                    'macro_metrics_over_observed_prefix':{}}))
                if name=='early':raise ValueError('failed verification')
                return {'status':'trace_verified'}
            with patch('motion_challenge.entrypoint.run_mode',side_effect=mode):
                with self.assertRaisesRegex(RuntimeError,'early'):
                    run_pair(output,SimpleNamespace(),{},output,{},'start')
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['status'],'error')
            self.assertNotIn('metrics',summary['modes']['early'])
            self.assertEqual(summary['modes']['none']['status'],'trace_verified')
