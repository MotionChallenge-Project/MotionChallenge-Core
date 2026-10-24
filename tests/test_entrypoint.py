import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from motion_challenge.entrypoint import artifact, resolve_artifacts, run_pair, source_snapshot


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
    def test_one_rollout_yields_both_metric_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            def mode(root,args,runtime,dest,name,source,stamp,derive_prefix=False):
                self.assertEqual(name,'none')
                self.assertTrue(derive_prefix)
                self.assertEqual(runtime['weights_sha256'],'fixed')
                for scope, filename in [('early','prefix-result.json'), ('none','result.json')]:
                    clip={'id':'motion','success':False,'duration_s':2.,'completion':.5,'failure_time_s':1.,'failure_reasons':['height'], 'sample_count':50 if scope=='early' else 100,'metric_means_over_observed_prefix':{'error':1 if scope=='early' else 10}}
                    (dest/filename).write_text(json.dumps({'state_access':'ideal-proprioceptive-no-position-v1','controller_id':'test','controller_config_sha256':'same','suite_sha256':'same','robot_profile_sha256':'same','mujoco_version':'3.17.0','backend':'batch','clips':[clip],'success_count':0,'macro_completion':.5,
                        'macro_metrics_over_observed_prefix':clip['metric_means_over_observed_prefix']}))
                (dest/'receipt.json').write_text(json.dumps({'status':'trace_verified'}))
                return {'status':'trace_verified'}
            with patch('motion_challenge.entrypoint.run_mode',side_effect=mode) as run:
                result=run_pair(output,SimpleNamespace(),{'weights_sha256':'fixed'},output,{},'start')
            self.assertEqual(run.call_count,1)
            self.assertEqual(result['status'],'trace_verified')
            self.assertEqual(result['modes']['early']['metrics']['error'],1)
            self.assertEqual(result['modes']['none']['metrics']['error'],10)
            self.assertEqual(result['modes']['none']['result'],'rollout/result.json')
            unified=json.loads((output/'result.json').read_text())
            self.assertEqual(unified['total'],1)
            self.assertEqual(unified['success_count'],0)
            self.assertEqual(unified['metrics']['full_motion']['error'],10)
            self.assertEqual(unified['metrics']['until_first_failure']['error'],1)

    def test_failed_trace_does_not_publish_a_unified_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            with patch('motion_challenge.entrypoint.run_mode',side_effect=ValueError('failed verification')):
                with self.assertRaisesRegex(ValueError,'failed verification'):
                    run_pair(output,SimpleNamespace(),{},output,{},'start')
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['status'],'error')
            self.assertFalse((output/'result.json').exists())


class SourceLayoutTests(unittest.TestCase):
    def init_repo(self, path):
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', '-q', str(path)], check=True)
        (path/'fixture.txt').write_text('source fixture')
        subprocess.run(['git', '-C', str(path), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(path), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()

    def test_core_identity_and_dirty_state_in_both_layouts(self):
        for name in ('MotionChallenge-Core', 'motion-challenge'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'adapter';self.init_repo(root)
                core=root/name;revision=self.init_repo(core)
                self.assertEqual(source_snapshot(root)['core_commit'], revision)
                self.assertFalse(source_snapshot(root)['core_dirty'])
                (core/'fixture.txt').write_text('changed core')
                self.assertTrue(source_snapshot(root)['core_dirty'])

    def test_ambiguous_or_missing_core_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.init_repo(root)
            with self.assertRaisesRegex(ValueError, 'one core submodule'):
                source_snapshot(root)
            for name in ('MotionChallenge-Core', 'motion-challenge'):
                self.init_repo(root/name)
            with self.assertRaisesRegex(ValueError, 'one core submodule'):
                source_snapshot(root)
