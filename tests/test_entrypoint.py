import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from motion_challenge.entrypoint import artifact, resolve_artifacts


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
