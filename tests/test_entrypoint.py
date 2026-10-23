import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from motion_challenge.entrypoint import artifact


class ArtifactTests(unittest.TestCase):
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
