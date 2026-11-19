"""CPU-only artifact preparation, asset identity, and cache corruption regressions."""
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from motion_challenge.runtime import mjwarp_entrypoint as entrypoint


def digest(value):
    return hashlib.sha256(value).hexdigest()


class ArtifactCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        previous_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), previous_path))

    def install_module(self, module):
        patch = mock.patch.dict(sys.modules, {module.__name__: module})
        patch.start()
        self.addCleanup(patch.stop)

    def exporter(self):
        calls = []
        content = b"original-author-checkpoint"
        module = types.ModuleType("fixture_original_export")

        def prepare(destination, cache=None):
            calls.append(cache)
            Path(destination).write_bytes(content)

        module.prepare = prepare
        self.install_module(module)
        runtime = {"id": "fixture-model", "prepare_checkpoint": module.__name__+":prepare",
                   "weights_sha256": digest(content)}
        return runtime, module, calls, content

    def test_original_export_and_verified_cache_reuse(self):
        runtime, module, calls, content = self.exporter()
        cache = self.directory/"cache"
        destination = cache/"original/policy.onnx"
        result = entrypoint.prepared_checkpoint(self.directory, runtime, destination, cache, False)
        self.assertEqual(result.read_bytes(), content)
        self.assertEqual(calls, [cache/"checkpoint-preparation/fixture-model"])
        self.assertEqual(entrypoint.prepared_checkpoint(self.directory, runtime, destination, cache, True), result)
        self.assertEqual(len(calls), 1)

    def test_corrupt_export_is_not_published(self):
        runtime, module, calls, _ = self.exporter()
        module.prepare = lambda destination, cache=None: Path(destination).write_bytes(b"wrong-model")
        destination = self.directory/"cache/original/policy.onnx"
        with self.assertRaisesRegex(ValueError, "Prepared checkpoint checksum mismatch"):
            entrypoint.prepared_checkpoint(self.directory, runtime, destination, self.directory/"cache", False)
        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.glob("tmp*")), [])

    def test_offline_missing_checkpoint_never_runs_exporter(self):
        runtime, module, calls, _ = self.exporter()
        with self.assertRaisesRegex(FileNotFoundError, "No valid prepared checkpoint"):
            entrypoint.prepared_checkpoint(self.directory, runtime, self.directory/"absent.onnx", self.directory, True)
        self.assertEqual(calls, [])

    def test_tensor_assets_reject_missing_changed_and_external_files(self):
        root = self.directory/"model"
        root.mkdir()
        asset = root/"robot.json"
        asset.write_bytes(b"canonical robot")
        runtime = {"tensor_assets_sha256": {asset.name: digest(asset.read_bytes())}}
        self.assertEqual(entrypoint.tensor_assets(root, runtime), runtime["tensor_assets_sha256"])
        asset.write_bytes(b"modified robot")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            entrypoint.tensor_assets(root, runtime)
        asset.unlink()
        with self.assertRaisesRegex(ValueError, "Missing model tensor asset"):
            entrypoint.tensor_assets(root, runtime)
        external = self.directory/"outside.json"
        external.write_bytes(b"external")
        with self.assertRaisesRegex(ValueError, "Missing model tensor asset"):
            entrypoint.tensor_assets(root, {"tensor_assets_sha256": {"../outside.json": digest(external.read_bytes())}})

    def test_converted_cache_rebuilds_corruption_and_tracks_asset_content(self):
        root = self.directory/"model"
        root.mkdir()
        (root/"policy.py").write_text("MODEL_VERSION = 1\n")
        asset = root/"robot.json"
        asset.write_bytes(b"robot-one")
        source = self.directory/"original.onnx"
        source.write_bytes(b"verified-original")
        resources = self.directory/"core"
        shared = resources/"src/motion_challenge/networks"
        shared.mkdir(parents=True)
        (shared/"common.py").write_text("NETWORK_VERSION = 1\n")
        patch = mock.patch.object(entrypoint, "resource_root", lambda: resources)
        patch.start()
        self.addCleanup(patch.stop)
        calls = []
        module = types.ModuleType("fixture_tensor_converter")

        def prepare(model_root, runtime, original, output, artifacts, device):
            calls.append(device)
            (output/"weights.bin").write_bytes(original.read_bytes()+asset.read_bytes())

        module.prepare = prepare
        self.install_module(module)
        runtime = {"id": "fixture-model", "weights_sha256": digest(source.read_bytes()),
                   "inference_config_sha256": digest(b"configuration"),
                   "tensor_conversion": module.__name__+":prepare",
                   "tensor_assets_sha256": {asset.name: digest(asset.read_bytes())}}
        cache = self.directory/"cache"
        result = entrypoint.converted_weights(root, runtime, source, {}, cache, "cpu")
        expected = source.read_bytes()+asset.read_bytes()
        self.assertEqual((result/"weights.bin").read_bytes(), expected)
        self.assertEqual(entrypoint.converted_weights(root, runtime, source, {}, cache, "cpu"), result)
        self.assertEqual(calls, ["cpu"])
        receipt = json.loads((result/"conversion-receipt.json").read_text())
        self.assertEqual(receipt["identity"]["tensor_assets_sha256"], runtime["tensor_assets_sha256"])

        (result/"weights.bin").write_bytes(b"corrupted-cache")
        self.assertEqual(entrypoint.converted_weights(root, runtime, source, {}, cache, "cpu"), result)
        self.assertEqual((result/"weights.bin").read_bytes(), expected)
        self.assertEqual(calls, ["cpu", "cpu"])
        backups = list(result.parent.glob(result.name+".invalid-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0]/"weights.bin").read_bytes(), b"corrupted-cache")

        asset.write_bytes(b"robot-two")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            entrypoint.converted_weights(root, runtime, source, {}, cache, "cpu")
        runtime["tensor_assets_sha256"][asset.name] = digest(asset.read_bytes())
        newer = entrypoint.converted_weights(root, runtime, source, {}, cache, "cpu")
        self.assertNotEqual(newer, result)
        self.assertEqual((newer/"weights.bin").read_bytes(), source.read_bytes()+asset.read_bytes())
        self.assertEqual(calls, ["cpu", "cpu", "cpu"])


if __name__ == "__main__":
    unittest.main()
