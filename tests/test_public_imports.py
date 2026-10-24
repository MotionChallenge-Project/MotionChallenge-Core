"""Published adapter imports and CLI commands survive internal file moves."""
import importlib
import subprocess
import sys
import unittest

from motion_challenge.resources import lafan_paths


class PublicImportsTests(unittest.TestCase):
    def test_compatibility_modules_share_the_implementation(self):
        locations = {
            'entrypoint': 'runtime', 'cli': 'runtime', 'physics': 'simulation',
            'run': 'evaluation', 'verify': 'evaluation', 'merge': 'evaluation',
        }
        for name, package in locations.items():
            with self.subTest(module=name):
                self.assertIs(importlib.import_module(f'motion_challenge.{name}'),
                              importlib.import_module(f'motion_challenge.{package}.{name}'))

    def test_released_cli_module_paths_still_work(self):
        for name in ('cli', 'run', 'verify', 'merge'):
            with self.subTest(module=name):
                result = subprocess.run([sys.executable, '-m', f'motion_challenge.{name}', '--help'],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_packaged_protocol_paths_still_resolve(self):
        for mode in ('early', 'none'):
            for path in lafan_paths(mode).values():
                self.assertTrue(path.is_file(), str(path))
