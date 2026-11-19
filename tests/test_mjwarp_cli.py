"""Public commands and resource contracts for the supported MJWarp pipeline."""
import json
import subprocess
import sys
import unittest

from motion_challenge.resources import resource_root


class PublicCommandTests(unittest.TestCase):
    def test_dataset_selecting_command_and_verification_tools(self):
        for module in ['motion_challenge.runtime.mjwarp_entrypoint',
                       'motion_challenge.evaluation.tensor_verify',
                       'motion_challenge.evaluation.reproduction']:
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, '-m', module, '--help'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)
                if module.endswith('mjwarp_entrypoint'):
                    for option in ('--dataset', '--data-format', '--device', '--slots', '--offline'):
                        self.assertIn(option, result.stdout)
                    self.assertNotIn('--backend', result.stdout)

    def test_current_protocol_resources_match_control_and_observation_contract(self):
        for dataset in ('lafan', 'seed'):
            with self.subTest(dataset=dataset):
                path = resource_root() / f'suites/{dataset}/protocol.mjwarp311-no-linear-velocity-v1.json'
                protocol = json.loads(path.read_text())
                self.assertEqual(protocol['backend'], 'mjwarp')
                self.assertEqual(protocol['state_access'], 'proprioceptive-no-position-no-linear-velocity-v1')
                self.assertEqual(protocol['physics_substeps'], 4)
                self.assertEqual(protocol['physics_dt_s'], .005)
                self.assertEqual(protocol['control_period_s'], .02)
                self.assertEqual(protocol['metric_period_s'], .02)
                self.assertFalse(protocol['terminate_on_failure'])


if __name__ == '__main__':
    unittest.main()
