import math
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motion_challenge.metrics import EpisodeOutcome, orientation_error_rad, summarize
from motion_challenge.reference import Motion, reference_window
from motion_challenge.validation import confined_path, external_output_path, validate_controller, verify_artifacts


class ContractTests(unittest.TestCase):
    def test_outputs_reject_repository_and_symlink_into_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root/'repo'
            repo.mkdir()
            (repo/'.git').mkdir()
            (root/'alias').symlink_to(repo, target_is_directory=True)
            for path in (repo/'runs/result.json', root/'alias/result.json'):
                with self.assertRaises(ValueError):
                    external_output_path(path)
            self.assertEqual(external_output_path(root/'runs/result.json'), root/'runs/result.json')

    def test_reference_history_is_explicit_and_clamped(self):
        with self.assertRaises(ValueError):
            reference_window(self.motion(), 0, (-0.2, 0.0), 0)
        frames = reference_window(self.motion(), 0, (-0.2, 0.0), 0, 0.2)
        self.assertEqual(frames[0].source_time_s, 0.0)

    def motion(self):
        return Motion(("joint",), (0.0, 1.0), ((0.0,), (2.0,)),
                      ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                      ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0)))

    def test_interpolation_keeps_same_time_root_joint_rotation(self):
        frame = self.motion().sample(0.5)
        self.assertEqual(frame.joint_position_rad, (1.0,))
        self.assertEqual(frame.root_position_world_m, (0.5, 0.0, 0.0))
        self.assertAlmostEqual(orientation_error_rad(frame.root_orientation_xyzw, (0, 0, 0, 1)), math.pi/2)

    def test_quaternion_sign_is_not_a_rotation_error(self):
        self.assertEqual(orientation_error_rad((0, 0, 0, 1), (0, 0, 0, -1)), 0)

    def test_preview_clamps_without_shortening_episode(self):
        motion = self.motion()
        frames = reference_window(motion, 0.9, (0.0, 0.2), 0.2)
        self.assertAlmostEqual(frames[1].requested_time_s, 1.1)
        self.assertEqual(frames[1].source_time_s, 1.0)
        self.assertEqual(frames[1].joint_position_rad, (2.0,))
        self.assertEqual(motion.duration_s, 1.0)

    def test_preview_access_is_bounded(self):
        with self.assertRaises(ValueError):
            reference_window(self.motion(), 0, (0, 0.3), 0.2)
        with self.assertRaises(ValueError):
            reference_window(self.motion(), 0, (0, float("nan")), 0.2)

    def test_failure_on_final_step_is_failure(self):
        outcome = EpisodeOutcome("a", 10, 10, 10)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.completion, 0.9)

    def test_missing_or_duplicate_clip_cannot_rank(self):
        outcome = EpisodeOutcome("a", 10, 10, None)
        for outcomes in ([outcome], [outcome, outcome]):
            with self.assertRaises(ValueError):
                summarize(outcomes, ["a", "b"])

    def test_early_crash_is_not_a_policy_failure_result(self):
        with self.assertRaises(ValueError):
            summarize([EpisodeOutcome("a", 10, 3, None)], ["a"])

    def test_macro_average_does_not_weight_by_motion_length(self):
        result = summarize([EpisodeOutcome("a", 10, 10, None),
                            EpisodeOutcome("b", 100, 51, 51)], ["a", "b"])
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["macro_completion_fraction"], 0.75)

    def test_artifacts_cannot_escape_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            root.mkdir()
            (root/"escape").symlink_to(root.parent, target_is_directory=True)
            for path in ("../outside", "/outside", "escape/outside"):
                with self.assertRaises(ValueError):
                    confined_path(root, path)

    def controller_manifest(self):
        core = Path(__file__).resolve().parents[1]
        manifest = json.loads((core/"tests/fixtures/controller_manifest.json").read_text())
        manifest["adapter"]["commit"] = "a" * 40
        manifest["runtime_image_digest"] = "sha256:" + "b" * 64
        for artifact in manifest["artifacts"]:
            artifact["sha256"] = hashlib.sha256(b"test fixture").hexdigest()
        return manifest

    def test_unknown_override_and_wrong_joint_order_are_rejected(self):
        manifest = self.controller_manifest()
        joints = list(manifest["joint_names"])
        validate_controller(manifest, joints)
        wrong = copy.deepcopy(manifest)
        wrong["friction_override"] = 0.1
        with self.assertRaises(ValueError):
            validate_controller(wrong, joints)
        wrong = copy.deepcopy(manifest)
        wrong["joint_names"][0:2] = reversed(wrong["joint_names"][0:2])
        with self.assertRaises(ValueError):
            validate_controller(wrong, joints)

    def test_artifact_mutation_is_detected(self):
        manifest = self.controller_manifest()
        with tempfile.TemporaryDirectory() as root:
            for artifact in manifest["artifacts"]:
                (Path(root)/artifact["path"]).write_bytes(b"test fixture")
            self.assertEqual(verify_artifacts(manifest, root), 2)
            (Path(root)/manifest["artifacts"][0]["path"]).write_bytes(b"modified")
            with self.assertRaises(ValueError):
                verify_artifacts(manifest, root)


if __name__ == "__main__":
    unittest.main()
