import copy
import unittest

from motion_challenge.evaluation.regression import compare


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.result = {"checkpoint_sha256": "abc", "clips": [
            {"name": "g1/test.csv", "duration_s": 1., "samples": 50,
             "success": True, "completion": 1., "failure_time_s": None,
             "full": {"mpjpe_xy_aligned_m": .01}, "prefix": {"mpjpe_xy_aligned_m": .01}}]}

    def test_identical(self):
        report = compare(self.result, self.result)
        self.assertEqual(report["success_changed"], [])
        self.assertEqual(report["metrics"]["full"]["mpjpe_xy_aligned_m"]["delta"], 0)

    def test_cohort_checkpoint_and_clock(self):
        for key, value in (("name", "other"), ("duration_s", 2.), ("samples", 49)):
            altered = copy.deepcopy(self.result)
            altered["clips"][0][key] = value
            with self.assertRaises(ValueError):
                compare(self.result, altered)
        altered = copy.deepcopy(self.result)
        altered["checkpoint_sha256"] = "def"
        with self.assertRaises(ValueError):
            compare(self.result, altered)

    def test_success_change_survives_equal_total(self):
        altered = copy.deepcopy(self.result)
        altered["clips"][0].update(success=False, completion=.5, failure_time_s=.52)
        self.assertEqual(compare(self.result, altered)["success_changed"], ["g1/test.csv"])


if __name__ == "__main__":
    unittest.main()
