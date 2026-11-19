import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motion_challenge.data.selection import seed_path_from_v3_name, select_source_files


class SelectionTests(unittest.TestCase):
    def test_source_mapping_preserves_mirrored_suffix(self):
        self.assertEqual(seed_path_from_v3_name("221011__crowd_cheer_001__A049_M__d49bad3bad81"),
                         "g1/csv/221011/crowd_cheer_001__A049_M.csv")
        for bad in ("../x", "221011__../x__d49bad3bad81", "no_source_identity"):
            with self.assertRaises(ValueError):
                seed_path_from_v3_name(bad)

    def fixture(self, directory):
        root = Path(directory)
        data = (json.dumps({"path": "g1/csv/221011/a.csv", "frames": 4}) + "\n").encode()
        ledger = root/"selection.jsonl.gz"
        ledger.write_bytes(gzip.compress(data, mtime=0))
        config = {"schema": "motionchallenge/dataset-selection-v1", "id": "test",
                  "expected_episodes": 1, "expected_frames": 4,
                  "selection": {"path": ledger.name, "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest()},
                  "exclude": [{"path": "g1/csv/221011/b.csv"}]}
        path = root/"filter.json"
        path.write_text(json.dumps(config))
        return path

    def test_cohort_does_not_expand_with_extra_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.fixture(d)
            rows, report = select_source_files(p, ["g1/csv/221011/a.csv", "g1/csv/221011/b.csv", "g1/csv/221011/new.csv"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(report["explicitly_excluded"], 1)
            self.assertEqual(report["outside_frozen_cohort"], 1)

    def test_missing_duplicates_tampering_and_traversal_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.fixture(d)
            for available in ([], ["../bad"], ["g1/csv/221011/a.csv"]*2):
                with self.assertRaises(ValueError):
                    select_source_files(p, available)
            (Path(d)/"selection.jsonl.gz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                select_source_files(p, ["g1/csv/221011/a.csv"])


if __name__ == "__main__":
    unittest.main()
