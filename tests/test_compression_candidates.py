import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CompressionCandidatesTest(unittest.TestCase):
    def test_candidates_have_comparability_and_epic05_status(self):
        document = json.loads(
            (ROOT / "benchmark" / "compression-candidates.json").read_text(encoding="utf-8")
        )
        candidates = document["candidates"]
        self.assertEqual(document["authority_host"], "EPIC-05")
        self.assertGreaterEqual(len(candidates), 10)
        self.assertEqual(len({row["id"] for row in candidates}), len(candidates))
        self.assertTrue(all(row["track"] and row["epic05_status"] for row in candidates))
        self.assertTrue(any(row["retraining"] for row in candidates))
        self.assertTrue(any(not row["retraining"] for row in candidates))


if __name__ == "__main__":
    unittest.main()
