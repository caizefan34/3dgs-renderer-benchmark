import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RendererResearchExperimentsTest(unittest.TestCase):
    def test_all_thirty_ideas_have_explicit_epic05_acceptance(self):
        document = json.loads(
            (ROOT / "benchmark" / "renderer-research-experiments.json").read_text(encoding="utf-8")
        )
        experiments = document["experiments"]
        self.assertEqual(document["authority_host"], "EPIC-05")
        self.assertEqual([row["id"] for row in experiments], list(range(1, 31)))
        self.assertTrue(all(row["baseline"] and row["acceptance"] for row in experiments))
        self.assertTrue(all(row["status"] != "measured" for row in experiments))


if __name__ == "__main__":
    unittest.main()
