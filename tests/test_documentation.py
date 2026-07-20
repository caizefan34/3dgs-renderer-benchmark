import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentationConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ranking = json.loads(
            (ROOT / "docs" / "leaderboard" / "ranking.json").read_text(
                encoding="utf-8"
            )
        )

    def test_comparison_analysis_matches_generated_overall_rows(self):
        measured = self.ranking["tiers"]["measured"]
        analysis = (ROOT / "docs" / "comparison-analysis.md").read_text(
            encoding="utf-8"
        )

        self.assertEqual(len(measured["overall"]), 5)
        for row in measured["overall"]:
            self.assertIn(f"{row['fps']:.2f}", analysis)
            self.assertIn(f"{row['speed_index']:.3f}x", analysis)
        normalized_analysis = (
            analysis.replace("-", "").replace("_", "").replace(" ", "").lower()
        )
        for renderer in measured["recommendations"].values():
            self.assertIn(renderer.replace("_", "").lower(), normalized_analysis)

    def test_public_entry_points_describe_complete_tier_a(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        index = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")

        self.assertIn("complete Tier A coverage", readme)
        self.assertIn("Tier A complete: 25/25 runs", index)
        self.assertNotIn("No overall Matrix v2 winner yet", index)

    def test_current_and_historical_reports_are_separated(self):
        reports = ROOT / "reports"
        archive = reports / "archive" / "windows-rtx5070-2026-07"

        self.assertTrue((reports / "benchmark_report.md").is_file())
        self.assertTrue((reports / "machine_report.md").is_file())
        self.assertTrue((archive / "failure_report.md").is_file())
        self.assertFalse((reports / "generated").exists())


if __name__ == "__main__":
    unittest.main()
