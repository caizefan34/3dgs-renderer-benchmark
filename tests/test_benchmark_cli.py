import contextlib
import io
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_cli import main


class BenchmarkCliTest(unittest.TestCase):
    def test_dry_run_expands_renderer_and_dataset(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["run", "gsplat", "--dataset", "garden", "--dry-run"])
        plan = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["renderer"], "gsplat")
        self.assertEqual(plan[0]["case"], "small-garden-1080p")
        self.assertIn("results", plan[0]["output"])
        self.assertIn("--width", plan[0]["command"])
        self.assertIn("--height", plan[0]["command"])

    def test_pending_adapter_cannot_claim_tier_a(self):
        with self.assertRaises(SystemExit):
            main(["run", "flashgs", "--dry-run"])

    def test_registered_configuration_resolves_to_its_family_spec(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["run", "gsplat_dense", "--dataset", "garden", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())[0]["renderer"], "gsplat_dense")

    def test_diagnostic_configuration_cannot_enter_primary_track(self):
        with self.assertRaises(SystemExit):
            main(["run", "speedy_splat_raw", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
