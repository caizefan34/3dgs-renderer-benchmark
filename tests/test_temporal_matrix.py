import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.run_temporal_matrix import build_plan


class TemporalMatrixTest(unittest.TestCase):
    def test_plan_uses_completed_metric_paths_and_case_gt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "benchmark").mkdir()
            (root / "benchmark" / "suite.json").write_text(json.dumps({
                "cases": [{"case_id": "case", "ground_truth_path": "datasets/case/gt"}],
            }), encoding="utf-8")
            session = {"completed": [{
                "case_id": "case", "renderer": "gsplat",
                "metrics_path": "results/measured/gsplat/case/metrics.json",
            }]}

            plan = build_plan(root, session)

        self.assertEqual(plan[0]["renderer"], "gsplat")
        self.assertTrue(plan[0]["ground_truth"].replace("\\", "/").endswith("datasets/case/gt"))


if __name__ == "__main__":
    unittest.main()
