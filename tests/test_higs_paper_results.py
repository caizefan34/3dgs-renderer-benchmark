import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_paper_protocol import build_experiment_plan
from higs_paper_results import HigsPaperResultError, validate_result, validate_result_set


class HigsPaperResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "benchmark" / "higs-paper-protocol.json").read_text(encoding="utf-8")
        )
        cls.job = build_experiment_plan(cls.protocol)[0]

    def _result(self):
        return {
            "schema_version": "1.0",
            "job_id": self.job["job_id"],
            "status": "complete",
            "method": self.job["method"],
            "scene": self.job["scene"],
            "hardware": self.job["hardware"],
            "seed": self.job["seed"],
            "training": {"initialization": "from_scratch_sfm", "iterations": 30000},
            "performance": {"wall_time_seconds": 600.0, "time_to_quality_seconds": 500.0},
            "quality": {"psnr_db": 25.0, "ssim": 0.8, "lpips": 0.2},
            "resources": {"peak_gpu_memory_mib": 12000.0, "energy_joules": 100000.0,
                          "final_gaussian_count": 1000000},
            "quality_curve": [
                {"iteration": 7000, "wall_time_seconds": 100.0, "psnr_db": 20.0,
                 "ssim": 0.6, "lpips": 0.4},
                {"iteration": 30000, "wall_time_seconds": 600.0, "psnr_db": 25.0,
                 "ssim": 0.8, "lpips": 0.2},
            ],
            "artifact": {"sha256": "a" * 64},
            "provenance": {"timing_boundary": "dataset_ready_to_final_checkpoint",
                           "clean_process": True},
        }

    def test_accepts_complete_from_scratch_result(self):
        validate_result(self._result(), self.protocol)

    def test_rejects_pretrained_or_incomplete_quality_curve(self):
        result = self._result()
        result["training"]["initialization"] = "pretrained_ply"
        with self.assertRaisesRegex(HigsPaperResultError, "initialization"):
            validate_result(result, self.protocol)

        result = self._result()
        result["quality_curve"][-1]["iteration"] = 29999
        with self.assertRaisesRegex(HigsPaperResultError, "final iteration"):
            validate_result(result, self.protocol)

    def test_result_set_reports_missing_jobs_and_rejects_duplicates(self):
        result = self._result()
        summary = validate_result_set([result], self.protocol)
        self.assertEqual(summary["complete"], 1)
        self.assertGreater(summary["missing"], 300)
        with self.assertRaisesRegex(HigsPaperResultError, "duplicate job_id"):
            validate_result_set([result, copy.deepcopy(result)], self.protocol)


if __name__ == "__main__":
    unittest.main()
