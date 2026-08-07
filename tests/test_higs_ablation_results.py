import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_ablation_protocol import build_ablation_experiment_plan  # noqa: E402
from higs_ablation_results import (  # noqa: E402
    HigsAblationResultError,
    validate_ablation_result,
    validate_ablation_result_set,
)


class HigsAblationResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "benchmark" / "higs-ablation-protocol.json").read_text(
                encoding="utf-8"
            )
        )
        cls.job = build_ablation_experiment_plan(cls.protocol)[0]

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
            "performance": {
                "wall_time_seconds": 600.0,
                "time_to_quality_seconds": 500.0,
            },
            "quality": {"psnr_db": 25.0, "ssim": 0.8, "lpips": 0.2},
            "resources": {
                "peak_gpu_memory_mib": 12000.0,
                "energy_joules": 100000.0,
                "final_gaussian_count": 1000000,
            },
            "quality_curve": [
                {
                    "iteration": 7000,
                    "wall_time_seconds": 100.0,
                    "psnr_db": 20.0,
                    "ssim": 0.6,
                    "lpips": 0.4,
                },
                {
                    "iteration": 30000,
                    "wall_time_seconds": 600.0,
                    "psnr_db": 25.0,
                    "ssim": 0.8,
                    "lpips": 0.2,
                },
            ],
            "artifact": {"sha256": "a" * 64},
            "provenance": {
                "timing_boundary": "dataset_ready_to_final_checkpoint",
                "clean_process": True,
            },
        }

    def test_accepts_complete_ablation_result(self):
        validate_ablation_result(self._result(), self.protocol)

    def test_rejects_result_from_other_protocol(self):
        result = self._result()
        result["job_id"] = "ablation_pilot_30k--not-a-job--a100--s0"
        with self.assertRaises(HigsAblationResultError):
            validate_ablation_result(result, self.protocol)

    def test_result_set_reports_missing_jobs(self):
        summary = validate_ablation_result_set([self._result()], self.protocol)
        self.assertEqual(summary["complete"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["missing"], 39)


if __name__ == "__main__":
    unittest.main()
