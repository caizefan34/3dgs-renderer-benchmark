import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_matrix import MatrixValidationError, generate_matrix_report, validate_result, write_report


class BenchmarkMatrixTest(unittest.TestCase):
    def setUp(self):
        self.suite = {
            "suite_id": "suite",
            "version": "2.0",
            "primary_track": "common_representation",
            "protocol_sha256": "f" * 64,
            "ranking": {"reference_renderer_id": "reference", "primary_resolution": [1920, 1080]},
            "cases": [{"case_id": "a"}, {"case_id": "b"}],
        }

    def result(self, renderer, case, tier="measured", fps=100, psnr=30, lpips=.1):
        source_type = {
            "measured": "repository_run",
            "reproduced": "official_implementation_run",
            "paper_reported": "paper",
        }[tier]
        provenance = {
            "source_type": source_type,
            "source_uri": "https://example.invalid/source",
            "measured_at": "2026-01-01T00:00:00Z",
            "raw_samples_uri": "raw.json",
            "raw_samples_sha256": "a" * 64,
        }
        if tier == "paper_reported":
            provenance.update({"citation": "Example et al.", "paper_table_or_figure": "Table 1"})
        return {
            "schema_version": "2.0",
            "result_id": f"{tier}-{renderer}-{case}",
            "evidence_tier": tier,
            "status": "complete",
            "renderer": {
                "id": renderer, "config_id": renderer, "name": renderer, "version": "1",
                "source_uri": "https://example.invalid/renderer", "source_commit": "b" * 40,
                "build_command": "pip install .", "runtime_command": "benchmark run",
                "api": "PyTorch", "backend": "CUDA", "platforms": ["Linux"], "features": ["SH"],
            },
            "benchmark": {
                "suite_id": "suite", "suite_version": "2.0", "track_id": "common_representation",
                "case_id": case, "dataset_id": "dataset", "dataset_sha256": "2" * 64,
                "scene_id": case, "scene_tier": "small",
                "checkpoint_sha256": "c" * 64, "camera_trajectory_id": "eval",
                "gaussian_count": 1000, "sh_degree": 3,
                "camera_trajectory_sha256": "d" * 64, "quality_reference_sha256": "e" * 64,
                "resolution": {"width": 1920, "height": 1080}, "color_space": "sRGB",
                "background": "black", "protocol_id": "p",
                "protocol_sha256": "f" * 64,
            },
            "environment": {
                "hardware_profile_id": "gpu", "gpu": "GPU", "gpu_uuid": "uuid", "gpu_vram_mb": 8192,
                "cpu": "CPU", "ram_mb": 32768, "os": "Linux", "driver": "1",
                "cuda": "13", "python": "3.12", "pytorch": "2", "benchmark_commit": "1" * 40,
                "clock_policy": "default", "power_limit_w": None,
            },
            "metrics": {
                "performance": {"fps": fps, "fps_ci95_low": fps * .95, "fps_ci95_high": fps * 1.05,
                                "frame_time_ms": 1000 / fps, "peak_vram_mb": 1024,
                                "p95_frame_time_ms": 11, "p99_frame_time_ms": 12,
                                "startup_time_ms": 10, "renderer_init_time_ms": 5,
                                "scene_load_time_ms": 100,
                                "renderer_prepare_time_ms": 20, "time_to_first_frame_ms": 30},
                "quality": {"psnr_db": psnr, "ssim": .95, "lpips": lpips},
                "raw_samples": {"frame_time_ms": [1000 / fps]},
            },
            "provenance": provenance,
        }

    def test_rejects_tier_provenance_mismatch(self):
        result = self.result("r", "a")
        result["provenance"]["source_type"] = "paper"
        with self.assertRaises(MatrixValidationError):
            validate_result(result)

    def test_never_mixes_evidence_tiers(self):
        documents = []
        for tier in ("measured", "reproduced"):
            documents.extend((self.result("r", "a", tier), self.result("r", "b", tier)))
        report = generate_matrix_report(documents, self.suite)
        self.assertEqual(len(report["tiers"]["measured"]["overall"]), 1)
        self.assertEqual(len(report["tiers"]["reproduced"]["overall"]), 1)
        self.assertEqual(report["tier_policy"], "never_mix")
        self.assertEqual(report["preferred_tier"], "measured")

    def test_incomplete_renderer_is_excluded_from_overall(self):
        report = generate_matrix_report([self.result("r", "a")], self.suite)
        self.assertEqual(report["tiers"]["measured"]["overall"], [])
        self.assertEqual(report["tiers"]["measured"]["excluded"]["r"]["missing_cases"], ["b"])

    def test_generates_both_requested_two_dimensional_frontiers(self):
        documents = []
        for case in ("a", "b"):
            documents.append(self.result("fast", case, fps=200, psnr=29, lpips=.12))
            documents.append(self.result("quality", case, fps=100, psnr=32, lpips=.06))
            documents.append(self.result("dominated", case, fps=90, psnr=28, lpips=.15))
        report = generate_matrix_report(documents, self.suite)["tiers"]["measured"]
        self.assertEqual(report["pareto_frontiers"]["speed_psnr"], ["fast", "quality"])
        self.assertEqual(report["pareto_frontiers"]["speed_lpips"], ["fast", "quality"])
        self.assertNotIn("dominated", report["pareto_frontiers"]["combined"])

    def test_duplicate_case_rows_are_not_aggregated(self):
        duplicate = copy.deepcopy(self.result("r", "a"))
        duplicate["result_id"] = "duplicate"
        report = generate_matrix_report(
            [self.result("r", "a"), duplicate, self.result("r", "b")], self.suite
        )
        self.assertEqual(report["tiers"]["measured"]["overall"], [])

    def test_hardware_cohorts_never_share_an_overall_table(self):
        documents = [self.result("r", "a"), self.result("r", "b")]
        other = [self.result("s", "a"), self.result("s", "b")]
        for row in other:
            row["environment"]["hardware_profile_id"] = "other-gpu"
            row["environment"]["gpu"] = "Other GPU"
        report = generate_matrix_report(documents + other, self.suite)["tiers"]["measured"]
        self.assertEqual(report["overall"], [])
        self.assertEqual(len(report["suite_cohorts"]), 2)

    def test_wrong_protocol_cannot_enter_a_ranking(self):
        result = self.result("r", "a")
        result["benchmark"]["protocol_sha256"] = "0" * 64
        report = generate_matrix_report([result], self.suite)
        self.assertEqual(report["tiers"]["measured"]["overall"], [])
        self.assertEqual(report["rejected"][0]["reason"], "protocol hash mismatch")

    def test_different_checkpoints_for_one_case_cannot_share_overall(self):
        documents = []
        for renderer in ("r", "s"):
            documents.extend((self.result(renderer, "a"), self.result(renderer, "b")))
        documents[2]["benchmark"]["checkpoint_sha256"] = "9" * 64
        report = generate_matrix_report(documents, self.suite)["tiers"]["measured"]
        self.assertEqual(report["overall"], [])
        self.assertIn("multiple immutable cohorts", report["excluded"]["r"]["reason"])

    def test_empty_report_is_explicit_and_writable(self):
        report = generate_matrix_report([], self.suite)
        with tempfile.TemporaryDirectory() as temp_dir:
            write_report(report, temp_dir)
            payload = json.loads((Path(temp_dir) / "ranking.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["tiers"]["measured"]["overall"], [])


if __name__ == "__main__":
    unittest.main()
