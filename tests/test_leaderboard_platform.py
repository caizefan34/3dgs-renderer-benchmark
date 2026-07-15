import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.hardware import summarize_hardware_profile
from analysis.regression import compare_regressions
from analysis.roofline import roofline_classification
from benchmark_suite import BENCHMARK_SUITE_VERSION
from benchmark_framework import RendererMetrics
from leaderboard.generator import generate_leaderboard, load_records, write_leaderboard
from schema_validation import SchemaValidationError, validate_schema


class LeaderboardGenerationTest(unittest.TestCase):
    def test_generates_separate_leaderboards_without_quality_leakage(self):
        records = [
            {
                "renderer": "synthetic_fast",
                "benchmark_type": "synthetic_stress",
                "fps": 300,
                "peak_vram_mb": 500,
                "psnr": None,
                "ssim": None,
                "lpips": None,
            },
            {
                "renderer": "quality_ref",
                "benchmark_type": "real_scene_quality",
                "psnr": 30,
                "ssim": 0.95,
                "lpips": 0.1,
            },
            {
                "renderer": "balanced",
                "benchmark_type": "real_scene_speed",
                "fps": 150,
                "peak_vram_mb": 600,
                "psnr": 29.8,
                "ssim": 0.94,
                "lpips": 0.11,
            },
        ]

        leaderboard = generate_leaderboard(records)

        self.assertEqual(leaderboard["benchmark_suite_version"], BENCHMARK_SUITE_VERSION)
        self.assertEqual(leaderboard["leaderboards"]["speed"][0]["renderer"], "synthetic_fast")
        quality_renderers = {row["renderer"] for row in leaderboard["leaderboards"]["quality"]}
        self.assertNotIn("synthetic_fast", quality_renderers)
        self.assertIn("balanced", quality_renderers)

    def test_writes_three_artifacts(self):
        leaderboard = generate_leaderboard([{"renderer": "a", "fps": 1.0}])
        with tempfile.TemporaryDirectory() as temp_dir:
            write_leaderboard(leaderboard, temp_dir)
            self.assertTrue((Path(temp_dir) / "leaderboard.json").exists())
            self.assertTrue((Path(temp_dir) / "leaderboard.md").exists())
            self.assertTrue((Path(temp_dir) / "leaderboard.html").exists())

    def test_loads_existing_result_formats(self):
        root = Path(__file__).resolve().parents[1]
        records = load_records([
            str(root / "data" / "results" / "rtx5070_laptop_2026-07-13.json"),
            str(root / "data" / "results" / "rtx5070_train_reference_summary_2026-07-14.json"),
        ])

        self.assertTrue(any(record["renderer"] == "HiGS tile16" for record in records))
        self.assertTrue(any(record["renderer"] == "original_3dgs" for record in records))


class SchemaValidationTest(unittest.TestCase):
    def test_schema_validator_rejects_missing_required_key(self):
        schema = {"type": "object", "required": ["schema_version"]}
        with self.assertRaises(SchemaValidationError):
            validate_schema({}, schema)

    def test_committed_leaderboard_schema_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        instance = json.loads((root / "docs" / "leaderboard" / "leaderboard.json").read_text(encoding="utf-8"))
        schema = json.loads((root / "schemas" / "leaderboard.schema.json").read_text(encoding="utf-8"))

        validate_schema(instance, schema)

    def test_schema_validator_checks_additional_property_values(self):
        schema = {"type": "object", "additionalProperties": {"type": "number"}}
        validate_schema({"valid": 1.0}, schema)
        with self.assertRaises(SchemaValidationError):
            validate_schema({"invalid": "not-a-number"}, schema)


class RegressionDetectionTest(unittest.TestCase):
    def test_detects_fps_regression(self):
        baseline = [{"renderer": "gsplat", "benchmark_type": "synthetic_stress", "fps": 100.0}]
        candidate = [{"renderer": "gsplat", "benchmark_type": "synthetic_stress", "fps": 91.0}]

        report = compare_regressions(baseline, candidate, {"fps_drop_pct": 5.0})

        self.assertEqual(report["status"], "regression")
        self.assertEqual(report["regressions"][0]["renderer"], "gsplat")
        self.assertEqual(report["regressions"][0]["fps_change_pct"], -9.0)


class ResearchExtensionTest(unittest.TestCase):
    def test_roofline_classification(self):
        result = roofline_classification(2.0, 80.0, 40.0)

        self.assertEqual(result["classification"], "memory-bound")

    def test_hardware_profile_is_optional(self):
        profile = summarize_hardware_profile({"sm_utilization_pct": 75})

        self.assertTrue(profile["optional"])
        self.assertEqual(profile["sm_utilization_pct"], 75)


class MetadataExportTest(unittest.TestCase):
    def test_renderer_metrics_include_benchmark_metadata_fields(self):
        metrics = RendererMetrics(renderer_name="test", frame_times_ms=[1.0])
        metrics.compute()
        data = metrics.to_dict()

        self.assertEqual(data["benchmark_suite_version"], BENCHMARK_SUITE_VERSION)
        self.assertIn("renderer_commit_hash", data)
        self.assertIn("driver_version", data)
        self.assertIn("hardware_metadata", data)


if __name__ == "__main__":
    unittest.main()

