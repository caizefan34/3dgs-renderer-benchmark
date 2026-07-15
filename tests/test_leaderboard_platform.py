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
from benchmark_suite import resolve_suite_case
from benchmark_framework import RendererMetrics
from leaderboard.generator import generate_leaderboard, load_records, write_leaderboard
from schema_validation import SchemaValidationError, validate_schema


class LeaderboardGenerationTest(unittest.TestCase):
    def _official(self, **values):
        case = resolve_suite_case("garden", "1080p", verify_assets=False)
        return {
            "official_eligible": True,
            "official": True,
            "validated": True,
            "suite_version": case["suite_version"],
            **{key: case[key] for key in (
                "suite_id", "suite_case_id", "dataset_sha256", "scene_sha256",
                "camera_sha256", "resolution_profile", "resolution",
            )},
            "hardware": "Test GPU",
            **values,
        }

    def test_generates_quality_constrained_and_efficiency_leaderboards(self):
        records = [
            self._official(renderer="fast30", fps=200, psnr=30.5, ssim=.94, lpips=.12),
            self._official(renderer="slow32", fps=120, psnr=32.2, ssim=.96, lpips=.08),
            self._official(renderer="fail", fps=300, psnr=29.9, ssim=.93, lpips=.14),
        ]

        leaderboard = generate_leaderboard(records)

        self.assertEqual(leaderboard["benchmark_suite_version"], BENCHMARK_SUITE_VERSION)
        constrained = leaderboard["leaderboards"]["quality_constrained"]
        self.assertEqual([row["renderer"] for row in constrained["30"]], ["fast30", "slow32"])
        self.assertEqual([row["renderer"] for row in constrained["31"]], ["slow32"])
        self.assertEqual([row["renderer"] for row in constrained["32"]], ["slow32"])
        self.assertEqual(leaderboard["leaderboards"]["efficiency"][0]["renderer"], "fast30")

    def test_official_ranking_excludes_unvalidated_records(self):
        forged = self._official(renderer="forged", fps=1000, psnr=40, ssim=1, lpips=0)
        forged["scene_sha256"] = "0" * 64
        leaderboard = generate_leaderboard([
            {"renderer": "arbitrary", "fps": 999, "psnr": 40, "official_eligible": False},
            forged,
            self._official(renderer="official", fps=100, psnr=31, ssim=.95, lpips=.1),
        ])

        self.assertEqual(leaderboard["source_record_count"], 3)
        self.assertEqual(leaderboard["official_record_count"], 1)
        self.assertEqual(leaderboard["leaderboards"]["efficiency"][0]["renderer"], "official")

    def test_merges_speed_and_quality_for_the_same_suite_case(self):
        records = [
            self._official(renderer="joined", benchmark_type="real_scene_speed", fps=180),
            self._official(renderer="joined", benchmark_type="real_scene_quality", psnr=31.5, ssim=.95, lpips=.1),
        ]

        leaderboard = generate_leaderboard(records)

        row = leaderboard["leaderboards"]["quality_constrained"]["31"][0]
        self.assertEqual(row["renderer"], "joined")
        self.assertEqual(row["fps"], 180)

    def test_loads_and_joins_official_speed_and_quality_documents(self):
        case = resolve_suite_case("garden", "1080p", verify_assets=False)
        suite = {
            **{key: value for key, value in case.items() if key not in {"paths", "protocol"}},
            "validated": True,
        }
        speed = {
            "benchmark_suite": suite,
            "environment": {"gpu": "Test GPU"},
            "protocol": {"resolution": [1920, 1080]},
            "results": {"r": {"renderer": "r", "mean_fps": 100}},
        }
        quality = {
            "benchmark_suite": suite,
            "environment": {"gpu": "Test GPU"},
            "benchmark_type": "real_scene_quality",
            "results": [{
                "renderer": "r",
                "quality": {"mean_psnr_db": 31, "mean_ssim": .95, "mean_lpips": .1},
            }],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            speed_path = Path(temp_dir) / "speed.json"
            quality_path = Path(temp_dir) / "quality.json"
            speed_path.write_text(json.dumps(speed), encoding="utf-8")
            quality_path.write_text(json.dumps(quality), encoding="utf-8")
            leaderboard = generate_leaderboard(load_records([str(speed_path), str(quality_path)]))

        self.assertEqual(
            leaderboard["leaderboards"]["quality_constrained"]["31"][0]["renderer"],
            "r",
        )

    def test_writes_three_artifacts(self):
        leaderboard = generate_leaderboard([
            self._official(renderer="a", fps=1.0, psnr=30, ssim=.9, lpips=.2)
        ])
        root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (root / "schemas" / "leaderboard.schema.json").read_text(encoding="utf-8")
        )
        validate_schema(leaderboard, schema)
        with tempfile.TemporaryDirectory() as temp_dir:
            write_leaderboard(leaderboard, temp_dir)
            self.assertTrue((Path(temp_dir) / "leaderboard.json").exists())
            self.assertTrue((Path(temp_dir) / "leaderboard.md").exists())
            self.assertTrue((Path(temp_dir) / "leaderboard.html").exists())
            self.assertTrue((Path(temp_dir) / "quality_speed.html").exists())

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

