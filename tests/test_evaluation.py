import math
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.efficiency import (
    QualityAdjustmentConfig,
    calculate_quality_adjusted_efficiency,
)
from analysis.pareto import pareto_analysis
from analysis.recommendations import build_recommendations
from benchmark.difficulty import (
    DifficultyConfig,
    DifficultyInputs,
    calculate_difficulty,
)
from benchmark_framework import RendererMetrics
from run_full_benchmark import (
    _build_result_document,
    _collect_environment,
    _load_synthetic_difficulty_catalog,
    _preload_gsplat_extension,
)


class EnvironmentCollectionTest(unittest.TestCase):
    def test_cuda_visible_device_selects_physical_nvidia_smi_gpu(self):
        def fake_check_output(command, **kwargs):
            self.assertIn("--id=2", command)
            if "--query-gpu=driver_version" in command:
                return "580.105.08\n"
            return "GPU-selected, 400.0\n"

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}), \
             mock.patch("run_full_benchmark.subprocess.check_output", side_effect=fake_check_output):
            environment = _collect_environment()

        self.assertEqual(environment["gpu_uuid"], "GPU-selected")
        self.assertEqual(environment["driver"], "580.105.08")


class DifficultyScoreTest(unittest.TestCase):
    def test_score_is_normalized_and_versioned(self):
        config = DifficultyConfig(
            visible_gaussian_scale=100.0,
            overlap_ratio_scale=4.0,
            average_tile_density_scale=8.0,
            depth_complexity_scale=2.0,
        )
        result = calculate_difficulty(
            DifficultyInputs(100, 4.0, 8.0, 2.0), config
        )

        self.assertEqual(result.score, 10.0)
        self.assertEqual(result.formula_id, "geometric_mean_v1")
        self.assertEqual(result.schema_version, 1)
        self.assertEqual(result.to_dict()["inputs"]["visible_gaussian_count"], 100)

    def test_all_factors_affect_score_and_values_are_clamped(self):
        config = DifficultyConfig(
            visible_gaussian_scale=100.0,
            overlap_ratio_scale=4.0,
            average_tile_density_scale=8.0,
            depth_complexity_scale=2.0,
        )
        lower = calculate_difficulty(DifficultyInputs(50, 2.0, 4.0, 1.0), config)
        saturated = calculate_difficulty(
            DifficultyInputs(1000, 40.0, 80.0, 20.0), config
        )

        self.assertAlmostEqual(lower.score, 5.0)
        self.assertEqual(saturated.score, 10.0)

    def test_negative_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate_difficulty(DifficultyInputs(-1, 1.0, 1.0, 1.0))


class QualityAdjustedEfficiencyTest(unittest.TestCase):
    def test_no_regression_preserves_raw_fps(self):
        result = calculate_quality_adjusted_efficiency(
            500.0,
            {"psnr": 30.0, "ssim": 0.95, "lpips": 0.10},
            {"psnr": 30.0, "ssim": 0.95, "lpips": 0.10},
        )

        self.assertEqual(result.quality_factor, 1.0)
        self.assertEqual(result.effective_fps, 500.0)

    def test_configurable_penalties_compound(self):
        config = QualityAdjustmentConfig(
            psnr_drop_weight=1.0,
            ssim_drop_weight=10.0,
            lpips_increase_weight=5.0,
        )
        result = calculate_quality_adjusted_efficiency(
            500.0,
            {"psnr": 29.0, "ssim": 0.94, "lpips": 0.12},
            {"psnr": 30.0, "ssim": 0.95, "lpips": 0.10},
            config,
        )

        expected = math.exp(-(1.0 + 0.1 + 0.1))
        self.assertAlmostEqual(result.quality_factor, expected)
        self.assertAlmostEqual(result.effective_fps, 500.0 * expected)

    def test_missing_quality_is_not_treated_as_equivalent(self):
        result = calculate_quality_adjusted_efficiency(
            500.0,
            {"psnr": None, "ssim": None, "lpips": None},
            {"psnr": 30.0, "ssim": 0.95, "lpips": 0.10},
        )

        self.assertIsNone(result.quality_factor)
        self.assertIsNone(result.effective_fps)


class ParetoAnalysisTest(unittest.TestCase):
    def test_frontier_requires_speed_and_all_quality_metrics(self):
        records = [
            {"renderer": "balanced", "fps": 100, "psnr": 30, "ssim": 0.95, "lpips": 0.10},
            {"renderer": "dominated", "fps": 90, "psnr": 29, "ssim": 0.94, "lpips": 0.12},
            {"renderer": "fast_tradeoff", "fps": 120, "psnr": 28, "ssim": 0.93, "lpips": 0.15},
            {"renderer": "synthetic_only", "fps": 200, "psnr": None, "ssim": None, "lpips": None},
            {"renderer": "synthetic_diagnostic", "benchmark_type": "synthetic_stress", "fps": 300, "psnr": 99, "ssim": 1.0, "lpips": 0.0},
        ]

        result = pareto_analysis(records)

        self.assertEqual(result["frontier"], ["balanced", "fast_tradeoff"])
        self.assertEqual(result["dominated"]["dominated"], ["balanced"])
        self.assertEqual(result["excluded"]["synthetic_only"], "missing_quality_metrics")
        self.assertEqual(result["excluded"]["synthetic_diagnostic"], "synthetic_not_quality_eligible")

    def test_latency_can_be_the_speed_axis(self):
        records = [
            {"renderer": "a", "latency_ms": 2, "psnr": 30, "ssim": 0.95, "lpips": 0.10},
            {"renderer": "b", "latency_ms": 3, "psnr": 30, "ssim": 0.95, "lpips": 0.10},
        ]

        result = pareto_analysis(records, speed_metric="latency_ms")

        self.assertEqual(result["frontier"], ["a"])


class RecommendationTest(unittest.TestCase):
    def test_rules_are_deterministic_and_keep_memory_tradeoff(self):
        records = [
            {
                "renderer": "fast",
                "fps": 200,
                "effective_fps": 120,
                "quality_factor": 0.6,
                "peak_vram_mb": 800,
                "p99_latency_ms": 7,
                "stability_score": 0.8,
            },
            {
                "renderer": "balanced",
                "fps": 150,
                "effective_fps": 145,
                "quality_factor": 0.97,
                "peak_vram_mb": 900,
                "p99_latency_ms": 8,
                "stability_score": 0.9,
            },
            {
                "renderer": "small",
                "fps": 100,
                "effective_fps": 100,
                "quality_factor": 1.0,
                "peak_vram_mb": 400,
                "p99_latency_ms": 12,
                "stability_score": 0.95,
            },
            {
                "renderer": "synthetic",
                "benchmark_type": "synthetic_stress",
                "fps": 50,
                "effective_fps": 1000,
                "quality_factor": 1.0,
                "peak_vram_mb": 300,
                "p99_latency_ms": 2,
                "stability_score": 1.0,
            },
            {
                "renderer": "failed_quality_gate",
                "benchmark_type": "real_scene_speed",
                "quality_status": "failed",
                "fps": 75,
                "effective_fps": 999,
                "quality_factor": 1.0,
                "peak_vram_mb": 1000,
                "p99_latency_ms": 15,
                "stability_score": 0.5,
            },
        ]

        result = build_recommendations(records, pareto_renderers=["fast", "balanced"])
        recommendations = result["recommendations"]

        self.assertEqual(recommendations["best_absolute_speed"]["renderer"], "fast")
        self.assertEqual(recommendations["best_quality_preserving"]["renderer"], "small")
        self.assertEqual(recommendations["best_balanced"]["renderer"], "balanced")
        self.assertEqual(recommendations["best_memory_efficiency"]["renderer"], "synthetic")
        self.assertEqual(recommendations["best_low_latency"]["renderer"], "synthetic")
        self.assertEqual(recommendations["best_pareto_candidate"]["renderer"], "balanced")


class StabilityMetricTest(unittest.TestCase):
    def test_cv_and_stability_are_exported_without_removing_jitter(self):
        metrics = RendererMetrics(renderer_name="test", frame_times_ms=[1.0, 3.0])
        metrics.compute()
        data = metrics.to_dict()

        self.assertEqual(data["std_latency_ms"], 1.0)
        self.assertEqual(data["coefficient_of_variation"], 0.5)
        self.assertAlmostEqual(data["stability_score"], 2.0 / data["p99_latency_ms"])
        self.assertEqual(data["jitter_pct"], 50.0)


class SyntheticStressSuiteTest(unittest.TestCase):
    def test_full_benchmark_can_attach_catalog_difficulty(self):
        catalog = _load_synthetic_difficulty_catalog()
        heavy = catalog[400000]

        self.assertEqual(heavy["difficulty_score"], 5.5334)
        self.assertEqual(heavy["difficulty_formula"], "geometric_mean_v1")
        self.assertEqual(heavy["synthetic_stress_class"], "Heavy Overlap")
        self.assertEqual(heavy["difficulty_inputs"]["visible_gaussian_count"], 400000)

    def test_result_document_preserves_timing_protocol(self):
        document = _build_result_document(
            results={"gsplat_50K": {"mean_fps": 100.0}},
            frames=20,
            warmup_frames=5,
            repeats=3,
            resolution=(1920, 1080),
            environment={"gpu": "test"},
            date="2026-07-15",
        )

        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["protocol"]["warmup_frames"], 5)
        self.assertEqual(document["protocol"]["measured_frames_per_repeat"], 20)
        self.assertEqual(document["protocol"]["repeats"], 3)
        self.assertEqual(document["protocol"]["total_measured_frames"], 60)
        self.assertEqual(document["protocol"]["resolution"], [1920, 1080])

    def test_preload_gsplat_extension_aliases_cached_module(self):
        fake_extension = types.ModuleType("gsplat_cuda")
        fake_inference = types.ModuleType(
            "experimental_gaussian_render_inference_scene_cuda"
        )
        with mock.patch.dict(
            sys.modules,
            {
                "gsplat_cuda": fake_extension,
                "experimental_gaussian_render_inference_scene_cuda": fake_inference,
            },
        ):
            loaded = _preload_gsplat_extension(
                "cached-extension",
                inference_extension_dir="cached-inference",
            )
            self.assertIs(loaded, fake_extension)
            self.assertIs(sys.modules["gsplat.csrc"], fake_extension)
            self.assertIs(
                sys.modules["gsplat.experimental.render.kernels.csrc"],
                fake_inference,
            )


if __name__ == "__main__":
    unittest.main()
