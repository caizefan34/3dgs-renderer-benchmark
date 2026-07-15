import inspect
import math
import os
import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters.base import RendererAdapter
from adapters.quality import QualityThresholds, evaluate_quality_gate


class _ContractAdapter(RendererAdapter):
    def __init__(self, output_factory=None):
        super().__init__(device="cpu")
        self.calls = 0
        self.output_factory = output_factory

    def load_checkpoint(self, path: Path):
        return {"path": path}

    def render(self, camera_params: dict):
        self.calls += 1
        height, width = camera_params["height"], camera_params["width"]
        if self.output_factory:
            return self.output_factory(height, width)
        return {
            "rgb": torch.zeros(height, width, 3, dtype=torch.float32),
            "depth": torch.zeros(height, width, 1, dtype=torch.float32),
            "alpha": torch.ones(height, width, 1, dtype=torch.float32),
        }

    def get_memory_usage_mb(self) -> float:
        return 0.0

    def get_device_properties(self) -> dict:
        return {"device": "cpu", "cuda_available": False}


class RendererAdapterContractTest(unittest.TestCase):
    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_abc_rejects_incomplete_adapter(self):
        class IncompleteAdapter(RendererAdapter):
            pass

        with self.assertRaises(TypeError):
            IncompleteAdapter()

    def test_registered_renderers_conform_to_abc(self):
        from renderers import get_renderer_class, list_renderers

        for name in list_renderers():
            adapter_class = get_renderer_class(name)
            self.assertTrue(issubclass(adapter_class, RendererAdapter), name)
            self.assertFalse(inspect.isabstract(adapter_class), name)

    def test_output_shapes_and_dtypes_are_strict(self):
        output = _ContractAdapter().render_checked({"height": 12, "width": 20})
        self.assertEqual(output["rgb"].shape, (12, 20, 3))
        self.assertEqual(output["depth"].shape, (12, 20, 1))
        self.assertEqual(output["alpha"].shape, (12, 20, 1))
        self.assertTrue(all(value.dtype == torch.float32 for value in output.values()))

    def test_rejects_invalid_shape_and_dtype(self):
        def invalid(height, width):
            return {
                "rgb": torch.zeros(3, height, width, dtype=torch.float32),
                "depth": torch.zeros(height, width, 1, dtype=torch.float32),
                "alpha": torch.ones(height, width, 1, dtype=torch.float64),
            }

        with self.assertRaisesRegex(ValueError, "rgb.*shape"):
            _ContractAdapter(invalid).render_checked({"height": 12, "width": 20})

        def invalid_dtype(height, width):
            return {
                "rgb": torch.zeros(height, width, 3, dtype=torch.float64),
                "depth": torch.zeros(height, width, 1, dtype=torch.float32),
                "alpha": torch.ones(height, width, 1, dtype=torch.float32),
            }

        with self.assertRaisesRegex(TypeError, "rgb.*dtype"):
            _ContractAdapter(invalid_dtype).render_checked(
                {"height": 12, "width": 20}
            )

    def test_rejects_invalid_resolution(self):
        adapter = _ContractAdapter()
        for params in ({"height": 0, "width": 20}, {"height": 12.5, "width": 20}):
            with self.assertRaisesRegex(ValueError, "height and width"):
                adapter.render_checked(params)

    def test_variable_resolution(self):
        adapter = _ContractAdapter()
        for height, width in ((16, 16), (45, 80), (1080, 1920)):
            output = adapter.render_checked({"height": height, "width": width})
            self.assertEqual(output["rgb"].shape, (height, width, 3))

    def test_warmup_and_repeats_produce_deterministic_timing_array(self):
        timestamps = iter((0.000, 0.001, 0.010, 0.012, 0.020, 0.023))
        adapter = _ContractAdapter()
        timings = adapter.benchmark(
            {"height": 4, "width": 8},
            warmup_iterations=2,
            repeats=3,
            timer=lambda: next(timestamps),
        )
        self.assertEqual(adapter.calls, 5)
        self.assertEqual(timings, [1.0, 2.0, 3.0])


class QualityGateTest(unittest.TestCase):
    def test_rejects_each_below_quality_render(self):
        thresholds = QualityThresholds(
            min_psnr_db=30.0,
            min_ssim=0.95,
            max_lpips=0.10,
        )
        self.assertTrue(
            evaluate_quality_gate(30.0, 0.95, 0.10, thresholds).passed
        )
        for metrics in ((29.99, 0.95, 0.10), (30.0, 0.949, 0.10), (30.0, 0.95, 0.101)):
            self.assertFalse(evaluate_quality_gate(*metrics, thresholds).passed)


@unittest.skipUnless(
    os.environ.get("RUN_RENDERER_REGRESSION") == "1" and torch.cuda.is_available(),
    "set RUN_RENDERER_REGRESSION=1 on a configured CUDA host",
)
class RendererReproducibilityRegressionTest(unittest.TestCase):
    """Opt-in cross-backend test; optional CUDA extensions are not CI dependencies."""

    def tearDown(self):
        torch.cuda.empty_cache()

    def test_synthetic_scene_is_reproducible_across_all_available_adapters(self):
        from benchmark_framework import generate_cameras
        from renderers import get_renderer, list_available

        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        count = 128
        scene = {
            "xyz": torch.randn(count, 3, device="cuda") * 0.2,
            "opacity": torch.zeros(count, device="cuda"),
            "scales": torch.full((count, 3), -3.0, device="cuda"),
            "rotations": torch.nn.functional.normalize(
                torch.randn(count, 4, device="cuda"), dim=-1
            ),
            "shs": torch.zeros(count, 16, 3, device="cuda"),
            "sh_degree": 3,
            "num_points": count,
        }
        camera = generate_cameras(
            1, image_width=64, image_height=64, device="cuda"
        )[0]
        reference = torch.zeros(64, 64, 3, device="cuda")
        tested = []
        for name in list_available():
            renderer = get_renderer(name)
            prepared = renderer.prepare_scene(scene)
            with torch.inference_mode():
                first = renderer.render(prepared, camera)
                second = renderer.render(prepared, camera)
                torch.cuda.synchronize()
            mse_first = torch.mean((first.float() - reference) ** 2).item()
            mse_second = torch.mean((second.float() - reference) ** 2).item()
            psnr_first = -10.0 * math.log10(max(mse_first, 1e-12))
            psnr_second = -10.0 * math.log10(max(mse_second, 1e-12))
            self.assertLess(abs(psnr_first - psnr_second), 0.01, name)
            tested.append(name)
            del renderer, prepared, first, second
            torch.cuda.empty_cache()
        self.assertTrue(tested, "no renderer backend was available")


if __name__ == "__main__":
    unittest.main()
