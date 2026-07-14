import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.validate_quality import LPIPSMetric, compute_psnr, compute_ssim
from benchmark_framework import RendererMetrics


class QualityMetricTest(unittest.TestCase):
    def test_identical_images(self):
        image = torch.rand(16, 16, 3)
        self.assertEqual(compute_psnr(image, image), float("inf"))
        self.assertAlmostEqual(compute_ssim(image, image), 1.0, places=5)

    def test_known_psnr(self):
        reference = torch.zeros(16, 16, 3)
        prediction = torch.full_like(reference, 0.1)
        self.assertAlmostEqual(compute_psnr(prediction, reference), 20.0, places=5)

    def test_lpips_receives_nchw_minus_one_to_one(self):
        class FakeModel(torch.nn.Module):
            def forward(self, prediction, reference):
                self.prediction = prediction
                self.reference = reference
                return torch.tensor([0.125])

        model = FakeModel()
        metric = LPIPSMetric(device="cpu", model=model)
        prediction = torch.ones(16, 16, 3)
        reference = torch.zeros(16, 16, 3)
        self.assertEqual(metric(prediction, reference), 0.125)
        self.assertEqual(tuple(model.prediction.shape), (1, 3, 16, 16))
        self.assertEqual(model.prediction.min().item(), 1.0)
        self.assertEqual(model.reference.max().item(), -1.0)

    def test_wall_metrics_are_computed_separately(self):
        metrics = RendererMetrics(
            renderer_name="test",
            frame_times_ms=[1.0, 3.0],
            wall_frame_times_ms=[2.0, 4.0],
        )
        metrics.compute()
        self.assertEqual(metrics.mean_latency_ms, 2.0)
        self.assertEqual(metrics.mean_wall_latency_ms, 3.0)
        self.assertEqual(metrics.wall_fps, 333.3)


if __name__ == "__main__":
    unittest.main()
