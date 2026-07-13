import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.validate_quality import compute_psnr, compute_ssim
from benchmark_framework import RendererMetrics


class QualityMetricTest(unittest.TestCase):
    def test_identical_images(self):
        image = torch.rand(16, 16, 3)
        self.assertEqual(compute_psnr(image, image), float("inf"))
        self.assertAlmostEqual(compute_ssim(image, image), 1.0, places=5)

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
