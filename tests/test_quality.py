import sys
import tempfile
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.validate_quality import (
    LPIPSMetric,
    _ground_truth_manifest,
    compute_psnr,
    compute_ssim,
)
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

    def test_ssim_matches_graphdeco_zero_padding(self):
        prediction = torch.zeros(16, 16, 3)
        reference = prediction.clone()
        prediction[0, 0] = 1.0

        def graphdeco_ssim(first, second):
            coords = torch.arange(11, dtype=first.dtype) - 5
            kernel_1d = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
            kernel_1d /= kernel_1d.sum()
            kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, 11, 11)
            first = first.permute(2, 0, 1).unsqueeze(0)
            second = second.permute(2, 0, 1).unsqueeze(0)
            blur = lambda image: torch.nn.functional.conv2d(
                image, kernel, padding=5, groups=3
            )
            mu_first, mu_second = blur(first), blur(second)
            var_first = blur(first * first) - mu_first * mu_first
            var_second = blur(second * second) - mu_second * mu_second
            covariance = blur(first * second) - mu_first * mu_second
            score = ((2 * mu_first * mu_second + 0.01 ** 2) *
                     (2 * covariance + 0.03 ** 2)) / (
                (mu_first * mu_first + mu_second * mu_second + 0.01 ** 2) *
                (var_first + var_second + 0.03 ** 2)
            )
            return score.mean().item()

        self.assertAlmostEqual(
            compute_ssim(prediction, reference),
            graphdeco_ssim(prediction, reference),
            places=7,
        )

    def test_ground_truth_manifest_hashes_file_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "00001.png"
            image.write_bytes(b"first")
            entries_before, digest_before = _ground_truth_manifest([image])
            image.write_bytes(b"second")
            entries_after, digest_after = _ground_truth_manifest([image])

        self.assertEqual(entries_before[0]["image"], "00001.png")
        self.assertNotEqual(entries_before[0]["sha256"], entries_after[0]["sha256"])
        self.assertNotEqual(digest_before, digest_after)

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
