import sys
import json
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_framework import RendererMetrics, ResultsManager


class ResultsExportTest(unittest.TestCase):
    def test_analysis_artifacts_are_additive(self):
        reference = RendererMetrics(
            renderer_name="reference",
            frame_times_ms=[10.0],
            psnr=30.0,
            ssim=0.95,
            lpips=0.10,
            peak_vram_mb=500,
            benchmark_type="real_scene_speed",
        )
        candidate = RendererMetrics(
            renderer_name="candidate",
            frame_times_ms=[5.0],
            psnr=29.9,
            ssim=0.949,
            lpips=0.101,
            peak_vram_mb=600,
            benchmark_type="real_scene_speed",
        )
        reference.compute()
        candidate.compute()
        manager = ResultsManager()
        manager.add_result("reference", reference)
        manager.add_result("candidate", candidate)
        manager.apply_quality_adjustment("reference")

        with tempfile.TemporaryDirectory() as temp_dir:
            manager.export_analysis(temp_dir)
            pareto = json.loads((Path(temp_dir) / "pareto_frontier.json").read_text())
            recommendations = json.loads(
                (Path(temp_dir) / "recommendations.json").read_text()
            )

        self.assertEqual(set(pareto["frontier"]), {"reference", "candidate"})
        self.assertEqual(
            recommendations["recommendations"]["best_absolute_speed"]["renderer"],
            "candidate",
        )
        self.assertIsNotNone(candidate.to_dict()["effective_fps"])

    def test_unmeasured_quality_is_null_and_na(self):
        metrics = RendererMetrics(renderer_name="test", frame_times_ms=[2.0])
        metrics.compute()
        self.assertIsNone(metrics.to_dict()["psnr"])
        manager = ResultsManager()
        manager.add_result("test", metrics)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.md"
            manager.export_markdown(str(report))
            markdown = report.read_text(encoding="utf-8")
        self.assertIn("PSNR vs GT", markdown)
        self.assertIn("N/A", markdown)

    def test_html_metadata_uses_frame_count(self):
        metrics = RendererMetrics(
            renderer_name="test",
            frame_times_ms=[2.0, 4.0],
            num_frames=73,
            image_width=640,
            image_height=360,
            gpu_name="Test GPU",
        )
        metrics.compute()
        manager = ResultsManager()
        manager.add_result("test", metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.html"
            manager.export_html(str(report))
            html = report.read_text(encoding="utf-8")

        self.assertIn("<b>Frames</b>: 73", html)
        self.assertIn("<b>Resolution</b>: 640x360", html)
        self.assertIn("<td>1</td>", html)
        self.assertNotIn("<b>Frames</b>: 3.0", html)

    def test_empty_html_report_is_supported(self):
        manager = ResultsManager()
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.html"
            manager.export_html(str(report))
            html = report.read_text(encoding="utf-8")

        self.assertIn("<b>GPU</b>: N/A", html)
        self.assertIn("<b>Resolution</b>: N/A", html)
        self.assertIn("<b>Frames</b>: N/A", html)

    def test_jitter_has_percentage_name(self):
        metrics = RendererMetrics(renderer_name="test", frame_times_ms=[1.0, 3.0])
        metrics.compute()

        self.assertEqual(metrics.jitter_pct, 50.0)
        self.assertEqual(metrics.to_dict()["jitter_pct"], 50.0)

    def test_timing_protocol_fields_are_serialized(self):
        metrics = RendererMetrics(
            renderer_name="test",
            frame_times_ms=[1.0] * 6,
            warmup_frames=2,
            measured_frames_per_repeat=3,
            repeats=2,
            num_frames=6,
        )
        metrics.compute()

        data = metrics.to_dict()
        self.assertEqual(data["warmup_frames"], 2)
        self.assertEqual(data["measured_frames_per_repeat"], 3)
        self.assertEqual(data["repeats"], 2)


if __name__ == "__main__":
    unittest.main()
