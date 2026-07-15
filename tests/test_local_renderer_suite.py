import argparse
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renderers import get_renderer_class, list_available, list_renderers
from scripts.run_local_renderer_suite import run_suite


class RendererStubTest(unittest.TestCase):
    def test_missing_renderer_stubs_keep_interface(self):
        for name in ("flashgs", "local_gs", "gemm_gs", "stopthepop"):
            self.assertIn(name, list_renderers())
            cls = get_renderer_class(name)
            self.assertIsNotNone(cls)
            renderer = cls(device="cuda")
            self.assertFalse(renderer.is_available())
            with self.assertRaises(NotImplementedError):
                renderer.prepare_scene({})

    def test_missing_renderer_stubs_are_not_available(self):
        available = set(list_available())

        self.assertNotIn("flashgs", available)
        self.assertNotIn("gemm_gs", available)


class LocalRendererSuiteTest(unittest.TestCase):
    def test_suite_reports_missing_inputs_without_faking_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                scene=None,
                cameras=None,
                ground_truth_dir=None,
                renderers=["flashgs"],
                output_dir=Path(temp_dir),
                frames=1,
                warmup=0,
                repeats=1,
                device="cuda",
                benchmark_type="real_scene_speed",
            )

            report = run_suite(args)

        self.assertEqual(report["speed_skip_reason"], "missing_scene")
        self.assertEqual(
            report["quality_skip_reason"],
            "missing_scene_cameras_or_ground_truth",
        )
        self.assertEqual(report["availability"][0]["renderer"], "flashgs")
        self.assertFalse(report["availability"][0]["available"])
        self.assertEqual(report["speed_runs"], [])
        self.assertEqual(report["quality_runs"], [])


if __name__ == "__main__":
    unittest.main()

