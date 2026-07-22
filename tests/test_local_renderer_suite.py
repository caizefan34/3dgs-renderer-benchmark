import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renderers import get_renderer_class, list_available, list_renderers
from scripts.run_local_renderer_suite import _run_command, _speed_command, run_suite


class RendererStubTest(unittest.TestCase):
    def test_missing_renderer_stubs_keep_interface(self):
        for name in ("local_gs", "gemm_gs", "stopthepop"):
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
    def test_external_camera_command_skips_synthetic_facing_heuristic(self):
        command = _speed_command(
            "original_3dgs",
            Path("scene.ply"),
            Path("eval_cameras.json"),
            Path("output"),
            100,
            30,
            5,
            "real_scene_speed",
            1920,
            1080,
        )

        self.assertIn("--allow-backfacing-cameras", command)

    def test_failed_command_retains_full_output_and_gpu_state(self):
        long_error = "stack-line\n" * 30
        command = [sys.executable, "-c", f"import sys; print('full stdout'); sys.stderr.write({long_error!r}); raise SystemExit(3)"]
        with mock.patch(
            "scripts.run_local_renderer_suite._capture_gpu_state",
            return_value={"gpu": "snapshot"},
        ):
            result = _run_command(
                command,
                Path.cwd(),
                {"renderer_config_id": "test", "scene": "scene.ply", "phase": "speed"},
            )

        evidence = result["failure_evidence"]
        self.assertEqual(result["returncode"], 3)
        self.assertEqual(evidence["stdout"].strip(), "full stdout")
        self.assertEqual(evidence["stderr"], long_error)
        self.assertEqual(evidence["stack_trace"], long_error)
        self.assertEqual(evidence["gpu_memory_state"], {"gpu": "snapshot"})
        self.assertEqual(evidence["renderer_config_id"], "test")
        self.assertIn("started_at_utc", evidence)

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
