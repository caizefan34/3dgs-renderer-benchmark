import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.collect_compression_result import quality_gate
from scripts.run_linux_compression_matrix import build_plan, run


class CompressionResultTest(unittest.TestCase):
    def test_compression_plan_has_reference_and_three_codecs_per_case(self):
        root = Path(__file__).resolve().parents[1]
        plan = build_plan(root)
        self.assertEqual(len(plan), 20)
        self.assertEqual(
            [row["codec"] for row in plan[:4]],
            ["reference-ply", "block-float", "tile-codebook", "spz"],
        )
        self.assertTrue(plan[3]["archive"].endswith("small-garden-1080p.spz"))
        self.assertFalse(plan[3]["archive"].endswith(".spz.spz"))

    def test_compression_plan_can_select_case_and_codecs(self):
        root = Path(__file__).resolve().parents[1]
        run_root = root / "artifacts" / "compression" / "runs" / "qualification"
        plan = build_plan(
            root, {"medium-train-1080p"}, ("reference-ply", "spz"), run_root,
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual(
            [(row["case_id"], row["codec"]) for row in plan],
            [
                ("medium-train-1080p", "reference-ply"),
                ("medium-train-1080p", "spz"),
            ],
        )
        self.assertTrue(plan[0]["run_dir"].startswith(str(run_root)))

    def test_near_lossless_gate_requires_numeric_and_visual_pass(self):
        reference = {"psnr_db": 30.0, "ssim": 0.95, "lpips": 0.10}
        candidate = {"psnr_db": 29.85, "ssim": 0.949, "lpips": 0.103}
        delta, pending = quality_gate(reference, candidate, "pending")
        _, passed = quality_gate(reference, candidate, "pass")
        self.assertAlmostEqual(delta["psnr_db"], -0.15)
        self.assertTrue(pending["numeric_pass"])
        self.assertFalse(pending["overall_pass"])
        self.assertTrue(passed["overall_pass"])

    def test_quality_gate_rejects_boundary_or_worse(self):
        reference = {"psnr_db": 30.0, "ssim": 0.95, "lpips": 0.10}
        candidate = {"psnr_db": 29.8, "ssim": 0.948, "lpips": 0.105}
        _, gate = quality_gate(reference, candidate, "pass")
        self.assertFalse(gate["numeric_pass"])
        self.assertFalse(gate["overall_pass"])

    def test_resume_rejects_different_selection(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session.json"
            session.write_text(json.dumps({
                "benchmark_commit": "a" * 40,
                "selection": {
                    "case_ids": ["small-garden-1080p"],
                    "codecs": ["reference-ply"],
                    "run_root": str(Path(temp_dir) / "runs"),
                },
                "completed": [], "encoded": [],
            }), encoding="utf-8")
            args = argparse.Namespace(
                root=root, python=Path(sys.executable), session=session,
                run_root=Path(temp_dir) / "runs", resume=True,
                case_id=["medium-train-1080p"], codec=["reference-ply"],
                encode_only=False, wait_gpu=0, idle_max_memory_mib=1024.0,
                idle_max_utilization=5.0, idle_samples=1, idle_poll_seconds=0.0,
                report_output=Path(temp_dir) / "report",
            )
            with patch(
                "scripts.run_linux_compression_matrix.subprocess.check_output",
                return_value="a" * 40,
            ), self.assertRaisesRegex(RuntimeError, "selection"):
                run(args)


if __name__ == "__main__":
    unittest.main()
