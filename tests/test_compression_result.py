import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.collect_compression_result import quality_gate
from scripts.run_linux_compression_matrix import build_plan


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
        plan = build_plan(
            root, {"medium-train-1080p"}, ("reference-ply", "spz"),
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual(
            [(row["case_id"], row["codec"]) for row in plan],
            [
                ("medium-train-1080p", "reference-ply"),
                ("medium-train-1080p", "spz"),
            ],
        )

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


if __name__ == "__main__":
    unittest.main()
