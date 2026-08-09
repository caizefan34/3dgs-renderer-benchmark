"""Focused tests for the Phase-15 fused-SSIM + post-refine lever protocol.

Pre-registered exploration matrix exploration_accel15_11s0: 6 methods x 11
scenes x seed 0 = 66 jobs. accel15 runs only in an environment where the
fused-ssim package is installed (gsplat.losses.ssim_loss auto-switches to the
fused CUDA kernel, ~10x faster and numerically near-identical: max grad diff
~9e-7). The in-matrix official gsplat control (gsplat) and the HiGS-tree fused
baseline (gsplat_30k_fused) give same-window speed pairing; candidates stack
post-refine opacity pruning (thresh 0.10), SH-fp16 (sh_fp16), and radius clip
(higs_radius_clip, active only after higs_radius_clip_start_step=15000 so
densification topology is untouched), plus optional accum2. All patched
methods use a NEW audited source tree gsplat-higs-accel15 (patch
patches/higs-accel15.patch) which adds the radius_clip plumbing at the three
training render call sites.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_ablation_protocol import (  # noqa: E402
    build_ablation_experiment_plan,
    validate_ablation_protocol,
)

ACCEL15_MATRIX = "exploration_accel15_11s0"
ACCEL15_METHODS = [
    "gsplat",
    "gsplat_30k_fused",
    "gsplat_30k_fused_prune10",
    "gsplat_30k_fused_prune10_shfp16",
    "gsplat_30k_fused_prune10_shfp16_rclip05",
    "gsplat_30k_fused_prune10_shfp16_rclip05_accum2",
]
PATCH_SHA256 = "4e5ae587f53a302f61e3f7c3b95f2e5cf47bea9a5b41c1c16ec7eb768b57016f"
TRAINER_SHA256 = "61335b1d7dad3ae192ca0ca5c0dca199af4b9a68290c7aea38cadc4be3d7356d"
SOURCE_STATE_SHA256 = "9b00469d4c8748eb965423e7d348edaaebd70997c636de39da59d926629a172a"


class HigsAccel15ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel15-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel15_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["executable_jobs"], 66)
        self.assertEqual(report["planned_jobs"], 66)

    def test_accel15_matrix_shape(self):
        matrix = next(m for m in self.protocol["matrices"] if m["id"] == ACCEL15_MATRIX)
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL15_METHODS)
        self.assertEqual(matrix["seeds"], [0])

    def test_accel15_hashes_pinned(self):
        for method in ACCEL15_METHODS[1:]:
            spec = self.protocol["methods"][method]
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            self.assertEqual(spec["source_state_sha256"], SOURCE_STATE_SHA256)
            self.assertEqual(spec["patches"], ["patches/higs-accel15.patch"])

    def test_accel15_levers_wired(self):
        cfg = self.protocol["methods"]["gsplat_30k_fused_prune10_shfp16_rclip05_accum2"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_prune_opacity_thresh"], 0.10)
        self.assertEqual(cfg["higs_prune_start_step"], 15000)
        self.assertEqual(cfg["sh_fp16"], True)
        self.assertEqual(cfg["higs_radius_clip"], 0.5)
        self.assertEqual(cfg["higs_radius_clip_start_step"], 15000)
        self.assertEqual(cfg["higs_accum_steps"], 2)
        self.assertEqual(cfg["higs_ssim_scale"], 1.0)

    def test_accel15_official_control_in_matrix(self):
        matrix = next(m for m in self.protocol["matrices"] if m["id"] == ACCEL15_MATRIX)
        self.assertIn("gsplat", matrix["methods"])
        spec = self.protocol["methods"]["gsplat"]
        self.assertIsNone(spec.get("algorithm"))


if __name__ == "__main__":
    unittest.main()
