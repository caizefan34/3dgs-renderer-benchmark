"""Focused tests for the Phase-15 fused-SSIM + post-refine lever protocol.

Pre-registered exploration matrix exploration_accel15_11s0: 6 methods x 11
scenes x seed 0 = 66 jobs. accel15 runs only in an environment where the
fused-ssim package is installed (gsplat.losses.ssim_loss auto-switches to the
fused CUDA kernel, ~10x faster and numerically near-identical: max grad diff
~9e-7). The in-matrix official gsplat control (gsplat) and the HiGS-tree fused
baseline (gsplat_30k_fused) give same-window speed pairing; candidates stack
post-refine opacity pruning (thresh 0.10) and radius clip
(higs_radius_clip, active only after higs_radius_clip_start_step=15000 so
densification topology is untouched), plus optional accum2.

NOTE: the sh_fp16 lever was dropped after smoke tests
proved the HiGS CUDA backend requires float32 SH coefficients in both the
forward packer and the native backward kernel (RuntimeError: sh_coeffs must
be float32); keeping it would crash every job. accum2 standalone is added
for attribution. All patched
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
    "gsplat_30k_fused_prune10_rclip05",
    "gsplat_30k_fused_prune10_accum2",
    "gsplat_30k_fused_prune10_rclip05_accum2",
]
PATCH_SHA256 = "04f950778f617295cf28611487a017d081dd4dcc5f056560d60c598a5eafa5e7"
TRAINER_SHA256 = "12bb76d2a7d4f9640badfa2c58f93f262572b3cd74a7b1d769bf169248c8a7f5"
SOURCE_STATE_SHA256 = "87880d7af2619ae7eb406384b88b7c2b1987073be7de667d2f79f198376bbb7f"


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
        cfg = self.protocol["methods"]["gsplat_30k_fused_prune10_rclip05_accum2"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_prune_opacity_thresh"], 0.10)
        self.assertEqual(cfg["higs_prune_start_step"], 15000)
        self.assertEqual(cfg["sh_fp16"], False)
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
