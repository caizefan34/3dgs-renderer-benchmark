"""Focused tests for the Phase-22 system-lever stack protocol.

Pre-registered exploration matrix exploration_accel22_11s0: 6 methods x 11
scenes x seed 0 = 66 jobs. accel22 revives the sh_fp16 lever on the frozen
accel15 candidate by fixing the HiGS native backward path to upcast fp16 SH
coefficients (and the color gradient buffer) to fp32 before the CUDA kernel,
restoring the input dtype for autograd afterwards. accel15 had dropped the
lever because it crashed every job (RuntimeError: sh_coeffs must be float32);
accel22 adds the missing backward upcast on a NEW audited tree
gsplat-higs-accel22 (patch patches/higs-accel22.patch).

In-matrix methods: gsplat (official 30k control), gsplat_30k_fused (accel15
base), gsplat_30k_fused_prune10_rclip05 (frozen accel15 candidate), plus
system levers stacked on the candidate: +shfp16 (fp16 SH forward/backward),
+dens600 (Faster-GS densify-every-600), +shfp16_dens600. All levers keep the
30k full-res schedule and densification topology; shfp16 is pure precision
(system) and dens600 changes only the densification cadence.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_ablation_protocol import (  # noqa: E402
    validate_ablation_protocol,
)

ACCEL22_MATRIX = "exploration_accel22_11s0"
ACCEL22_METHODS = [
    "gsplat",
    "gsplat_30k_fused",
    "gsplat_30k_fused_prune10_rclip05",
    "gsplat_30k_fused_prune10_rclip05_shfp16",
    "gsplat_30k_fused_prune10_rclip05_dens600",
    "gsplat_30k_fused_prune10_rclip05_shfp16_dens600",
]
PATCH_SHA256 = "27197f5bff97b6ee2cb0ddc61a9a7426412fddb530cb15bdb78204c400add62a"
TRAINER_SHA256 = "12bb76d2a7d4f9640badfa2c58f93f262572b3cd74a7b1d769bf169248c8a7f5"
STATE_SHA256 = "ede0dba7825cb6aeb769bdff571ec3bc91edc78859923ab9f0279a152555afe8"


class HigsAccel22ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel22-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel22_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 66)

    def test_accel22_matrix_six_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL22_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL22_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["jobs"], 66)

    def test_accel22_hashes_pinned(self):
        for method in ACCEL22_METHODS[1:]:
            spec = self.protocol["methods"][method]
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            self.assertEqual(spec["source_state_sha256"], STATE_SHA256)
            self.assertEqual(spec["patches"], ["patches/higs-accel22.patch"])

    def test_accel22_levers_wired(self):
        base = self.protocol["methods"]["gsplat_30k_fused_prune10_rclip05"]
        self.assertEqual(base["algorithm"]["trainer_cfg"]["higs_prune_opacity_thresh"], 0.1)
        self.assertEqual(base["algorithm"]["trainer_cfg"]["higs_radius_clip"], 0.5)
        sh = self.protocol["methods"]["gsplat_30k_fused_prune10_rclip05_shfp16"]
        self.assertTrue(sh["algorithm"]["trainer_cfg"]["sh_fp16"])
        de = self.protocol["methods"]["gsplat_30k_fused_prune10_rclip05_dens600"]
        self.assertEqual(de["algorithm"]["trainer_cfg"]["higs_densify_every"], 600)
        both = self.protocol["methods"]["gsplat_30k_fused_prune10_rclip05_shfp16_dens600"]
        self.assertTrue(both["algorithm"]["trainer_cfg"]["sh_fp16"])
        self.assertEqual(both["algorithm"]["trainer_cfg"]["higs_densify_every"], 600)

    def test_accel22_official_control_in_matrix(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL22_MATRIX
        )
        self.assertIn("gsplat", matrix["methods"])
        self.assertIsNone(self.protocol["methods"]["gsplat"].get("algorithm"))


if __name__ == "__main__":
    unittest.main()
