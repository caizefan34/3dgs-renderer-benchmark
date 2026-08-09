"""Focused tests for the Phase-19 prune-threshold sensitivity grid protocol.

Pre-registered exploration matrix exploration_accel19_11s0: 6 methods x 11
scenes x seed 0 = 66 jobs. accel19 completes the opacity-prune lever grid of
accel15 (prune10 -> prune05/prune03/no-extra-prune) on the same audited tree
gsplat-higs-accel15 + patches/higs-accel15.patch (no new code). Rationale:
accel15 in-matrix pairing showed prune10_rclip05 holds SSIM/LPIPS and gives
1.18x speed while PSNR drifts to -0.054 (CI lo -0.13); gentler thresholds
trade a little of the large speed margin for PSNR headroom.
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

ACCEL19_MATRIX = "exploration_accel19_11s0"
ACCEL19_METHODS = [
    "gsplat",
    "gsplat_30k_fused",
    "gsplat_30k_fused_prune05",
    "gsplat_30k_fused_prune05_rclip05",
    "gsplat_30k_fused_prune03_rclip05",
    "gsplat_30k_fused_rclip05",
]
PATCH_SHA256 = "04f950778f617295cf28611487a017d081dd4dcc5f056560d60c598a5eafa5e7"
TRAINER_SHA256 = "12bb76d2a7d4f9640badfa2c58f93f262572b3cd74a7b1d769bf169248c8a7f5"
SOURCE_STATE_SHA256 = "87880d7af2619ae7eb406384b88b7c2b1987073be7de667d2f79f198376bbb7f"


class HigsAccel19ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel19-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel19_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["executable_jobs"], 66)
        self.assertEqual(report["planned_jobs"], 66)

    def test_accel19_matrix_shape(self):
        matrix = next(m for m in self.protocol["matrices"] if m["id"] == ACCEL19_MATRIX)
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL19_METHODS)
        self.assertEqual(matrix["seeds"], [0])

    def test_accel19_hashes_pinned(self):
        for method in ACCEL19_METHODS[2:]:
            spec = self.protocol["methods"][method]
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            self.assertEqual(spec["source_state_sha256"], SOURCE_STATE_SHA256)
            self.assertEqual(spec["patches"], ["patches/higs-accel15.patch"])

    def test_accel19_prune_grid_wired(self):
        m05 = self.protocol["methods"]["gsplat_30k_fused_prune05"]["algorithm"]["trainer_cfg"]
        self.assertEqual(m05["higs_prune_opacity_thresh"], 0.05)
        self.assertEqual(m05["higs_radius_clip"], 0.0)
        r = self.protocol["methods"]["gsplat_30k_fused_prune05_rclip05"]["algorithm"]["trainer_cfg"]
        self.assertEqual(r["higs_prune_opacity_thresh"], 0.05)
        self.assertEqual(r["higs_radius_clip"], 0.5)
        self.assertEqual(r["higs_ssim_scale"], 1.0)
        self.assertEqual(r["higs_train_res_scale"], 1.0)
        self.assertFalse(r["higs_accum_steps"] > 1)

    def test_accel19_rclip_only_keeps_base_prune(self):
        r = self.protocol["methods"]["gsplat_30k_fused_rclip05"]["algorithm"]["trainer_cfg"]
        self.assertEqual(r["higs_radius_clip"], 0.5)
        self.assertEqual(r["higs_prune_opacity_thresh"], 0.02)

    def test_accel19_plan_jobs(self):
        plan = [j for j in build_ablation_experiment_plan(self.protocol) if j["matrix"] == ACCEL19_MATRIX]
        self.assertEqual(len(plan), 66)
        for job in plan:
            self.assertIn(job["method"], ACCEL19_METHODS)
            self.assertEqual(job["seed"], 0)


if __name__ == "__main__":
    unittest.main()
