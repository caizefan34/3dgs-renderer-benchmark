"""Focused tests for the Phase-20 low-resolution window-schedule protocol.

Pre-registered exploration matrix exploration_accel20_11s0: 6 methods x 11
scenes x seed 0 = 66 jobs. accel20 tests the Phase-3 topology-preserving
window schedule (full-res densification 0-15k, cached 0.7x training
[15k,22k), full-res refinement 22k-30k) stacked on the fused-SSIM full-res
baseline, using the same audited accel15 tree (no new code). The frozen
accel15 candidate gsplat_30k_fused_prune10_rclip05 is included in-matrix so
the window's marginal speed/quality tradeoff is directly measurable.
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

ACCEL20_MATRIX = "exploration_accel20_11s0"
ACCEL20_METHODS = [
    "gsplat",
    "gsplat_30k_fused",
    "gsplat_30k_fused_win07",
    "gsplat_30k_fused_win07_prune10",
    "gsplat_30k_fused_win07_prune10_rclip05",
    "gsplat_30k_fused_prune10_rclip05",
]
PATCH_SHA256 = "04f950778f617295cf28611487a017d081dd4dcc5f056560d60c598a5eafa5e7"
TRAINER_SHA256 = "12bb76d2a7d4f9640badfa2c58f93f262572b3cd74a7b1d769bf169248c8a7f5"
SOURCE_STATE_SHA256 = "87880d7af2619ae7eb406384b88b7c2b1987073be7de667d2f79f198376bbb7f"
WINDOW = {"higs_train_res_scale": 0.7, "higs_lowres_start_step": 15000,
          "higs_lowres_end_step": 22000}


class HigsAccel20ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel20-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel20_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["executable_jobs"], 66)
        self.assertEqual(report["planned_jobs"], 66)

    def test_accel20_matrix_shape(self):
        matrix = next(m for m in self.protocol["matrices"] if m["id"] == ACCEL20_MATRIX)
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL20_METHODS)
        self.assertEqual(matrix["seeds"], [0])

    def test_accel20_hashes_pinned(self):
        for method in ACCEL20_METHODS[1:]:
            spec = self.protocol["methods"][method]
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            self.assertEqual(spec["source_state_sha256"], SOURCE_STATE_SHA256)
            self.assertEqual(spec["patches"], ["patches/higs-accel15.patch"])

    def test_accel20_window_wired(self):
        w = self.protocol["methods"]["gsplat_30k_fused_win07"]["algorithm"]["trainer_cfg"]
        self.assertEqual(w["higs_train_res_scale"], 0.7)
        self.assertEqual(w["higs_lowres_start_step"], 15000)
        self.assertEqual(w["higs_lowres_end_step"], 22000)
        self.assertTrue(w["higs_res_cache"])
        self.assertEqual(w["higs_densify_anchor_fullres"], False)
        # window is full-res outside [15k,22k) so densification 0-15k untouched
        self.assertEqual(w["higs_full_res_step"], 0)

    def test_accel20_window_stack(self):
        for method, prune, rclip in [
            ("gsplat_30k_fused_win07", 0.02, 0.0),
            ("gsplat_30k_fused_win07_prune10", 0.10, 0.0),
            ("gsplat_30k_fused_win07_prune10_rclip05", 0.10, 0.5),
            ("gsplat_30k_fused_win07_rclip05", 0.02, 0.5),
        ]:
            tc = self.protocol["methods"][method]["algorithm"]["trainer_cfg"]
            self.assertEqual(tc["higs_train_res_scale"], 0.7)
            self.assertEqual(tc["higs_prune_opacity_thresh"], prune)
            self.assertEqual(tc["higs_radius_clip"], rclip)
            self.assertEqual(tc["higs_ssim_scale"], 1.0)
            self.assertFalse(tc["higs_accum_steps"] > 1)

    def test_accel20_frozen_candidate_control_present(self):
        c = self.protocol["methods"]["gsplat_30k_fused_prune10_rclip05"]["algorithm"]["trainer_cfg"]
        self.assertEqual(c["higs_train_res_scale"], 1.0)
        self.assertEqual(c["higs_lowres_start_step"], 0)
        self.assertEqual(c["higs_lowres_end_step"], 0)
        self.assertEqual(c["higs_prune_opacity_thresh"], 0.10)
        self.assertEqual(c["higs_radius_clip"], 0.5)

    def test_accel20_plan_jobs(self):
        plan = [j for j in build_ablation_experiment_plan(self.protocol)
                if j["matrix"] == ACCEL20_MATRIX]
        self.assertEqual(len(plan), 66)
        for job in plan:
            self.assertIn(job["method"], ACCEL20_METHODS)
            self.assertEqual(job["seed"], 0)


if __name__ == "__main__":
    unittest.main()
