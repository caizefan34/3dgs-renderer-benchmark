"""Focused tests for the Phase-23 deep gradient-accumulation protocol.

Pre-registered exploration matrix exploration_accel23_11s0: 6 methods x 11
scenes x seed 0 = 66 jobs. accel23 tests Faster-GS / Taming-3DGS style deep
gradient accumulation (accum 16) stacked on the frozen accel15 candidate
gsplat_30k_fused_prune10_rclip05. The trainer accumulates per-step gradients
and runs one Adam step every `higs_accum_steps` iterations WITHOUT gradient
scaling, so Adam's per-update step scales as sqrt(accum); to keep the
per-iteration movement matched to baseline the compensated variants scale all
parameter learning rates by sqrt(accum)=4 (means_lr/scales_lr/opacities_lr/
quats_lr/sh0_lr/shN_lr). The means LR scheduler steps every iteration so the
compensated LR trajectory tracks 4x the baseline LR at every iteration.
Naive accum16 (no LR compensation) is the in-matrix negative control; accel7/8
already showed uncompensated accum8 collapses quality (PSNR -0.40/-0.58 dB).

Methods: gsplat / gsplat_30k_fused / frozen candidate in-matrix /
+accum16 (naive) / +accum16_lr4 (compensated) / +accum16_lr4_dens600.
Reuses the audited gsplat-higs-accel15 tree + patch 04f95077 (zero new code;
LR overrides flow through trainer_cfg_kwargs to the trainer dataclass).
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

ACCEL23_MATRIX = "exploration_accel23_11s0"
ACCEL23_METHODS = [
    "gsplat",
    "gsplat_30k_fused",
    "gsplat_30k_fused_prune10_rclip05",
    "gsplat_30k_fused_prune10_rclip05_accum16",
    "gsplat_30k_fused_prune10_rclip05_accum16_lr4",
    "gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600",
]
PATCH_SHA256 = "04f950778f617295cf28611487a017d081dd4dcc5f056560d60c598a5eafa5e7"
TRAINER_SHA256 = "12bb76d2a7d4f9640badfa2c58f93f262572b3cd74a7b1d769bf169248c8a7f5"
SOURCE_STATE_SHA256 = "87880d7af2619ae7eb406384b88b7c2b1987073be7de667d2f79f198376bbb7f"
BASE_LRS = {
    "means_lr": 1.6e-4,
    "scales_lr": 5e-3,
    "opacities_lr": 5e-2,
    "quats_lr": 1e-3,
    "sh0_lr": 2.5e-3,
    "shN_lr": 2.5e-3 / 20,
}


class HigsAccel23ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel23-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel23_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["executable_jobs"], 66)
        self.assertEqual(report["planned_jobs"], 66)

    def test_accel23_matrix_shape(self):
        matrix = next(m for m in self.protocol["matrices"] if m["id"] == ACCEL23_MATRIX)
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL23_METHODS)
        self.assertEqual(matrix["seeds"], [0])

    def test_accel23_hashes_pinned(self):
        for method in ACCEL23_METHODS[1:]:
            spec = self.protocol["methods"][method]
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            self.assertEqual(spec["source_state_sha256"], SOURCE_STATE_SHA256)
            self.assertEqual(spec["patches"], ["patches/higs-accel15.patch"])

    def test_accel23_accum_wired(self):
        for method, accum, dens, lr_scale in [
            ("gsplat_30k_fused_prune10_rclip05_accum16", 16, 0, 1.0),
            ("gsplat_30k_fused_prune10_rclip05_accum16_lr4", 16, 0, 4.0),
            ("gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600", 16, 600, 4.0),
        ]:
            tc = self.protocol["methods"][method]["algorithm"]["trainer_cfg"]
            self.assertEqual(tc["higs_accum_steps"], accum)
            self.assertEqual(tc["higs_densify_every"], dens)
            self.assertEqual(tc["higs_prune_opacity_thresh"], 0.10)
            self.assertEqual(tc["higs_radius_clip"], 0.5)
            self.assertEqual(tc["higs_ssim_scale"], 1.0)
            self.assertEqual(tc["higs_train_res_scale"], 1.0)
            for k, v in BASE_LRS.items():
                expected = round(v * lr_scale, 12)
                self.assertAlmostEqual(tc.get(k, v), expected, places=10,
                                       msg=f"{method} {k}")

    def test_accel23_naive_has_no_lr_override(self):
        tc = self.protocol["methods"]["gsplat_30k_fused_prune10_rclip05_accum16"]["algorithm"]["trainer_cfg"]
        for k in BASE_LRS:
            self.assertNotIn(k, tc)

    def test_accel23_controls_unchanged(self):
        c = self.protocol["methods"]["gsplat_30k_fused_prune10_rclip05"]["algorithm"]["trainer_cfg"]
        self.assertEqual(c["higs_accum_steps"], 1)
        self.assertEqual(c["higs_densify_every"], 0)
        self.assertEqual(c["higs_prune_opacity_thresh"], 0.10)
        self.assertEqual(c["higs_radius_clip"], 0.5)
        f = self.protocol["methods"]["gsplat_30k_fused"]["algorithm"]["trainer_cfg"]
        self.assertEqual(f["higs_accum_steps"], 1)
        self.assertEqual(f["higs_prune_opacity_thresh"], 0.02)

    def test_accel23_plan_jobs(self):
        plan = [j for j in build_ablation_experiment_plan(self.protocol)
                if j["matrix"] == ACCEL23_MATRIX]
        self.assertEqual(len(plan), 66)
        for job in plan:
            self.assertIn(job["method"], ACCEL23_METHODS)
            self.assertEqual(job["seed"], 0)


if __name__ == "__main__":
    unittest.main()