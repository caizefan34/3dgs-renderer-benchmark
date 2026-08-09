"""Focused tests for the Phase-18 Group-Training lever protocol.

Pre-registered exploration matrix exploration_accel18_11s0: 6 methods x 11
scenes x seed 0 = 66 jobs. accel18 (Group Training, ICCV 2025) adds opacity-
based Gaussian grouping on top of the fused-SSIM baseline
(gsplat_30k_fused): after max(500, refine_stop_iter=15000), every
higs_group_interval steps the training render uses only higs_group_utr
fraction of Gaussians sampled without replacement proportional to
sigmoid(opacity); cached Gaussians get zero gradient that step (their Adam
state persists); full-group merge passes at 14500/29000; eval renders always
use all Gaussians; the densification phase (0-15k) is untouched so topology
formation is preserved. All patched methods use a NEW audited source tree
gsplat-higs-accel18 (patch patches/higs-accel18.patch) whose only delta vs
accel15 is the group-training lever in simple_trainer.py plus a Stage.render
splats-override in gsplat/stage/components/stage.py.
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

ACCEL18_MATRIX = "exploration_accel18_11s0"
ACCEL18_METHODS = [
    "gsplat",
    "gsplat_30k_fused",
    "gsplat_30k_fused_group",
    "gsplat_30k_fused_group_utr075",
    "gsplat_30k_fused_group_prune10",
    "gsplat_30k_fused_group_prune10_rclip05",
]
PATCH_SHA256 = "7b2095c2918d0d6d08f8a14e7bf63b338049acb6b3550b75ca940bd640beba53"
TRAINER_SHA256 = "307ccf3643a7e3288c1efb826d3f5d7846a507ba2625f7a4766e26b9b53e986e"
SOURCE_STATE_SHA256 = "b67687887584c6a15be3845423e1ad4268415fc5cfbc211327e035f3ad46d983"


class HigsAccel18ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel18-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel18_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["executable_jobs"], 66)
        self.assertEqual(report["planned_jobs"], 66)

    def test_accel18_matrix_shape(self):
        matrix = next(m for m in self.protocol["matrices"] if m["id"] == ACCEL18_MATRIX)
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL18_METHODS)
        self.assertEqual(matrix["seeds"], [0])

    def test_accel18_hashes_pinned(self):
        for method in ACCEL18_METHODS[2:]:
            spec = self.protocol["methods"][method]
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            self.assertEqual(spec["source_state_sha256"], SOURCE_STATE_SHA256)
            self.assertEqual(spec["patches"], ["patches/higs-accel18.patch"])

    def test_accel18_group_lever_wired(self):
        cfg = self.protocol["methods"]["gsplat_30k_fused_group"]["algorithm"]["trainer_cfg"]
        self.assertTrue(cfg["higs_group_train"])
        self.assertEqual(cfg["higs_group_utr"], 0.6)
        self.assertEqual(cfg["higs_group_interval"], 500)
        self.assertEqual(cfg["higs_group_start_step"], 500)
        self.assertEqual(cfg["higs_group_merge_step_o"], 29000)
        self.assertEqual(cfg["higs_ssim_scale"], 1.0)
        self.assertEqual(cfg["higs_train_res_scale"], 1.0)
        self.assertFalse(cfg["higs_accum_steps"] > 1)
        self.assertFalse(cfg["higs_densify_every"] > 0)

    def test_accel18_utr075_and_stack(self):
        u = self.protocol["methods"]["gsplat_30k_fused_group_utr075"]["algorithm"]["trainer_cfg"]
        self.assertEqual(u["higs_group_utr"], 0.75)
        full = self.protocol["methods"]["gsplat_30k_fused_group_prune10_rclip05"]["algorithm"]["trainer_cfg"]
        self.assertEqual(full["higs_prune_mode"], "opacity")
        self.assertEqual(full["higs_prune_opacity_thresh"], 0.1)
        self.assertEqual(full["higs_radius_clip"], 0.5)

    def test_accel18_fused_baseline_group_off(self):
        fused = self.protocol["methods"]["gsplat_30k_fused"]["algorithm"]["trainer_cfg"]
        self.assertFalse(fused["higs_group_train"])

    def test_accel18_plan_jobs(self):
        plan = [j for j in build_ablation_experiment_plan(self.protocol) if j["matrix"] == ACCEL18_MATRIX]
        self.assertEqual(len(plan), 66)
        for job in plan:
            self.assertIn(job["method"], ACCEL18_METHODS)
            self.assertEqual(job["seed"], 0)


if __name__ == "__main__":
    unittest.main()
