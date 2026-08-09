"""Focused tests for the Phase-16 system-stack lever protocol.

Pre-registered exploration matrix exploration_accel16_11s0: 6 methods x 11
scenes x seed 0 = 66 jobs. accel16 stacks quality-neutral system levers on the
fused-SSIM baseline (gsplat_30k_fused): image preloading (higs_preload_images,
zero trajectory change), Faster-GS-style densification frequency
(higs_densify_every=600), gradient accumulation 2 (higs_accum_steps=2), and a
gentle post-refine opacity prune (thresh 0.02, start 15k so densification
topology is untouched). All methods stay at full resolution
(higs_ssim_scale=1.0, higs_train_res_scale=1.0). The in-matrix official gsplat
control gives same-window speed pairing; the fused baseline isolates the
system-level speedup from the stacked candidates.
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

ACCEL16_MATRIX = "exploration_accel16_11s0"
ACCEL16_METHODS = [
    "gsplat",
    "gsplat_30k_fused",
    "gsplat_30k_fused_preload",
    "gsplat_30k_fused_preload_dens600",
    "gsplat_30k_fused_preload_dens600_accum2",
    "gsplat_30k_fused_preload_dens600_accum2_prune02",
]
PATCH_SHA256 = "04f950778f617295cf28611487a017d081dd4dcc5f056560d60c598a5eafa5e7"
TRAINER_SHA256 = "12bb76d2a7d4f9640badfa2c58f93f262572b3cd74a7b1d769bf169248c8a7f5"
SOURCE_STATE_SHA256 = "87880d7af2619ae7eb406384b88b7c2b1987073be7de667d2f79f198376bbb7f"


class HigsAccel16ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel16-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel16_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["executable_jobs"], 66)
        self.assertEqual(report["planned_jobs"], 66)

    def test_accel16_matrix_shape(self):
        matrix = next(m for m in self.protocol["matrices"] if m["id"] == ACCEL16_MATRIX)
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL16_METHODS)
        self.assertEqual(matrix["seeds"], [0])

    def test_accel16_hashes_pinned(self):
        for method in ACCEL16_METHODS[1:]:
            spec = self.protocol["methods"][method]
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            self.assertEqual(spec["source_state_sha256"], SOURCE_STATE_SHA256)
            self.assertEqual(spec["patches"], ["patches/higs-accel15.patch"])

    def test_accel16_levers_wired(self):
        cfg = self.protocol["methods"]["gsplat_30k_fused_preload_dens600_accum2_prune02"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_preload_images"], True)
        self.assertEqual(cfg["higs_densify_every"], 600)
        self.assertEqual(cfg["higs_accum_steps"], 2)
        self.assertEqual(cfg["higs_prune_opacity_thresh"], 0.02)
        self.assertEqual(cfg["higs_prune_start_step"], 15000)
        self.assertEqual(cfg["higs_ssim_scale"], 1.0)
        self.assertEqual(cfg["higs_train_res_scale"], 1.0)
        self.assertEqual(cfg["sh_fp16"], False)

    def test_accel16_incremental_levers(self):
        fused = self.protocol["methods"]["gsplat_30k_fused"]["algorithm"]["trainer_cfg"]
        preload = self.protocol["methods"]["gsplat_30k_fused_preload"]["algorithm"]["trainer_cfg"]
        dens = self.protocol["methods"]["gsplat_30k_fused_preload_dens600"]["algorithm"]["trainer_cfg"]
        accum = self.protocol["methods"]["gsplat_30k_fused_preload_dens600_accum2"]["algorithm"]["trainer_cfg"]
        self.assertFalse(fused["higs_preload_images"])
        self.assertTrue(preload["higs_preload_images"])
        self.assertEqual(dens["higs_densify_every"], 600)
        self.assertEqual(accum["higs_accum_steps"], 2)

    def test_accel16_official_control_in_matrix(self):
        matrix = next(m for m in self.protocol["matrices"] if m["id"] == ACCEL16_MATRIX)
        self.assertIn("gsplat", matrix["methods"])
        spec = self.protocol["methods"]["gsplat"]
        self.assertIsNone(spec.get("algorithm"))


if __name__ == "__main__":
    unittest.main()
