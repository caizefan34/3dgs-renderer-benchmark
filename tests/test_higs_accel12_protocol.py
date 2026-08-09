"""Focused tests for the Phase-12 cached-resolution-pyramid protocol.

Pre-registered exploration matrix exploration_accel12_11s0: 5 methods x 11
scenes x seed 0 = 55 jobs. accel12 (redesigned after accel11 honest negative)
trains DIRECTLY at a constant reduced resolution (0.75x / 0.85x) with the
audited accel10 res_cache pyramid: GT/mask/intrinsics are downscaled once per
(image, scale) and cached, and the render runs at the reduced resolution, so
per-step F.interpolate/contiguous/clone work is eliminated (accel11's
ssim075 failed speed CI 0.976 on the train scene because every step paid
F.interpolate on the full-res render). Orthogonal levers: Faster-GS-style
gradient accumulation (higs_accum_steps=2, flushed before densification
boundaries) and full-resolution densification anchors
(higs_densify_anchor_fullres) so split/duplicate decisions keep full-res
screen-space gradients (Phase-3 topology preservation). All patched methods
reuse the audited accel10 source tree (same patch sha256), so no second CUDA
build is needed.
"""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_ablation_protocol import (  # noqa: E402
    build_ablation_experiment_plan,
    validate_ablation_protocol,
)
from higs_training_commands import (  # noqa: E402
    _method_iterations,
)

ACCEL12_MATRIX = "exploration_accel12_11s0"
ACCEL12_METHODS = [
    "gsplat_27k",
    "gsplat_30k_res075",
    "gsplat_30k_res085",
    "higs_accum2_res075",
    "higs_anchor_res075",
]
PATCH_SHA256 = "6ba7ad454d3f0c2749cb50b75640f882e21471ced9af159e09cd24d7be3fca00"
TRAINER_SHA256 = "0b884b5b6dc3add3fa18179acb10ddb64f630d079db6ea401304f7baeccc8fa0"


class HigsAccel12ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel12-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel12_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel12_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL12_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL12_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel12_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel12 = [job for job in plan if job["matrix"] == ACCEL12_MATRIX]
        self.assertEqual(len(accel12), 55)
        methods = Counter(job["method"] for job in accel12)
        self.assertEqual(methods, {name: 11 for name in ACCEL12_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel12))
        self.assertTrue(all(job["executable"] for job in accel12))

    def test_iterations(self):
        expected = {
            "gsplat_27k": 27000,
            "gsplat_30k_res075": 30000,
            "gsplat_30k_res085": 30000,
            "higs_accum2_res075": 30000,
            "higs_anchor_res075": 30000,
        }
        for method_id, iters in expected.items():
            spec = self.protocol["methods"][method_id]
            self.assertEqual(
                _method_iterations(self.protocol, spec), iters, method_id
            )

    def test_gsplat_27k_control_contract(self):
        spec = self.protocol["methods"]["gsplat_27k"]
        self.assertEqual(spec["implementation"], "official")
        self.assertEqual(spec.get("patches"), None)

    def test_accel12_full_budget_no_early_stop(self):
        for method_id in [
            "gsplat_30k_res075",
            "gsplat_30k_res085",
            "higs_accum2_res075",
            "higs_anchor_res075",
        ]:
            spec = self.protocol["methods"][method_id]
            self.assertEqual(spec["algorithm"]["max_steps"], 30000, method_id)
            self.assertEqual(
                _method_iterations(self.protocol, spec), 30000, method_id
            )

    def test_accel12_constant_lowres_configs(self):
        cfg = self.protocol["methods"]["gsplat_30k_res075"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_train_res_scale"], 0.75)
        self.assertEqual(cfg["higs_full_res_step"], 30000)
        self.assertTrue(cfg["higs_res_cache"])
        self.assertFalse(cfg["higs_densify_anchor_fullres"])
        self.assertFalse(cfg["higs_calibrate_scene"])
        self.assertEqual(cfg["higs_ssim_scale"], 1.0)
        self.assertEqual(cfg["higs_ssim_every"], 1)
        self.assertEqual(cfg["higs_skip_bwd_start_step"], 0)
        self.assertEqual(cfg["higs_preload_images"], False)
        self.assertEqual(cfg["visible_adam"], False)
        cfg85 = self.protocol["methods"]["gsplat_30k_res085"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg85["higs_train_res_scale"], 0.85)
        self.assertEqual(cfg85["higs_full_res_step"], 30000)
        self.assertEqual(cfg85["higs_accum_steps"], 1)

    def test_accel12_accum_lever(self):
        cfg = self.protocol["methods"]["higs_accum2_res075"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_accum_steps"], 2)
        self.assertEqual(cfg["higs_train_res_scale"], 0.75)
        self.assertFalse(cfg["higs_densify_anchor_fullres"])

    def test_accel12_anchor_lever(self):
        cfg = self.protocol["methods"]["higs_anchor_res075"]["algorithm"]["trainer_cfg"]
        self.assertTrue(cfg["higs_densify_anchor_fullres"])
        self.assertEqual(cfg["higs_train_res_scale"], 0.75)
        self.assertEqual(cfg["higs_accum_steps"], 1)
        # anchor runs only inside the densification window
        self.assertFalse(cfg["higs_calibrate_scene"])

    def test_accel12_patch_hash_pinned(self):
        for method_id in ACCEL12_METHODS[1:]:
            spec = self.protocol["methods"][method_id]
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            self.assertEqual(
                spec["source_diff_sha256"], PATCH_SHA256,
                "reuses audited accel10 tree with no new diff",
            )
            self.assertEqual(spec["patches"], ["patches/higs-accel10.patch"])

    def test_accel12_segment_timing_on(self):
        for method_id in ACCEL12_METHODS[1:]:
            cfg = self.protocol["methods"][method_id]["algorithm"]["trainer_cfg"]
            self.assertTrue(cfg["higs_segment_timing"])
            self.assertEqual(cfg["higs_timing_start_step"], 15000)
            self.assertEqual(cfg["higs_timing_end_step"], 15300)


if __name__ == "__main__":
    unittest.main()
