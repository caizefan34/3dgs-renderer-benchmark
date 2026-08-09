"""Focused tests for the Phase-13 SSIM-GT-pyramid protocol.

Pre-registered exploration matrix exploration_accel13_11s0: 5 methods x 11
scenes x seed 0 = 55 jobs. accel12 (constant low-res render) is an honest
negative: early jobs (drjohnson) showed res075/res085 destroy SSIM (drjohnson
Delta SSIM -0.025 / -0.008) while saving almost nothing (anchor_res075
collapsed to 31k Gaussians). accel13 therefore keeps the render at FULL
resolution (L1 stays full-res) and attacks the accel11 bottleneck instead:
per-step GT-side F.interpolate in the 0.75x-SSIM path is eliminated by
caching the downscaled GT/mask per (image, ssim_scale)
(higs_ssim_gt_cache), leaving exactly one render-side interpolate per step.
Faster-GS-style gradient accumulation (higs_accum_steps=2) is added as an
orthogonal lever, and higs_ssim_polish_step=24000 reverts SSIM to full
resolution (no interpolation) for the final refinement window. All patched
methods use a NEW audited source tree gsplat-higs-accel13 (patch
patches/higs-accel13.patch).
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

ACCEL13_MATRIX = "exploration_accel13_11s0"
ACCEL13_METHODS = [
    "gsplat_27k",
    "gsplat_30k_ssim075_gtc",
    "gsplat_30k_ssim075_gtc_acc2",
    "gsplat_30k_ssim085_gtc_acc2",
    "gsplat_30k_ssim075_gtc_acc2_polish24",
]
PATCH_SHA256 = "b8ff024f5eacb5f334c9ca65f347ab926e3337958daccc58041043a46fb1c00d"
TRAINER_SHA256 = "cb67c94358f2f99ec62d0b6c6dec4df7e6defa32aa76b70abb2a802f8098f306"
SOURCE_STATE_SHA256 = "5e6b179d53583a8a61c35a2011397bbc00ba8b9baa0ac7dc5b40581c04f16464"
CANDIDATES = ACCEL13_METHODS[1:]


class HigsAccel13ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel13-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel13_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel13_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL13_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL13_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel13_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel13 = [job for job in plan if job["matrix"] == ACCEL13_MATRIX]
        self.assertEqual(len(accel13), 55)
        methods = Counter(job["method"] for job in accel13)
        self.assertEqual(methods, {name: 11 for name in ACCEL13_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel13))
        self.assertTrue(all(job["executable"] for job in accel13))

    def test_iterations(self):
        expected = {"gsplat_27k": 27000}
        for method_id in CANDIDATES:
            expected[method_id] = 30000
        for method_id, iters in expected.items():
            spec = self.protocol["methods"][method_id]
            self.assertEqual(
                _method_iterations(self.protocol, spec), iters, method_id
            )

    def test_accel13_full_budget_no_early_stop(self):
        for method_id in CANDIDATES:
            spec = self.protocol["methods"][method_id]
            self.assertEqual(spec["algorithm"]["max_steps"], 30000, method_id)

    def test_accel13_full_res_render_contract(self):
        # accel12 negative: low-res RENDER destroys quality; accel13 keeps
        # render at full resolution (L1 full-res) and only downscales SSIM.
        for method_id in CANDIDATES:
            cfg = self.protocol["methods"][method_id]["algorithm"]["trainer_cfg"]
            self.assertEqual(cfg["higs_train_res_scale"], 1.0, method_id)
            self.assertEqual(cfg["higs_full_res_step"], 0, method_id)
            self.assertFalse(cfg["higs_densify_anchor_fullres"], method_id)
            self.assertFalse(cfg["higs_calibrate_scene"], method_id)

    def test_accel13_ssim_gt_cache_and_polish(self):
        cfg0 = self.protocol["methods"]["gsplat_30k_ssim075_gtc"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg0["higs_ssim_scale"], 0.75)
        self.assertEqual(cfg0["higs_ssim_every"], 1)
        self.assertTrue(cfg0["higs_ssim_gt_cache"])
        self.assertEqual(cfg0["higs_ssim_polish_step"], 0)
        self.assertEqual(cfg0["higs_accum_steps"], 1)
        cfg2 = self.protocol["methods"]["gsplat_30k_ssim075_gtc_acc2"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg2["higs_accum_steps"], 2)
        self.assertEqual(cfg2["higs_ssim_scale"], 0.75)
        self.assertTrue(cfg2["higs_ssim_gt_cache"])
        cfg85 = self.protocol["methods"]["gsplat_30k_ssim085_gtc_acc2"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg85["higs_ssim_scale"], 0.85)
        self.assertEqual(cfg85["higs_accum_steps"], 2)
        self.assertTrue(cfg85["higs_ssim_gt_cache"])
        cfgp = self.protocol["methods"]["gsplat_30k_ssim075_gtc_acc2_polish24"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfgp["higs_ssim_scale"], 0.75)
        self.assertEqual(cfgp["higs_accum_steps"], 2)
        self.assertEqual(cfgp["higs_ssim_polish_step"], 24000)

    def test_accel13_renderer_config_contract(self):
        for method_id in CANDIDATES:
            algo = self.protocol["methods"][method_id]["algorithm"]
            self.assertEqual(
                algo["renderer"], "higs_dynamic_native_backward", method_id
            )

    def test_accel13_patch_hash_pinned(self):
        for method_id in CANDIDATES:
            spec = self.protocol["methods"][method_id]
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            self.assertEqual(spec["source_diff_sha256"], PATCH_SHA256)
            self.assertEqual(spec["source_state_sha256"], SOURCE_STATE_SHA256)
            self.assertEqual(spec["patches"], ["patches/higs-accel13.patch"])

    def test_accel13_segment_timing_on(self):
        for method_id in CANDIDATES:
            cfg = self.protocol["methods"][method_id]["algorithm"]["trainer_cfg"]
            self.assertTrue(cfg["higs_segment_timing"])
            self.assertEqual(cfg["higs_timing_start_step"], 15000)
            self.assertEqual(cfg["higs_timing_end_step"], 15300)