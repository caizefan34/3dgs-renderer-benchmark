"""Focused tests for the Phase-14 post-refine opacity-pruning protocol.

Pre-registered exploration matrix exploration_accel14_11s0: 5 methods x 11
scenes x seed 0 = 55 jobs. accel14 (FastGS-VCP-style) adds an in-place
post-refine opacity prune on top of the accel13 levers (full-res render +
cached GT/mask SSIM pyramid + single-side interpolate + accum2/polish): after
higs_prune_start_step, every higs_prune_every steps, Gaussians with
sigmoid(opacity) < higs_prune_opacity_thresh are removed via
gsplat.strategy.ops.remove. Densification runs untouched until the prune
window (topology preserved through the 15k densification phase), then the
final 3000-step gap after the last prune lets the model re-fit. The prune
only removes (never adds) so it cannot be mistaken for early-stop. All
patched methods use a NEW audited source tree gsplat-higs-accel14 (patch
patches/higs-accel14.patch), validated by 60-step smokes: thresh 0.02 is a
no-op at SFM init (opacity sigmoids ~0.04-0.24, all above 0.02) and matches
the prune-off control bit-for-bit; thresh 0.1 removes 80,861 -> 37,524
Gaussians with clean eval, proving the remove path + scene.on_remove +
optimizer-state slicing work end-to-end.
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

ACCEL14_MATRIX = "exploration_accel14_11s0"
ACCEL14_METHODS = [
    "gsplat_27k",
    "gsplat_30k_prune16",
    "gsplat_30k_ssim075_gtc_acc2_prune16",
    "gsplat_30k_ssim075_gtc_acc2_prune18",
    "gsplat_30k_ssim075_gtc_acc2_polish24_prune16",
]
PATCH_SHA256 = "e7f30680869547e5366837d05c5e24703c5bd95e78c310f31b1f87736b74d034"
TRAINER_SHA256 = "440ae7a28dee839a3525f1750cf4a616b9295a234036298821b968cedf3ef8b2"
SOURCE_STATE_SHA256 = "5a0a0eef42aa558dca564de6edb902db22a30761067947493c03cbc246c6d992"
CANDIDATES = ACCEL14_METHODS[1:]


class HigsAccel14ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel14-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel14_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel14_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL14_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL14_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel14_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel14 = [job for job in plan if job["matrix"] == ACCEL14_MATRIX]
        self.assertEqual(len(accel14), 55)
        methods = Counter(job["method"] for job in accel14)
        self.assertEqual(methods, {name: 11 for name in ACCEL14_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel14))
        self.assertTrue(all(job["executable"] for job in accel14))

    def test_iterations(self):
        expected = {"gsplat_27k": 27000}
        for method_id in CANDIDATES:
            expected[method_id] = 30000
        for method_id, iters in expected.items():
            spec = self.protocol["methods"][method_id]
            self.assertEqual(
                _method_iterations(self.protocol, spec), iters, method_id
            )

    def test_accel14_full_budget_no_early_stop(self):
        for method_id in CANDIDATES:
            spec = self.protocol["methods"][method_id]
            self.assertEqual(spec["algorithm"]["max_steps"], 30000, method_id)

    def test_accel14_full_res_render_contract(self):
        # accel12 negative: low-res RENDER destroys quality; accel13/14 keep
        # the render at full resolution (L1 full-res) and only downscale SSIM.
        for method_id in CANDIDATES:
            cfg = self.protocol["methods"][method_id]["algorithm"]["trainer_cfg"]
            self.assertEqual(cfg["higs_train_res_scale"], 1.0, method_id)
            self.assertEqual(cfg["higs_full_res_step"], 0, method_id)
            self.assertFalse(cfg["higs_densify_anchor_fullres"], method_id)
            self.assertFalse(cfg["higs_calibrate_scene"], method_id)

    def test_accel14_prune_levers(self):
        cfg_base = self.protocol["methods"]["gsplat_30k_prune16"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg_base["higs_prune_mode"], "opacity")
        self.assertEqual(cfg_base["higs_prune_start_step"], 15000)
        self.assertEqual(cfg_base["higs_prune_every"], 3000)
        self.assertEqual(cfg_base["higs_prune_opacity_thresh"], 0.02)
        # prune16/18 differ only in the prune start step
        cfg16 = self.protocol["methods"]["gsplat_30k_ssim075_gtc_acc2_prune16"]["algorithm"]["trainer_cfg"]
        cfg18 = self.protocol["methods"]["gsplat_30k_ssim075_gtc_acc2_prune18"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg16["higs_prune_start_step"], 15000)
        self.assertEqual(cfg18["higs_prune_start_step"], 18000)
        self.assertEqual(cfg16["higs_prune_every"], cfg18["higs_prune_every"])
        self.assertEqual(
            cfg16["higs_prune_opacity_thresh"],
            cfg18["higs_prune_opacity_thresh"],
        )
        # prune never removes before the densification phase (start >= 15k)
        for method_id in CANDIDATES:
            cfg = self.protocol["methods"][method_id]["algorithm"]["trainer_cfg"]
            self.assertGreaterEqual(cfg["higs_prune_start_step"], 15000, method_id)
        # prune + accel13 levers compose
        cfgp = self.protocol["methods"]["gsplat_30k_ssim075_gtc_acc2_polish24_prune16"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfgp["higs_ssim_scale"], 0.75)
        self.assertEqual(cfgp["higs_accum_steps"], 2)
        self.assertEqual(cfgp["higs_ssim_polish_step"], 24000)
        self.assertEqual(cfgp["higs_prune_start_step"], 15000)
        self.assertTrue(cfgp["higs_ssim_gt_cache"])

    def test_accel14_renderer_config_contract(self):
        for method_id in CANDIDATES:
            algo = self.protocol["methods"][method_id]["algorithm"]
            self.assertEqual(
                algo["renderer"], "higs_dynamic_native_backward", method_id
            )

    def test_accel14_patch_hash_pinned(self):
        for method_id in CANDIDATES:
            spec = self.protocol["methods"][method_id]
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            self.assertEqual(spec["source_diff_sha256"], PATCH_SHA256)
            self.assertEqual(spec["source_state_sha256"], SOURCE_STATE_SHA256)
            self.assertEqual(spec["patches"], ["patches/higs-accel14.patch"])

    def test_accel14_segment_timing_on(self):
        for method_id in CANDIDATES:
            cfg = self.protocol["methods"][method_id]["algorithm"]["trainer_cfg"]
            self.assertTrue(cfg["higs_segment_timing"])
            self.assertEqual(cfg["higs_timing_start_step"], 15000)
            self.assertEqual(cfg["higs_timing_end_step"], 15300)
