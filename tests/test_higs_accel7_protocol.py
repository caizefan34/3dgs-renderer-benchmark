"""Focused tests for the Phase-7 system-speedup exploration protocol.

Pre-registered exploration matrix exploration_accel7_11s0: 5 methods x 11
scenes x seed 0 = 55 jobs. accel7 adds three quality-neutral levers inside
the wall clock: (1) GPU image preload, (2) gradient accumulation with
accum_steps=8 (Adam step amortized, gradients flushed at densification /
opacity-reset boundaries), and (3) error-map refresh amortization (dense /
polish renders reuse their output to refresh the per-image tile-error
cache). The matrix includes gsplat_27k_preload_accum8 (system control: levers
1+2, no tile sampling) so system speedups are never attributed to HiGS.
"""
import hashlib
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
    trainer_cfg_kwargs,
)

ACCEL7_MATRIX = "exploration_accel7_11s0"
ACCEL7_METHODS = [
    "gsplat_27k",
    "gsplat_27k_preload_accum8",
    "higs_eg_sparse_phase_27k_r07_polish25_acc7",
    "higs_eg_sparse_phase_27k_r07_polish50_acc7",
    "higs_eg_sparse_phase_27k_r07_polish100_acc7",
]
HIGS_ACC7_METHODS = ACCEL7_METHODS[2:]
PATCH_SHA256 = "859b011ecce6f62ec922c7d624bdd60c25c9fe1885b2d97b69370e85075aabe1"
TRAINER_SHA256 = "9d7b08179a18aea0622608d4729e966417a4c88aac6d5a582b32f1586df766a8"
DENSIFY_END = 15000


class HigsAccel7ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel7-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel7_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel7_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL7_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL7_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel7_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel7 = [job for job in plan if job["matrix"] == ACCEL7_MATRIX]
        self.assertEqual(len(accel7), 55)
        methods = Counter(job["method"] for job in accel7)
        self.assertEqual(methods, {name: 11 for name in ACCEL7_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel7))
        self.assertTrue(all(job["executable"] for job in accel7))

    def test_system_control_contract(self):
        spec = self.protocol["methods"]["gsplat_27k_preload_accum8"]
        algo = spec["algorithm"]
        self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
        self.assertEqual(algo["optimizer"], "adam_full")
        self.assertIsNone(algo["resolution_schedule"])
        self.assertIsNone(algo["tile_sampling_schedule"])
        self.assertEqual(algo["max_steps"], 27000)
        cfg = algo["trainer_cfg"]
        self.assertTrue(cfg["higs_preload_images"])
        self.assertEqual(cfg["higs_accum_steps"], 8)
        self.assertFalse(cfg.get("higs_tile_sampling_ratio", 1.0) < 1.0)
        self.assertFalse(cfg.get("higs_err_refresh_amortize", False))
        self.assertFalse(cfg["packed"])
        self.assertFalse(cfg["sparse_grad"])
        self.assertEqual(spec["patches"], ["patches/higs-accel7.patch"])
        for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
            self.assertEqual(len(spec[key]), 64)

    def test_higs_acc7_method_contract(self):
        for method_id in HIGS_ACC7_METHODS:
            spec = self.protocol["methods"][method_id]
            algo = spec["algorithm"]
            self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
            self.assertEqual(algo["optimizer"], "adam_full")
            self.assertIsNone(algo["resolution_schedule"])
            schedule = algo["tile_sampling_schedule"]
            self.assertEqual(schedule["mode"], "phase_split_error_guided_ssim_polish_accel7")
            self.assertEqual(schedule["start_step"], DENSIFY_END)
            self.assertTrue(schedule["sparse_loss"])
            self.assertTrue(schedule["preload_images"])
            self.assertEqual(schedule["accum_steps"], 8)
            self.assertTrue(schedule["err_refresh_amortize"])
            cfg = algo["trainer_cfg"]
            self.assertFalse(cfg["packed"])
            self.assertFalse(cfg["sparse_grad"])
            self.assertFalse(cfg["visible_adam"])
            self.assertEqual(cfg["higs_tile_sampling_ratio"], 0.7)
            self.assertEqual(cfg["higs_tile_sampling_start_step"], DENSIFY_END)
            self.assertTrue(cfg["higs_sparse_loss"])
            self.assertTrue(cfg["higs_preload_images"])
            self.assertEqual(cfg["higs_accum_steps"], 8)
            self.assertTrue(cfg["higs_err_refresh_amortize"])
            self.assertEqual(
                cfg["higs_ssim_polish_every"], schedule["ssim_polish_every"]
            )
            self.assertGreater(cfg["higs_ssim_polish_every"], 0)
            self.assertEqual(spec["patches"], ["patches/higs-accel7.patch"])
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
                self.assertEqual(len(spec[key]), 64)

    def test_acc7_method_budgets(self):
        for method_id in ACCEL7_METHODS[1:]:
            self.assertEqual(
                _method_iterations(self.protocol, self.protocol["methods"][method_id]),
                27000,
            )

    def test_trainer_cfg_kwargs_passes_acc7_fields(self):
        for method_id in HIGS_ACC7_METHODS + ["gsplat_27k_preload_accum8"]:
            spec = self.protocol["methods"][method_id]
            kwargs = trainer_cfg_kwargs(spec)
            self.assertTrue(kwargs["higs_preload_images"])
            self.assertEqual(kwargs["higs_accum_steps"], 8)

    def test_patch_file_present_with_recorded_sha(self):
        patch = ROOT / "patches" / "higs-accel7.patch"
        self.assertTrue(patch.is_file())
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        self.assertEqual(digest, PATCH_SHA256)

    def test_patch_implements_accel7_levers(self):
        """Regression: the accel7 patch must add preload + grad accum."""
        patch = (ROOT / "patches" / "higs-accel7.patch").read_text(encoding="utf-8")
        self.assertIn("higs_preload_images", patch)
        self.assertIn("higs_accum_steps", patch)
        self.assertIn("higs_err_refresh_amortize", patch)
        self.assertIn("do_opt_step", patch)
        self.assertIn("_higs_preloaded", patch)


if __name__ == "__main__":
    unittest.main()