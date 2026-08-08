"""Focused tests for the Phase-5 SSIM-polish exploration protocol.

Pre-registered exploration matrix exploration_accel5_11s0: 5 methods x 11 scenes
x seed 0 = 55 jobs. Mechanism (accel5): inside the error-guided tile-sampling
window [15000, 27000) every ``higs_ssim_polish_every`` steps runs a FULL
L1+SSIM loss on the full image; the other steps keep the sparse L1-only tile
loss. accel4 skipped SSIM for the whole window (SSIM/LPIPS regressed); polish
re-anchors the perceptual loss at negligible wall-clock cost.
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

ACCEL5_MATRIX = "exploration_accel5_11s0"
ACCEL5_METHODS = [
    "gsplat_27k",
    "higs_eg_sparse_phase_27k_r07",
    "higs_eg_sparse_phase_27k_r07_polish25",
    "higs_eg_sparse_phase_27k_r07_polish50",
    "higs_eg_sparse_phase_27k_r07_polish100",
]
POLISH_METHODS = ACCEL5_METHODS[2:]
PATCH_SHA256 = "b7e99eb078a07d6de1b90ba83acbb4bfda14940bdb9742e220a76bc6086ef981"
TRAINER_SHA256 = "af55fdf525fe4d249817a828d64432d890bf3f115236061cea98a148f2c7b950"
DENSIFY_END = 15000


class HigsAccel5ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel5-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel5_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel5_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL5_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL5_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel5_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel5 = [job for job in plan if job["matrix"] == ACCEL5_MATRIX]
        self.assertEqual(len(accel5), 55)
        methods = Counter(job["method"] for job in accel5)
        self.assertEqual(methods, {name: 11 for name in ACCEL5_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel5))
        self.assertTrue(all(job["executable"] for job in accel5))

    def test_polish_method_contract(self):
        for method_id in POLISH_METHODS:
            spec = self.protocol["methods"][method_id]
            algo = spec["algorithm"]
            self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
            self.assertEqual(algo["optimizer"], "adam_full")
            self.assertEqual(algo["resolution_schedule"], None)
            schedule = algo["tile_sampling_schedule"]
            self.assertEqual(schedule["mode"], "phase_split_error_guided_ssim_polish")
            self.assertEqual(schedule["sampling_mode"], "error_guided")
            self.assertEqual(schedule["start_step"], DENSIFY_END)
            self.assertTrue(schedule["sparse_loss"])
            cfg = algo["trainer_cfg"]
            self.assertFalse(cfg["packed"])
            self.assertFalse(cfg["sparse_grad"])
            self.assertFalse(cfg["visible_adam"])
            self.assertEqual(cfg["higs_tile_sampling_ratio"], 0.7)
            self.assertEqual(cfg["higs_tile_sampling_mode"], "error_guided")
            self.assertEqual(cfg["higs_tile_sampling_start_step"], DENSIFY_END)
            self.assertTrue(cfg["higs_sparse_loss"])
            # polish knob must match the schedule
            self.assertEqual(cfg["higs_ssim_polish_every"], schedule["ssim_polish_every"])
            self.assertGreater(cfg["higs_ssim_polish_every"], 0)
            # patch identity
            self.assertEqual(spec["patches"], ["patches/higs-accel5.patch"])
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
                self.assertEqual(len(spec[key]), 64)

    def test_reference_method_keeps_polish_off(self):
        spec = self.protocol["methods"]["higs_eg_sparse_phase_27k_r07"]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_ssim_polish_every"], 0)
        self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
        self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)

    def test_polish_method_budgets(self):
        for method_id in ACCEL5_METHODS[1:]:
            self.assertEqual(
                _method_iterations(self.protocol, self.protocol["methods"][method_id]),
                27000,
            )

    def test_trainer_cfg_kwargs_passes_polish_field(self):
        for method_id in POLISH_METHODS:
            spec = self.protocol["methods"][method_id]
            kwargs = trainer_cfg_kwargs(spec)
            self.assertEqual(kwargs["higs_tile_sampling_ratio"], 0.7)
            self.assertEqual(kwargs["higs_tile_sampling_mode"], "error_guided")
            self.assertEqual(kwargs["higs_tile_sampling_start_step"], DENSIFY_END)
            self.assertTrue(kwargs["higs_sparse_loss"])
            self.assertEqual(
                kwargs["higs_ssim_polish_every"],
                spec["algorithm"]["trainer_cfg"]["higs_ssim_polish_every"],
            )

    def test_patch_file_present_with_recorded_sha(self):
        patch = ROOT / "patches" / "higs-accel5.patch"
        self.assertTrue(patch.is_file())
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        self.assertEqual(digest, PATCH_SHA256)

    def test_patch_implements_periodic_ssim_polish(self):
        """Regression: the polish knob must bypass tile sampling on polish
        steps so the full-frame L1+SSIM loss runs (no sparse SSIM)."""
        patch = (ROOT / "patches" / "higs-accel5.patch").read_text(encoding="utf-8")
        self.assertIn("higs_ssim_polish_every", patch)
        self.assertIn("polish = (", patch)
        self.assertIn("if not polish:", patch)


if __name__ == "__main__":
    unittest.main()
