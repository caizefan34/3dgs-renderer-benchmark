"""Focused tests for the Phase-6 calibrated SSIM-polish exploration protocol.

Pre-registered exploration matrix exploration_accel6_11s0: 5 methods x 11
scenes x seed 0 = 55 jobs. Mechanism (accel6) = accel5 SSIM polish plus a
deterministic per-scene guard: 20 dense + 20 sparse real training steps are
timed at the sparse-window start; the sparse phase is kept only if it is
>= 1.15x faster per step, otherwise the run falls back to the dense path
(full L1+SSIM every step) for the rest of the window. This prevents scenes
where sampling overhead exceeds render savings (e.g. tanks train/truck)
from making the candidate slower than dense 27k.
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

ACCEL6_MATRIX = "exploration_accel6_11s0"
ACCEL6_METHODS = [
    "gsplat_27k",
    "higs_eg_sparse_phase_27k_r07_polish50",
    "higs_eg_sparse_phase_27k_r07_polish25_cal",
    "higs_eg_sparse_phase_27k_r07_polish50_cal",
    "higs_eg_sparse_phase_27k_r07_polish100_cal",
]
CAL_METHODS = ACCEL6_METHODS[2:]
PATCH_SHA256 = "8c7df57000c443de8075516dfaee30cc00629b1b07694bbff7b2653f26a248bb"
TRAINER_SHA256 = "c4d2cd67d92349618d70cae99f8fefd351779dcb9330d43085b3237f9b121ff5"
DENSIFY_END = 15000


class HigsAccel6ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel6-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel6_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel6_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL6_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL6_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel6_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel6 = [job for job in plan if job["matrix"] == ACCEL6_MATRIX]
        self.assertEqual(len(accel6), 55)
        methods = Counter(job["method"] for job in accel6)
        self.assertEqual(methods, {name: 11 for name in ACCEL6_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel6))
        self.assertTrue(all(job["executable"] for job in accel6))

    def test_cal_method_contract(self):
        for method_id in CAL_METHODS:
            spec = self.protocol["methods"][method_id]
            algo = spec["algorithm"]
            self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
            self.assertEqual(algo["optimizer"], "adam_full")
            self.assertEqual(algo["resolution_schedule"], None)
            schedule = algo["tile_sampling_schedule"]
            self.assertEqual(
                schedule["mode"], "phase_split_error_guided_ssim_polish_calibrated"
            )
            self.assertEqual(schedule["start_step"], DENSIFY_END)
            self.assertTrue(schedule["sparse_loss"])
            self.assertTrue(schedule["calibrate"])
            self.assertEqual(schedule["calibrate_min_speedup"], 1.15)
            self.assertGreater(schedule["calibrate_steps"], 0)
            cfg = algo["trainer_cfg"]
            self.assertFalse(cfg["packed"])
            self.assertFalse(cfg["sparse_grad"])
            self.assertFalse(cfg["visible_adam"])
            self.assertEqual(cfg["higs_tile_sampling_ratio"], 0.7)
            self.assertEqual(cfg["higs_tile_sampling_start_step"], DENSIFY_END)
            self.assertTrue(cfg["higs_sparse_loss"])
            self.assertTrue(cfg["higs_sparse_calibrate"])
            self.assertEqual(cfg["higs_sparse_min_speedup"], 1.15)
            self.assertGreater(cfg["higs_sparse_calibrate_steps"], 0)
            self.assertEqual(
                cfg["higs_ssim_polish_every"], schedule["ssim_polish_every"]
            )
            self.assertGreater(cfg["higs_ssim_polish_every"], 0)
            self.assertEqual(spec["patches"], ["patches/higs-accel6.patch"])
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
                self.assertEqual(len(spec[key]), 64)

    def test_reference_method_keeps_calibration_off(self):
        spec = self.protocol["methods"]["higs_eg_sparse_phase_27k_r07_polish50"]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertFalse(cfg["higs_sparse_calibrate"])
        self.assertEqual(cfg["higs_ssim_polish_every"], 50)
        self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
        self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)

    def test_cal_method_budgets(self):
        for method_id in ACCEL6_METHODS[1:]:
            self.assertEqual(
                _method_iterations(self.protocol, self.protocol["methods"][method_id]),
                27000,
            )

    def test_trainer_cfg_kwargs_passes_cal_fields(self):
        for method_id in CAL_METHODS:
            spec = self.protocol["methods"][method_id]
            kwargs = trainer_cfg_kwargs(spec)
            self.assertEqual(kwargs["higs_tile_sampling_ratio"], 0.7)
            self.assertTrue(kwargs["higs_sparse_calibrate"])
            self.assertEqual(kwargs["higs_sparse_min_speedup"], 1.15)
            self.assertEqual(
                kwargs["higs_ssim_polish_every"],
                spec["algorithm"]["trainer_cfg"]["higs_ssim_polish_every"],
            )

    def test_patch_file_present_with_recorded_sha(self):
        patch = ROOT / "patches" / "higs-accel6.patch"
        self.assertTrue(patch.is_file())
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        self.assertEqual(digest, PATCH_SHA256)

    def test_patch_implements_calibration_gate(self):
        """Regression: the accel6 patch must add the per-scene sparse guard."""
        patch = (ROOT / "patches" / "higs-accel6.patch").read_text(encoding="utf-8")
        self.assertIn("higs_sparse_calibrate", patch)
        self.assertIn("sparse_calibrate", patch)
        self.assertIn("higs_sparse_min_speedup", patch)
        self.assertIn("sparse_enabled", patch)


if __name__ == "__main__":
    unittest.main()
