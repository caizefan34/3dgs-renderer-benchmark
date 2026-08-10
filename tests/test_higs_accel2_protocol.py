"""Focused tests for the Phase-4 HiGS tile-sampled acceleration protocol.

Covers the pre-registered exploration matrix (exploration_accel2_11s0), the
new tilesample patch identity, per-method training budgets, tile-sampling
trainer kwargs, and the sampling schedule contract. The HiGS tile-sampled
backward itself (Round 39 native path) is covered by the existing renderer
tests in tests/test_higs_native_backward.py.
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

ACCEL2_MATRIX = "exploration_accel2_11s0"
ACCEL2_METHODS = [
    "gsplat_27k",
    "higs_tilesamp_30k_r07",
    "higs_tilesamp_27k_r07",
    "higs_tilesamp_27k_r05",
    "higs_tilesamp_refine_27k_r05",
]
PATCH_SHA256 = "0454262b715bc4eb21c6c62ef00598426c382228abe9da67737716917ec7a470"
TRAINER_SHA256 = "dff6eea6b6bcb516fa42cd1faeef9d6c1b12a63e52debee2fad4e50fb12a78af"


class HigsAccel2ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel2-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel2_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel2_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL2_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL2_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel2_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel2 = [job for job in plan if job["matrix"] == ACCEL2_MATRIX]
        self.assertEqual(len(accel2), 55)
        methods = Counter(job["method"] for job in accel2)
        self.assertEqual(
            methods,
            {name: 11 for name in ACCEL2_METHODS},
        )
        self.assertTrue(all(job["seed"] == 0 for job in accel2))
        self.assertTrue(all(job["executable"] for job in accel2))

    def test_tilesample_method_contract(self):
        for method_id in ACCEL2_METHODS[1:]:
            spec = self.protocol["methods"][method_id]
            algo = spec["algorithm"]
            self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
            self.assertEqual(algo["optimizer"], "adam_full")
            self.assertEqual(algo["resolution_schedule"], None)
            schedule = algo["tile_sampling_schedule"]
            self.assertIn(schedule["sampling_mode"], ("uniform", "stratified"))
            self.assertGreaterEqual(schedule["ratio"], 0.5)
            self.assertLessEqual(schedule["ratio"], 0.7)
            cfg = algo["trainer_cfg"]
            self.assertFalse(cfg["packed"])
            self.assertFalse(cfg["sparse_grad"])
            self.assertFalse(cfg["visible_adam"])
            self.assertEqual(cfg["higs_tile_sampling_ratio"], schedule["ratio"])
            self.assertEqual(cfg["higs_tile_sampling_mode"], schedule["sampling_mode"])
            self.assertTrue(cfg["higs_segment_timing"])
            # patch identity
            self.assertEqual(spec["patches"], ["patches/higs-tilesample.patch"])
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
                self.assertEqual(len(spec[key]), 64)

    def test_tilesample_method_budgets(self):
        self.assertEqual(
            _method_iterations(self.protocol, self.protocol["methods"]["higs_tilesamp_30k_r07"]), 30000
        )
        for method_id in ("higs_tilesamp_27k_r07", "higs_tilesamp_27k_r05", "higs_tilesamp_refine_27k_r05"):
            self.assertEqual(
                _method_iterations(self.protocol, self.protocol["methods"][method_id]), 27000
            )

    def test_refine_window_contract(self):
        cfg = self.protocol["methods"]["higs_tilesamp_refine_27k_r05"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_tile_sampling_start_step"], 0)
        self.assertEqual(cfg["higs_tile_sampling_end_step"], 25500)
        sched = self.protocol["methods"]["higs_tilesamp_refine_27k_r05"]["algorithm"]["tile_sampling_schedule"]
        self.assertEqual(sched["mode"], "window")
        self.assertEqual(sched["end_step"], 25500)

    def test_trainer_cfg_kwargs_passes_tilesample_fields(self):
        kwargs = trainer_cfg_kwargs(self.protocol["methods"]["higs_tilesamp_27k_r05"])
        self.assertFalse(kwargs["packed"])
        self.assertFalse(kwargs["sparse_grad"])
        self.assertEqual(kwargs["higs_tile_sampling_ratio"], 0.5)
        self.assertEqual(kwargs["higs_tile_sampling_mode"], "stratified")

    def test_patch_file_present_with_recorded_sha(self):
        patch = ROOT / "patches" / "higs-tilesample.patch"
        self.assertTrue(patch.is_file())
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        self.assertEqual(digest, PATCH_SHA256)


if __name__ == "__main__":
    unittest.main()