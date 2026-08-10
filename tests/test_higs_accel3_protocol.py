"""Focused tests for the Phase-4b HiGS phase-split tile-sampling protocol.

Pre-registered exploration matrix exploration_accel3_11s0: 5 methods x 11 scenes
x seed 0 = 55 jobs. All HiGS candidates run full-resolution with refine_stop_iter
fixed at 15k and tile sampling active ONLY in the refinement window [15000,
max_steps) so densification split/duplicate decisions see full-frame gradients.
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

ACCEL3_MATRIX = "exploration_accel3_11s0"
ACCEL3_METHODS = [
    "gsplat_27k",
    "higs_tilesamp_phase_27k_r05",
    "higs_tilesamp_phase_27k_r05_polish",
    "higs_tilesamp_phase_27k_r06",
    "higs_tilesamp_phase_27k_r04",
]
PHASE_METHODS = ACCEL3_METHODS[1:]
PATCH_SHA256 = "0454262b715bc4eb21c6c62ef00598426c382228abe9da67737716917ec7a470"
TRAINER_SHA256 = "dff6eea6b6bcb516fa42cd1faeef9d6c1b12a63e52debee2fad4e50fb12a78af"
MAX_STEPS = 27000
DENSIFY_END = 15000


class HigsAccel3ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel3-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel3_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel3_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL3_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL3_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel3_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel3 = [job for job in plan if job["matrix"] == ACCEL3_MATRIX]
        self.assertEqual(len(accel3), 55)
        methods = Counter(job["method"] for job in accel3)
        self.assertEqual(methods, {name: 11 for name in ACCEL3_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel3))
        self.assertTrue(all(job["executable"] for job in accel3))

    def test_phase_method_contract(self):
        for method_id in PHASE_METHODS:
            spec = self.protocol["methods"][method_id]
            algo = spec["algorithm"]
            self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
            self.assertEqual(algo["optimizer"], "adam_full")
            self.assertEqual(algo["resolution_schedule"], None)
            self.assertEqual(algo["max_steps"], MAX_STEPS)
            schedule = algo["tile_sampling_schedule"]
            self.assertEqual(schedule["mode"], "phase_split")
            self.assertEqual(schedule["sampling_mode"], "stratified")
            self.assertEqual(schedule["start_step"], DENSIFY_END)
            cfg = algo["trainer_cfg"]
            self.assertFalse(cfg["packed"])
            self.assertFalse(cfg["sparse_grad"])
            self.assertFalse(cfg["visible_adam"])
            self.assertEqual(cfg["higs_tile_sampling_ratio"], schedule["ratio"])
            self.assertEqual(cfg["higs_tile_sampling_start_step"], DENSIFY_END)
            self.assertEqual(
                cfg["higs_tile_sampling_end_step"],
                schedule["end_step"] or 0,
            )
            self.assertTrue(cfg["higs_segment_timing"])
            # timing window inside the sampled refinement phase
            self.assertGreaterEqual(cfg["higs_timing_start_step"], DENSIFY_END)
            # patch identity
            self.assertEqual(spec["patches"], ["patches/higs-tilesample.patch"])
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
                self.assertEqual(len(spec[key]), 64)

    def test_phase_method_budgets(self):
        for method_id in PHASE_METHODS:
            self.assertEqual(
                _method_iterations(self.protocol, self.protocol["methods"][method_id]),
                MAX_STEPS,
            )

    def test_phase_window_contract(self):
        cfg = self.protocol["methods"]["higs_tilesamp_phase_27k_r05"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_tile_sampling_start_step"], DENSIFY_END)
        self.assertEqual(cfg["higs_tile_sampling_end_step"], 0)  # until max_steps
        polish = self.protocol["methods"]["higs_tilesamp_phase_27k_r05_polish"]["algorithm"]["trainer_cfg"]
        self.assertEqual(polish["higs_tile_sampling_start_step"], DENSIFY_END)
        self.assertEqual(polish["higs_tile_sampling_end_step"], 25500)
        sched = self.protocol["methods"]["higs_tilesamp_phase_27k_r05_polish"]["algorithm"]["tile_sampling_schedule"]
        self.assertEqual(sched["end_step"], 25500)

    def test_trainer_cfg_kwargs_passes_phase_fields(self):
        kwargs = trainer_cfg_kwargs(self.protocol["methods"]["higs_tilesamp_phase_27k_r05"])
        self.assertEqual(kwargs["higs_tile_sampling_ratio"], 0.5)
        self.assertEqual(kwargs["higs_tile_sampling_mode"], "stratified")
        self.assertEqual(kwargs["higs_tile_sampling_start_step"], DENSIFY_END)
        self.assertEqual(kwargs["higs_tile_sampling_end_step"], 0)

    def test_patch_file_present_with_recorded_sha(self):
        patch = ROOT / "patches" / "higs-tilesample.patch"
        self.assertTrue(patch.is_file())
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        self.assertEqual(digest, PATCH_SHA256)


if __name__ == "__main__":
    unittest.main()
