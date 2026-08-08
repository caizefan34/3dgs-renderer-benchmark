"""Focused tests for the Phase-8 scheduling-levers exploration protocol.

Pre-registered exploration matrix exploration_accel8_11s0: 5 methods x 11
scenes x seed 0 = 55 jobs. accel8 adds the two remaining low-risk scheduling
levers from the literature on top of accel7's system stack (preload + grad
accum): (1) FastGS-style SH schedule (SH0 [0,3000), SH1 [3000,15000), SH3
afterwards) and (2) Faster-GS-style densification cadence (refine every 600
instead of 100, refine_stop_iter untouched). The matrix is a 3-factor
fractional factorial: control, SH-only, densify-only, system-only (preload +
accum8), and the combined candidate higs_sched_27k.
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

ACCEL8_MATRIX = "exploration_accel8_11s0"
ACCEL8_METHODS = [
    "gsplat_27k",
    "gsplat_27k_sh_fast",
    "gsplat_27k_dens600",
    "gsplat_27k_preload_accum8",
    "higs_sched_27k",
]
PATCH_SHA256 = "4d7569adff2816f89988543dcf20603fcd65dbf71e7fce5be3de4a5aba5d271a"
TRAINER_SHA256 = "0cbf25d2be3e9d325ee4300d36a0e9dabe4159680145dd78025668aad1057ff8"


class HigsAccel8ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel8-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel8_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel8_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL8_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL8_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel8_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel8 = [job for job in plan if job["matrix"] == ACCEL8_MATRIX]
        self.assertEqual(len(accel8), 55)
        methods = Counter(job["method"] for job in accel8)
        self.assertEqual(methods, {name: 11 for name in ACCEL8_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel8))
        self.assertTrue(all(job["executable"] for job in accel8))

    def test_all_accel8_methods_27k(self):
        for method_id in ACCEL8_METHODS:
            spec = self.protocol["methods"][method_id]
            self.assertEqual(_method_iterations(self.protocol, spec), 27000)

    def test_gsplat_27k_control_contract(self):
        spec = self.protocol["methods"]["gsplat_27k"]
        self.assertEqual(spec["implementation"], "official")
        self.assertEqual(spec.get("patches"), None)
        self.assertEqual(spec["algorithm"]["max_steps"], 27000)

    def test_scheduling_method_contract(self):
        for method_id in ("gsplat_27k_sh_fast", "gsplat_27k_dens600",
                          "gsplat_27k_preload_accum8", "higs_sched_27k"):
            spec = self.protocol["methods"][method_id]
            algo = spec["algorithm"]
            self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
            self.assertEqual(algo["optimizer"], "adam_full")
            self.assertIsNone(algo["resolution_schedule"])
            self.assertIsNone(algo["tile_sampling_schedule"])
            cfg = algo["trainer_cfg"]
            self.assertFalse(cfg["packed"])
            self.assertFalse(cfg["sparse_grad"])
            self.assertFalse(cfg["visible_adam"])
            self.assertEqual(spec["patches"], ["patches/higs-accel8.patch"])
            for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
                self.assertEqual(len(spec[key]), 64)

    def test_sh_fast_lever_only(self):
        spec = self.protocol["methods"]["gsplat_27k_sh_fast"]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_sh_schedule"], "fast")
        self.assertEqual(cfg["higs_densify_every"], 0)
        self.assertFalse(cfg["higs_preload_images"])
        self.assertEqual(cfg["higs_accum_steps"], 1)

    def test_dens600_lever_only(self):
        spec = self.protocol["methods"]["gsplat_27k_dens600"]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_sh_schedule"], "")
        self.assertEqual(cfg["higs_densify_every"], 600)
        self.assertFalse(cfg["higs_preload_images"])
        self.assertEqual(cfg["higs_accum_steps"], 1)

    def test_system_lever_only(self):
        spec = self.protocol["methods"]["gsplat_27k_preload_accum8"]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_sh_schedule"], "")
        self.assertEqual(cfg["higs_densify_every"], 0)
        self.assertTrue(cfg["higs_preload_images"])
        self.assertEqual(cfg["higs_accum_steps"], 8)

    def test_combined_candidate(self):
        spec = self.protocol["methods"]["higs_sched_27k"]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_sh_schedule"], "fast")
        self.assertEqual(cfg["higs_densify_every"], 600)
        self.assertTrue(cfg["higs_preload_images"])
        self.assertEqual(cfg["higs_accum_steps"], 8)

    def test_no_sparse_window_in_any_accel8_method(self):
        for method_id in ACCEL8_METHODS:
            spec = self.protocol["methods"][method_id]
            cfg = (spec.get("algorithm") or {}).get("trainer_cfg") or {}
            ratio = cfg.get("higs_tile_sampling_ratio", 1.0)
            self.assertGreaterEqual(ratio, 1.0, method_id)

    def test_patch_hash_matches_local_patch(self):
        patch = ROOT / "patches" / "higs-accel8.patch"
        if patch.is_file():
            digest = hashlib.sha256(patch.read_bytes()).hexdigest()
            self.assertEqual(digest, PATCH_SHA256)


if __name__ == "__main__":
    unittest.main()
