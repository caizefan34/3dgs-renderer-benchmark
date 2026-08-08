"""Focused tests for the pre-registered confirmatory matrix for the accel8
scheduling candidate (confirmatory_higs_sched_11s3).

132 jobs = 4 methods x 11 scenes x 3 seeds. Methods: gsplat (official 30k
control), gsplat_27k (early-stop control), gsplat_27k_preload_accum8
(system-only control), and higs_sched_27k (frozen candidate: SH-fast
schedule + densify-600 + preload + grad-accum 8). The system control makes
the HiGS-attributable margin explicit; the early-stop control prevents
attributing step-budget gains to HiGS.
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
from higs_training_commands import _method_iterations  # noqa: E402

CONFIRMATORY_MATRIX = "confirmatory_higs_sched_11s3"
CONFIRMATORY_METHODS = [
    "gsplat",
    "gsplat_27k",
    "gsplat_27k_preload_accum8",
    "higs_sched_27k",
]
PATCH_SHA256 = "4d7569adff2816f89988543dcf20603fcd65dbf71e7fce5be3de4a5aba5d271a"
TRAINER_SHA256 = "0cbf25d2be3e9d325ee4300d36a0e9dabe4159680145dd78025668aad1057ff8"


class HigsConfirmatorySchedProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-confirmatory-sched-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_confirmatory_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertEqual(report["confirmatory_jobs"], 132)
        self.assertEqual(report["executable_jobs"], 132)

    def test_confirmatory_matrix_is_frozen(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == CONFIRMATORY_MATRIX
        )
        self.assertEqual(matrix["phase"], "confirmatory")
        self.assertEqual(matrix["methods"], CONFIRMATORY_METHODS)
        self.assertEqual(matrix["matched_controls"], CONFIRMATORY_METHODS[:-1])
        self.assertEqual(matrix["scenes"], "all")
        self.assertEqual(matrix["seeds"], [0, 1, 2])
        self.assertEqual(self.protocol["frozen_candidates"], ["higs_sched_27k"])

    def test_confirmatory_plan_132_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        confirm = [job for job in plan if job["matrix"] == CONFIRMATORY_MATRIX]
        self.assertEqual(len(confirm), 132)
        methods = Counter(job["method"] for job in confirm)
        self.assertEqual(methods, {name: 33 for name in CONFIRMATORY_METHODS})
        self.assertEqual(sorted(set(job["seed"] for job in confirm)), [0, 1, 2])
        self.assertTrue(all(job["executable"] for job in confirm))

    def test_27k_methods_keep_27k_budget(self):
        for method_id in ("gsplat_27k", "gsplat_27k_preload_accum8", "higs_sched_27k"):
            spec = self.protocol["methods"][method_id]
            self.assertEqual(_method_iterations(self.protocol, spec), 27000)

    def test_gsplat_control_is_30k_official(self):
        spec = self.protocol["methods"]["gsplat"]
        self.assertEqual(spec["implementation"], "official")
        self.assertEqual(spec.get("patches"), None)
        self.assertEqual(_method_iterations(self.protocol, spec), 30000)

    def test_system_only_control_contract(self):
        spec = self.protocol["methods"]["gsplat_27k_preload_accum8"]
        algo = spec["algorithm"]
        self.assertEqual(algo["optimizer"], "adam_full")
        self.assertIsNone(algo["resolution_schedule"])
        self.assertIsNone(algo["tile_sampling_schedule"])
        cfg = algo["trainer_cfg"]
        self.assertEqual(cfg["higs_sh_schedule"], "")
        self.assertEqual(cfg["higs_densify_every"], 0)
        self.assertTrue(cfg["higs_preload_images"])
        self.assertEqual(cfg["higs_accum_steps"], 8)

    def test_candidate_contract(self):
        spec = self.protocol["methods"]["higs_sched_27k"]
        algo = spec["algorithm"]
        self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
        self.assertEqual(algo["optimizer"], "adam_full")
        self.assertIsNone(algo["resolution_schedule"])
        self.assertIsNone(algo["tile_sampling_schedule"])
        cfg = algo["trainer_cfg"]
        self.assertEqual(cfg["higs_sh_schedule"], "fast")
        self.assertEqual(cfg["higs_densify_every"], 600)
        self.assertTrue(cfg["higs_preload_images"])
        self.assertEqual(cfg["higs_accum_steps"], 8)
        self.assertFalse(cfg["visible_adam"])
        self.assertEqual(spec["patches"], ["patches/higs-accel8.patch"])
        for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
            self.assertEqual(len(spec[key]), 64)

    def test_no_sparse_window_in_any_confirmatory_method(self):
        for method_id in CONFIRMATORY_METHODS:
            spec = self.protocol["methods"][method_id]
            cfg = (spec.get("algorithm") or {}).get("trainer_cfg") or {}
            ratio = cfg.get("higs_tile_sampling_ratio", 1.0)
            self.assertGreaterEqual(ratio, 1.0, method_id)

    def test_patch_hash_matches_local_patch(self):
        patch = ROOT / "patches" / "higs-accel8.patch"
        if patch.is_file():
            digest = hashlib_sha256(patch)
            self.assertEqual(digest, PATCH_SHA256)


def hashlib_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
