"""Focused tests for the Phase-11 SSIM-frequency-amortization protocol.

Pre-registered exploration matrix exploration_accel11_11s0: 5 methods x 11
scenes x seed 0 = 55 jobs. accel11 amortizes the full-frame gaussian-window
SSIM term (the single largest per-step cost: ~40-46% of step time in accel10
segment timing) across training steps WITHOUT changing its resolution:
`higs_ssim_every` > 1 computes the full-resolution SSIM term only every N
steps and uses an L1-only step in between, so the perceptual objective is
unchanged (full-res, full-weight) and only the compute frequency is reduced.
`higs_ssim_scale` = 0.75 is a milder downscale than accel9's 0.5 (which passed
the speed gate at 1.41x but failed SSIM by -0.0083).

The mechanism is strictly loss-cost amortization: it does not touch the
renderer, densification, topology, or the SSIM resolution when computed.
In-matrix controls: gsplat_27k (official early-stop) for attribution, plus
one compound candidate combining accel10's post-refine backward gating with
full-res SSIM every 2 steps. All patched methods reuse the audited accel10
source tree (same patch sha256), so no second CUDA build is needed.
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
)

ACCEL11_MATRIX = "exploration_accel11_11s0"
ACCEL11_METHODS = [
    "gsplat_27k",
    "gsplat_30k_ssim_e2",
    "gsplat_30k_ssim_e3",
    "gsplat_30k_ssim075",
    "higs_skipbwd_30k_ssim_e2",
]
PATCH_SHA256 = "6ba7ad454d3f0c2749cb50b75640f882e21471ced9af159e09cd24d7be3fca00"
TRAINER_SHA256 = "0b884b5b6dc3add3fa18179acb10ddb64f630d079db6ea401304f7baeccc8fa0"


class HigsAccel11ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel11-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel11_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel11_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL11_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL11_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel11_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel11 = [job for job in plan if job["matrix"] == ACCEL11_MATRIX]
        self.assertEqual(len(accel11), 55)
        methods = Counter(job["method"] for job in accel11)
        self.assertEqual(methods, {name: 11 for name in ACCEL11_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel11))
        self.assertTrue(all(job["executable"] for job in accel11))

    def test_iterations(self):
        expected = {
            "gsplat_27k": 27000,
            "gsplat_30k_ssim_e2": 30000,
            "gsplat_30k_ssim_e3": 30000,
            "gsplat_30k_ssim075": 30000,
            "higs_skipbwd_30k_ssim_e2": 30000,
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
        self.assertEqual(spec["algorithm"]["max_steps"], 27000)

    def test_ssim_every_methods_contract(self):
        for method_id, ssim_every in (("gsplat_30k_ssim_e2", 2),
                                      ("gsplat_30k_ssim_e3", 3)):
            spec = self.protocol["methods"][method_id]
            algo = spec["algorithm"]
            self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
            self.assertEqual(algo["optimizer"], "adam_full")
            self.assertEqual(algo["max_steps"], 30000)
            cfg = algo["trainer_cfg"]
            # Full-resolution SSIM: frequency amortization only.
            self.assertEqual(cfg["higs_ssim_scale"], 1.0)
            self.assertEqual(cfg["higs_ssim_every"], ssim_every)
            self.assertFalse(cfg["higs_preload_images"])
            self.assertEqual(cfg["higs_skip_bwd_start_step"], 0)
            self.assertEqual(spec["patches"], ["patches/higs-accel10.patch"])
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)

    def test_ssim075_milder_than_accel9_scale(self):
        spec = self.protocol["methods"]["gsplat_30k_ssim075"]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_ssim_scale"], 0.75)
        self.assertEqual(cfg["higs_ssim_every"], 1)
        self.assertEqual(cfg["higs_skip_bwd_start_step"], 0)

    def test_compound_skipbwd_ssim_e2_contract(self):
        spec = self.protocol["methods"]["higs_skipbwd_30k_ssim_e2"]
        algo = spec["algorithm"]
        self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
        self.assertEqual(algo["max_steps"], 30000)
        cfg = algo["trainer_cfg"]
        self.assertEqual(cfg["higs_ssim_scale"], 1.0)
        self.assertEqual(cfg["higs_ssim_every"], 2)
        self.assertEqual(cfg["higs_skip_bwd_start_step"], 15000)
        self.assertEqual(cfg["higs_skip_bwd_min_steps"], 5)
        self.assertEqual(cfg["higs_skip_bwd_thresh"], 0.95)
        self.assertTrue(cfg["higs_segment_timing"])

    def test_no_sparse_window_in_any_accel11_method(self):
        for method_id in ACCEL11_METHODS:
            spec = self.protocol["methods"][method_id]
            cfg = (spec.get("algorithm") or {}).get("trainer_cfg") or {}
            ratio = cfg.get("higs_tile_sampling_ratio", 1.0)
            self.assertGreaterEqual(ratio, 1.0, method_id)

    def test_patch_hash_matches_local_patch(self):
        patch = ROOT / "patches" / "higs-accel10.patch"
        if patch.is_file():
            digest = hashlib.sha256(patch.read_bytes()).hexdigest()
            self.assertEqual(digest, PATCH_SHA256)


if __name__ == "__main__":
    unittest.main(verbosity=2)