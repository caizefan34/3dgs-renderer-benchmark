"""Focused tests for the Phase-10 SkipGS-style backward-gating protocol.

Pre-registered exploration matrix exploration_accel10_11s0: 5 methods x 11
scenes x seed 0 = 55 jobs. accel10 gates the backward pass in the
post-densification window (step >= higs_skip_bwd_start_step): a view whose
current loss is below `higs_skip_bwd_thresh` x the running EMA baseline skips
loss.backward() (and the optimizer step), with a hard budget of at least one
backward every `higs_skip_bwd_min_steps` steps so the model cannot starve.

The mechanism is strictly timing-based: it does not touch the renderer, the
loss (SSIM/L1 stay full-resolution, every step), densification, or topology.
accel8's CUDA segment timing showed backward is 39-42% of step time in the
post-refine window; skipping a share of those backwards is the speed lever.
Quality protection comes from skipping only well-fit views (loss below the
EMA baseline), not from changing the objective. In-matrix controls: gsplat_27k
(official early-stop) and gsplat_30k_ssim05 (accel9 reference lever) for
attribution.
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

ACCEL10_MATRIX = "exploration_accel10_11s0"
ACCEL10_METHODS = [
    "gsplat_27k",
    "gsplat_30k_ssim05",
    "higs_skipbwd_30k",
    "higs_skipbwd_30k_agg",
    "higs_skipbwd_27k",
]
PATCH_SHA256 = "6ba7ad454d3f0c2749cb50b75640f882e21471ced9af159e09cd24d7be3fca00"
TRAINER_SHA256 = "0b884b5b6dc3add3fa18179acb10ddb64f630d079db6ea401304f7baeccc8fa0"


class HigsAccel10ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel10-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel10_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel10_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL10_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL10_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel10_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel10 = [job for job in plan if job["matrix"] == ACCEL10_MATRIX]
        self.assertEqual(len(accel10), 55)
        methods = Counter(job["method"] for job in accel10)
        self.assertEqual(methods, {name: 11 for name in ACCEL10_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel10))
        self.assertTrue(all(job["executable"] for job in accel10))

    def test_iterations(self):
        expected = {
            "gsplat_27k": 27000,
            "gsplat_30k_ssim05": 30000,
            "higs_skipbwd_30k": 30000,
            "higs_skipbwd_30k_agg": 30000,
            "higs_skipbwd_27k": 27000,
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

    def test_skipbwd_method_contract(self):
        for method_id in ("higs_skipbwd_30k", "higs_skipbwd_30k_agg",
                          "higs_skipbwd_27k"):
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
            self.assertEqual(cfg["higs_accum_steps"], 1)
            self.assertEqual(cfg["higs_densify_every"], 0)
            # Full-resolution, every-step SSIM: backward gating must be the
            # only speed lever, so any quality delta is attributable to it.
            self.assertEqual(cfg["higs_ssim_scale"], 1.0)
            self.assertEqual(cfg["higs_ssim_every"], 1)
            self.assertFalse(cfg["higs_preload_images"])
            self.assertEqual(cfg["higs_sh_schedule"], "")
            self.assertEqual(cfg["higs_skip_bwd_start_step"], 15000)
            self.assertEqual(spec["patches"], ["patches/higs-accel10.patch"])
            for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
                self.assertEqual(len(spec[key]), 64)

    def test_skipbwd_gating_params(self):
        cfg30 = self.protocol["methods"]["higs_skipbwd_30k"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg30["higs_skip_bwd_min_steps"], 5)
        self.assertEqual(cfg30["higs_skip_bwd_thresh"], 0.95)
        self.assertEqual(cfg30["higs_skip_bwd_ema_alpha"], 0.1)
        cfgagg = self.protocol["methods"]["higs_skipbwd_30k_agg"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfgagg["higs_skip_bwd_min_steps"], 3)
        self.assertEqual(cfgagg["higs_skip_bwd_thresh"], 0.90)
        cfg27 = self.protocol["methods"]["higs_skipbwd_27k"]["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg27["higs_skip_bwd_min_steps"], 5)
        self.assertEqual(cfg27["higs_skip_bwd_thresh"], 0.95)

    def test_segment_timing_window_in_candidates(self):
        for method_id in ("higs_skipbwd_30k", "higs_skipbwd_30k_agg",
                          "higs_skipbwd_27k"):
            cfg = self.protocol["methods"][method_id]["algorithm"]["trainer_cfg"]
            self.assertTrue(cfg["higs_segment_timing"])
            self.assertEqual(cfg["higs_timing_start_step"], 15000)
            self.assertEqual(cfg["higs_timing_end_step"], 15300)

    def test_no_sparse_window_in_any_accel10_method(self):
        for method_id in ACCEL10_METHODS:
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