"""Focused tests for the Phase-9 perceptual-loss amortization protocol.

Pre-registered exploration matrix exploration_accel9_11s0: 5 methods x 11
scenes x seed 0 = 55 jobs. accel9 attacks the single largest per-step cost in
the gsplat trainer measured by accel8's CUDA segment timing: the full-frame
gaussian-window SSIM term (~40-45% of step time at step 15k on the EPIC A100s,
via the torch conv fallback). Two orthogonal, topology-preserving knobs are
pre-registered:

1. ``higs_ssim_scale=0.5``: SSIM computed on a bilinearly downscaled
   render/GT pair (L1 stays at full resolution).
2. ``higs_ssim_every=2``: SSIM computed every 2 steps, L1-only steps in
   between.

Neither knob touches the renderer, densification, or topology, so any quality
delta vs the matched control is attributable to the perceptual-loss
amortization itself. Methods: in-matrix gsplat_27k official control, the SSIM
lever isolated at full 30k steps, the lever plus frequency at 30k, and the full
candidate (SSIM05 + every2 + preload + FastGS SH schedule) at 30k and 27k.
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

ACCEL9_MATRIX = "exploration_accel9_11s0"
ACCEL9_METHODS = [
    "gsplat_27k",
    "gsplat_30k_ssim05",
    "gsplat_30k_ssim05_ev2",
    "higs_ssim_30k",
    "higs_ssim_27k",
]
PATCH_SHA256 = "eec82b477443f835843684a769bd07572ca884575cd52c4da8f7e958c15dd014"
TRAINER_SHA256 = "39ffdb8501d1b8e4db14d8448928642d2dad5c7738a16cf68266079dc056cd4a"


class HigsAccel9ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel9-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel9_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel9_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL9_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL9_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel9_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel9 = [job for job in plan if job["matrix"] == ACCEL9_MATRIX]
        self.assertEqual(len(accel9), 55)
        methods = Counter(job["method"] for job in accel9)
        self.assertEqual(methods, {name: 11 for name in ACCEL9_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel9))
        self.assertTrue(all(job["executable"] for job in accel9))

    def test_iterations(self):
        expected = {
            "gsplat_27k": 27000,
            "gsplat_30k_ssim05": 30000,
            "gsplat_30k_ssim05_ev2": 30000,
            "higs_ssim_30k": 30000,
            "higs_ssim_27k": 27000,
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

    def test_higs_method_contract(self):
        for method_id in ("gsplat_30k_ssim05", "gsplat_30k_ssim05_ev2",
                          "higs_ssim_30k", "higs_ssim_27k"):
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
            self.assertEqual(spec["patches"], ["patches/higs-accel9.patch"])
            for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
                self.assertEqual(len(spec[key]), 64)

    def test_ssim05_lever_only(self):
        spec = self.protocol["methods"]["gsplat_30k_ssim05"]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_ssim_scale"], 0.5)
        self.assertEqual(cfg["higs_ssim_every"], 1)
        self.assertFalse(cfg["higs_preload_images"])
        self.assertEqual(cfg["higs_sh_schedule"], "")

    def test_ssim05_ev2_lever(self):
        spec = self.protocol["methods"]["gsplat_30k_ssim05_ev2"]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertEqual(cfg["higs_ssim_scale"], 0.5)
        self.assertEqual(cfg["higs_ssim_every"], 2)
        self.assertFalse(cfg["higs_preload_images"])

    def test_full_candidates(self):
        for method_id, steps in (("higs_ssim_30k", 30000), ("higs_ssim_27k", 27000)):
            spec = self.protocol["methods"][method_id]
            cfg = spec["algorithm"]["trainer_cfg"]
            self.assertEqual(cfg["higs_ssim_scale"], 0.5)
            self.assertEqual(cfg["higs_ssim_every"], 2)
            self.assertTrue(cfg["higs_preload_images"])
            self.assertEqual(cfg["higs_sh_schedule"], "fast")
            self.assertEqual(spec["algorithm"]["max_steps"], steps)

    def test_no_sparse_window_in_any_accel9_method(self):
        for method_id in ACCEL9_METHODS:
            spec = self.protocol["methods"][method_id]
            cfg = (spec.get("algorithm") or {}).get("trainer_cfg") or {}
            ratio = cfg.get("higs_tile_sampling_ratio", 1.0)
            self.assertGreaterEqual(ratio, 1.0, method_id)

    def test_patch_hash_matches_local_patch(self):
        patch = ROOT / "patches" / "higs-accel9.patch"
        if patch.is_file():
            digest = hashlib.sha256(patch.read_bytes()).hexdigest()
            self.assertEqual(digest, PATCH_SHA256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
