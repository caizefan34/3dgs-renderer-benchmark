"""Focused tests for the Phase-4c error-guided sparse tile-sampling protocol.

Pre-registered exploration matrix exploration_accel4_11s0: 5 methods x 11 scenes
x seed 0 = 55 jobs. All HiGS candidates use the accel4 source tree
(patches/higs-accel4.patch) with error-guided tile sampling plus an unbiased
importance-sampled tile L1 (higs_sparse_loss) active ONLY in the refinement
window [15000, max_steps) so densification sees full-frame gradients while the
sampled window skips the per-step SSIM + GT imputation + sync costs.
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

ACCEL4_MATRIX = "exploration_accel4_11s0"
ACCEL4_METHODS = [
    "gsplat_27k",
    "higs_eg_sparse_phase_27k_r07",
    "higs_eg_sparse_phase_27k_r05",
    "higs_eg_sparse_phase_27k_r07_mix05",
    "higs_eg_sparse_phase_30k_r07",
]
EG_METHODS = ACCEL4_METHODS[1:]
PATCH_SHA256 = "56b2d49889ffd63b1627c65c760ccf83a406981dde177852a26fd562994a8cb6"
TRAINER_SHA256 = "1d2fac9659f02d39c42314d7aa0ec1776e283ac771cec2beaf14cc1f77fd37b5"
DENSIFY_END = 15000


class HigsAccel4ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel4-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel4_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 55)

    def test_accel4_matrix_five_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL4_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL4_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel4_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel4 = [job for job in plan if job["matrix"] == ACCEL4_MATRIX]
        self.assertEqual(len(accel4), 55)
        methods = Counter(job["method"] for job in accel4)
        self.assertEqual(methods, {name: 11 for name in ACCEL4_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel4))
        self.assertTrue(all(job["executable"] for job in accel4))

    def test_eg_method_contract(self):
        for method_id in EG_METHODS:
            spec = self.protocol["methods"][method_id]
            algo = spec["algorithm"]
            self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
            self.assertEqual(algo["optimizer"], "adam_full")
            self.assertEqual(algo["resolution_schedule"], None)
            schedule = algo["tile_sampling_schedule"]
            self.assertEqual(schedule["mode"], "phase_split_error_guided")
            self.assertEqual(schedule["sampling_mode"], "error_guided")
            self.assertEqual(schedule["start_step"], DENSIFY_END)
            self.assertTrue(schedule["sparse_loss"])
            cfg = algo["trainer_cfg"]
            self.assertFalse(cfg["packed"])
            self.assertFalse(cfg["sparse_grad"])
            self.assertFalse(cfg["visible_adam"])
            self.assertEqual(cfg["higs_tile_sampling_ratio"], schedule["ratio"])
            self.assertEqual(cfg["higs_tile_sampling_mode"], "error_guided")
            self.assertEqual(cfg["higs_tile_sampling_start_step"], DENSIFY_END)
            self.assertEqual(cfg["higs_tile_sampling_end_step"], schedule["end_step"] or 0)
            self.assertTrue(cfg["higs_sparse_loss"])
            self.assertEqual(cfg["higs_error_alpha"], schedule["alpha"])
            self.assertEqual(cfg["higs_error_lambda_mix"], schedule["lambda_mix"])
            self.assertEqual(cfg["higs_error_refresh_every"], schedule["refresh_every"])
            self.assertTrue(cfg["higs_segment_timing"])
            self.assertGreaterEqual(cfg["higs_timing_start_step"], DENSIFY_END)
            # patch identity
            self.assertEqual(spec["patches"], ["patches/higs-accel4.patch"])
            self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
            self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
            for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256"):
                self.assertEqual(len(spec[key]), 64)

    def test_eg_method_budgets(self):
        expected = {
            "higs_eg_sparse_phase_27k_r07": 27000,
            "higs_eg_sparse_phase_27k_r05": 27000,
            "higs_eg_sparse_phase_27k_r07_mix05": 27000,
            "higs_eg_sparse_phase_30k_r07": 30000,
        }
        for method_id, steps in expected.items():
            self.assertEqual(
                _method_iterations(self.protocol, self.protocol["methods"][method_id]),
                steps,
            )
        # full-budget variant: no early-stop, pure HiGS mechanism attribution
        self.assertEqual(
            _method_iterations(
                self.protocol, self.protocol["methods"]["higs_eg_sparse_phase_30k_r07"]
            ),
            30000,
        )

    def test_trainer_cfg_kwargs_passes_eg_fields(self):
        kwargs = trainer_cfg_kwargs(
            self.protocol["methods"]["higs_eg_sparse_phase_27k_r07"]
        )
        self.assertEqual(kwargs["higs_tile_sampling_ratio"], 0.7)
        self.assertEqual(kwargs["higs_tile_sampling_mode"], "error_guided")
        self.assertEqual(kwargs["higs_tile_sampling_start_step"], DENSIFY_END)
        self.assertEqual(kwargs["higs_tile_sampling_end_step"], 0)
        self.assertTrue(kwargs["higs_sparse_loss"])
        self.assertEqual(kwargs["higs_error_alpha"], 1.0)
        self.assertEqual(kwargs["higs_error_lambda_mix"], 1.0)
        self.assertEqual(kwargs["higs_error_refresh_every"], 100)

    def test_patch_file_present_with_recorded_sha(self):
        patch = ROOT / "patches" / "higs-accel4.patch"
        self.assertTrue(patch.is_file())
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        self.assertEqual(digest, PATCH_SHA256)


    def test_patch_guards_ssim_outside_sparse_branch(self):
        """Regression: the sparse branch must not fall through to the
        unconditional ``ssimloss = ssim_loss(colors_ssim, ...)`` line. Before
        the fix, that line reused the previous step's ``colors_ssim`` /
        ``pixels_ssim`` loop variables (already-backwarded graph), crashing
        with "Trying to backward through the graph a second time" at the
        first sparse step (15000)."""
        patch = (ROOT / "patches" / "higs-accel4.patch").read_text(encoding="utf-8")
        self.assertIn("if not use_sparse_loss:", patch)
        self.assertNotIn(
            "            ssimloss = ssim_loss(\n"
            "                colors_ssim.permute(0, 3, 1, 2), pixels_ssim.permute(0, 3, 1, 2)\n"
            "            )\n"
            "            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)",
            patch,
        )


if __name__ == "__main__":
    unittest.main()

