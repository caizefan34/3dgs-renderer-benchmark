"""Focused tests for the Phase-21 per-view SkipGS backward-gating protocol.

Pre-registered exploration matrix exploration_accel21_11s0: 6 methods x 11
scenes x seed 0 = 66 jobs. accel21 implements the SkipGS (arXiv 2603.08997)
per-view backward gating mechanism -- one running loss EMA per sampled view,
skip when surprise = loss/EMA <= 1 + thresh after a 500-step warmup, with an
auto-calibrated minimum backward ratio floor -- on top of the frozen accel15
candidate (fused-SSIM + post-15k opacity prune 0.1 + radius clip 0.5).

Difference from accel10 (which used a single scene-wide EMA baseline and
failed quality): accel21 keeps per-view statistics so views that are still
improving keep getting gradients. Mechanism only decides WHEN backward runs;
forward + loss statistics always run and densification topology is untouched.

In-matrix controls: gsplat (official 30k), gsplat_30k_fused (accel15 base),
gsplat_30k_fused_prune10_rclip05 (frozen candidate, in-matrix to isolate the
marginal speed/quality tradeoff of the gating), plus per-view gating stacked
on the frozen candidate (default thresh 0.0 and aggressive thresh 0.1) and on
the fused baseline without prune (isolate the prune interaction).
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

ACCEL21_MATRIX = "exploration_accel21_11s0"
ACCEL21_METHODS = [
    "gsplat",
    "gsplat_30k_fused",
    "gsplat_30k_fused_prune10_rclip05",
    "gsplat_30k_fused_prune10_rclip05_skipbwd_pv",
    "gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg",
    "gsplat_30k_fused_skipbwd_pv",
]
PATCH_SHA256 = "2d97a52bf5010dd58a3144dc609541e1b59168ce77b1eda144fdb79cee43064c"
TRAINER_SHA256 = "a06c2166de3d2319e9fdf3d4d4191f14341d1eef4b0985fba6d3bfa6ce2e967b"
STATE_SHA256 = "80bbe958a6fede742978aa1210d5fd7eb9cbdc0e9bcc4acc1491535b723729f7"


class HigsAccel21ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel21-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel21_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertGreaterEqual(report["executable_jobs"], 66)

    def test_accel21_matrix_six_methods_seed0(self):
        matrix = next(
            m for m in self.protocol["matrices"] if m["id"] == ACCEL21_MATRIX
        )
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL21_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["scenes"], "all")

    def test_accel21_plan_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        accel21 = [job for job in plan if job["matrix"] == ACCEL21_MATRIX]
        self.assertEqual(len(accel21), 66)
        methods = Counter(job["method"] for job in accel21)
        self.assertEqual(methods, {name: 11 for name in ACCEL21_METHODS})
        self.assertTrue(all(job["seed"] == 0 for job in accel21))
        self.assertTrue(all(job["executable"] for job in accel21))

    def test_iterations(self):
        for method_id in ACCEL21_METHODS:
            spec = self.protocol["methods"][method_id]
            self.assertEqual(
                _method_iterations(self.protocol, spec), 30000, method_id
            )

    def test_frozen_candidate_contract_unchanged(self):
        spec = self.protocol["methods"]["gsplat_30k_fused_prune10_rclip05"]
        algo = spec["algorithm"]
        self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
        cfg = algo["trainer_cfg"]
        # frozen accel15 config must be untouched by the per-view gating
        self.assertEqual(cfg["higs_prune_mode"], "opacity")
        self.assertEqual(cfg["higs_prune_opacity_thresh"], 0.1)
        self.assertEqual(cfg["higs_prune_start_step"], 15000)
        self.assertEqual(cfg["higs_radius_clip"], 0.5)
        self.assertEqual(cfg["higs_skip_bwd_start_step"], 0)
        self.assertFalse(cfg["higs_skip_bwd_per_view"])
        # rebased onto the accel21 tree
        self.assertEqual(spec["patches"], ["patches/higs-accel21.patch"])
        self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
        self.assertEqual(spec["source_state_sha256"], STATE_SHA256)

    def test_per_view_skipbwd_contract(self):
        spec = self.protocol["methods"][
            "gsplat_30k_fused_prune10_rclip05_skipbwd_pv"
        ]
        algo = spec["algorithm"]
        self.assertEqual(algo["renderer"], "higs_dynamic_native_backward")
        cfg = algo["trainer_cfg"]
        self.assertTrue(cfg["higs_skip_bwd_per_view"])
        self.assertEqual(cfg["higs_skip_bwd_start_step"], 15000)
        # SkipGS defaults from arXiv 2603.08997
        self.assertEqual(cfg["higs_skip_bwd_thresh"], 0.0)
        self.assertEqual(cfg["higs_skip_bwd_ema_decay"], 0.95)
        self.assertEqual(cfg["higs_skip_bwd_warmup"], 500)
        self.assertEqual(cfg["higs_skip_bwd_min_ratio"], 0.5)
        self.assertEqual(cfg["higs_skip_bwd_floor_min_samples"], 100)
        # prune + radius clip still on top of the frozen candidate
        self.assertEqual(cfg["higs_prune_opacity_thresh"], 0.1)
        self.assertEqual(cfg["higs_radius_clip"], 0.5)

    def test_aggressive_per_view_contract(self):
        spec = self.protocol["methods"][
            "gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg"
        ]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertTrue(cfg["higs_skip_bwd_per_view"])
        self.assertEqual(cfg["higs_skip_bwd_thresh"], 0.1)
        self.assertEqual(cfg["higs_skip_bwd_start_step"], 15000)

    def test_per_view_on_fused_without_prune(self):
        spec = self.protocol["methods"]["gsplat_30k_fused_skipbwd_pv"]
        cfg = spec["algorithm"]["trainer_cfg"]
        self.assertTrue(cfg["higs_skip_bwd_per_view"])
        self.assertEqual(cfg["higs_skip_bwd_start_step"], 15000)
        # fused baseline keeps the gentle default prune (opacity 0.02)
        self.assertEqual(cfg["higs_prune_opacity_thresh"], 0.02)

    def test_no_sparse_pixel_sampling_in_any_accel21_method(self):
        for method_id in ACCEL21_METHODS:
            spec = self.protocol["methods"][method_id]
            cfg = (spec.get("algorithm") or {}).get("trainer_cfg") or {}
            self.assertNotIn("higs_tile_sampling_ratio", cfg, method_id)

    def test_patch_hash_matches_local_patch(self):
        patch = ROOT / "patches" / "higs-accel21.patch"
        if patch.is_file():
            digest = hashlib.sha256(patch.read_bytes()).hexdigest()
            self.assertEqual(digest, PATCH_SHA256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
