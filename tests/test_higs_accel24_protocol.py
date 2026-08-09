"""Focused tests for the accel24 error-guided densification protocol.

Pre-registered exploration matrix exploration_accel24_11s0: 6 methods x 11
scenes x seed 0 = 66 jobs. accel24 tests Revising-Densification-style
error-guided densification stacked on the frozen accel15 candidate
gsplat_30k_fused_prune10_rclip05 (new audited tree gsplat-higs-accel24 +
patch 7b9d4169). Before each DefaultStrategy refine pass the trainer injects
per-tile reconstruction-error signal into the running grad2d/count state:
Gaussians whose projected 2D mean falls in the top-error tiles are boosted
above the grow threshold ("hybrid" keeps gradient-selected Gaussians; "pure"
zeroes non-selected accumulated gradients). Only the strategy running state
is touched; renderer, loss, and optimizer are unchanged.

Methods: gsplat / gsplat_30k_fused / frozen candidate in-matrix /
+errdens_hybrid (ratio 0.3) / +errdens_pure (ratio 0.3) /
+errdens_hybrid_r05 (ratio 0.5).
"""
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from higs_ablation_protocol import (  # noqa: E402
    build_ablation_experiment_plan,
    validate_ablation_protocol,
)

ACCEL24_MATRIX = "exploration_accel24_11s0"
ACCEL24_METHODS = [
    "gsplat",
    "gsplat_30k_fused",
    "gsplat_30k_fused_prune10_rclip05",
    "gsplat_30k_fused_prune10_rclip05_errdens_hybrid",
    "gsplat_30k_fused_prune10_rclip05_errdens_pure",
    "gsplat_30k_fused_prune10_rclip05_errdens_hybrid_r05",
]
PATCH_SHA256 = "7b9d4169f2f695793a66723e5a1789c961ff2c942d699a5c5790e2988e69678b"
TRAINER_SHA256 = "24847bfe34dfd73ca7121b85f85bba18fbf4357a466efb2ecf2f54ae19a83630"
PATCH_SHA_ACCEL15 = "04f950778f617295cf28611487a017d081dd4dcc5f056560d60c598a5eafa5e7"
TRAINER_SHA_ACCEL15 = "12bb76d2a7d4f9640badfa2c58f93f262572b3cd74a7b1d769bf169248c8a7f5"
SOURCE_STATE_SHA256 = "f7bf953607426085d45377b68e1905018d9f2ce958841d3b35c24b4d81797d8f"
COMMIT = "77ab983ffe43420b2131669cb35776b883ca4c3c"


def _patch_added_trainer_lines():
    """Yield the added (\"+\") lines of the simple_trainer.py diff section."""
    patch = (ROOT / "patches" / "higs-accel24.patch").read_text(
        encoding="utf-8", errors="replace"
    )
    in_trainer = False
    for line in patch.splitlines():
        if line.startswith("diff --git a/examples/simple_trainer.py"):
            in_trainer = True
            continue
        if in_trainer and line.startswith("diff --git "):
            break
        if in_trainer and line.startswith("+"):
            yield line[1:]


def _extract_def(added_lines, name):
    """Extract a function's source from patch-added lines by indentation."""
    idx = next(
        i for i, l in enumerate(added_lines) if l.strip().startswith("def %s(" % name)
    )
    def_line = added_lines[idx]
    indent = len(def_line) - len(def_line.lstrip())
    out_lines = [def_line[indent:]]
    for l in added_lines[idx + 1:]:
        if l.strip() == "":
            out_lines.append("")
            continue
        if (len(l) - len(l.lstrip())) <= indent:
            break
        out_lines.append(l[indent:])
    return "\n".join(out_lines) + "\n"


def _load_higs_helpers():
    added = list(_patch_added_trainer_lines())
    ns = {"torch": torch}
    exec(_extract_def(added, "_higs_tile_mean_errors"), ns)
    exec(_extract_def(added, "_higs_error_densify_inject"), ns)
    return ns["_higs_error_densify_inject"], ns["_higs_tile_mean_errors"]


class HigsAccel24ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-accel24-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_accel24_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["executable_jobs"], 66)
        self.assertEqual(report["planned_jobs"], 66)

    def test_accel24_matrix_shape(self):
        matrix = next(m for m in self.protocol["matrices"] if m["id"] == ACCEL24_MATRIX)
        self.assertEqual(matrix["phase"], "exploration")
        self.assertEqual(matrix["methods"], ACCEL24_METHODS)
        self.assertEqual(matrix["seeds"], [0])
        self.assertEqual(matrix["hardware"], ["a100"])

    def test_accel24_hashes_pinned(self):
        patch_bytes = (ROOT / "patches" / "higs-accel24.patch").read_bytes()
        self.assertEqual(hashlib.sha256(patch_bytes).hexdigest(), PATCH_SHA256)
        for method in ACCEL24_METHODS[1:]:
            spec = self.protocol["methods"][method]
            if "errdens" in method:
                self.assertEqual(spec["commit"], COMMIT)
                self.assertEqual(spec["patch_sha256"], PATCH_SHA256)
                self.assertEqual(spec["trainer_sha256"], TRAINER_SHA256)
                self.assertEqual(spec["source_state_sha256"], SOURCE_STATE_SHA256)
                self.assertEqual(spec["patches"], ["patches/higs-accel24.patch"])
            else:
                # in-matrix controls reuse the audited accel15 tree
                self.assertEqual(spec["patch_sha256"], PATCH_SHA_ACCEL15)
                self.assertEqual(spec["trainer_sha256"], TRAINER_SHA_ACCEL15)

    def test_accel24_errdens_wired(self):
        for method, mode, ratio in [
            ("gsplat_30k_fused_prune10_rclip05_errdens_hybrid", "hybrid", 0.3),
            ("gsplat_30k_fused_prune10_rclip05_errdens_pure", "pure", 0.3),
            ("gsplat_30k_fused_prune10_rclip05_errdens_hybrid_r05", "hybrid", 0.5),
        ]:
            tc = self.protocol["methods"][method]["algorithm"]["trainer_cfg"]
            self.assertEqual(tc["higs_error_densify_mode"], mode)
            self.assertAlmostEqual(tc["higs_error_densify_ratio"], ratio)
            self.assertEqual(tc["higs_error_densify_boost"], 2.0)
            self.assertEqual(tc["higs_error_densify_tile"], 16)
            self.assertEqual(tc["higs_error_densify_start_step"], 500)
            # stacked on the frozen candidate: prune/rclip/full-res unchanged
            self.assertEqual(tc["higs_prune_opacity_thresh"], 0.10)
            self.assertEqual(tc["higs_radius_clip"], 0.5)
            self.assertEqual(tc["higs_train_res_scale"], 1.0)
            self.assertEqual(tc["higs_ssim_scale"], 1.0)

    def test_accel24_controls_unchanged(self):
        c = self.protocol["methods"]["gsplat_30k_fused_prune10_rclip05"][
            "algorithm"]["trainer_cfg"]
        self.assertEqual(c.get("higs_error_densify_mode", ""), "")
        self.assertEqual(c["higs_prune_opacity_thresh"], 0.10)
        self.assertEqual(c["higs_radius_clip"], 0.5)

    def test_accel24_plan_jobs(self):
        plan = [j for j in build_ablation_experiment_plan(self.protocol)
                if j["matrix"] == ACCEL24_MATRIX]
        self.assertEqual(len(plan), 66)
        for job in plan:
            self.assertIn(job["method"], ACCEL24_METHODS)
            self.assertEqual(job["seed"], 0)


class HigsErrorDensifyLogicTest(unittest.TestCase):
    """Unit tests on the real trainer method extracted from the patch."""

    @classmethod
    def setUpClass(cls):
        cls.helpers = {
            "inject": _load_higs_helpers()[0],
            "tile_err": _load_higs_helpers()[1],
        }

    def _harness(self, mode="hybrid", ratio=0.3, boost=2.0, step=1000,
                 refine_every=100, refine_stop=15000, n_gauss=512,
                 h=64, w=64):
        torch.manual_seed(0)
        grow = 2e-4
        cfg = types.SimpleNamespace(
            higs_error_densify_mode=mode,
            higs_error_densify_ratio=ratio,
            higs_error_densify_boost=boost,
            higs_error_densify_tile=16,
            higs_error_densify_start_step=500,
            strategy=types.SimpleNamespace(
                refine_every=refine_every,
                refine_stop_iter=refine_stop,
                grow_grad2d=grow,
            ),
        )
        # colors: white frame with a black patch in the top-left 16x16 tile
        colors = torch.full((1, h, w, 3), 0.9)
        colors[:, 0:16, 0:16, :] = 0.1
        pixels = torch.full((1, h, w, 3), 0.9)
        # means2d: N gaussians spread across the frame; the first 64 in the
        # top-left tile (high error), the rest in a low-error tile (bottom-right)
        means2d = torch.zeros((1, n_gauss, 2))
        means2d[0, :64, 0] = torch.arange(64).float() % 16
        means2d[0, :64, 1] = torch.arange(64).float() % 16
        means2d[0, 64:, 0] = w - 1
        means2d[0, 64:, 1] = h - 1
        state = {
            "grad2d": torch.full((n_gauss,), 1e-5),
            "count": torch.full((n_gauss,), 100.0),
        }
        info = {"means2d": means2d}
        self_ = types.SimpleNamespace(cfg=cfg, strategy_state=state)
        return self_, colors, pixels, info, grow

    def test_hybrid_boosts_error_selected_only(self):
        self_, colors, pixels, info, grow = self._harness(mode="hybrid")
        before = self_.strategy_state["grad2d"].clone()
        self.helpers["inject"](self_, info, colors, pixels, 1000)
        grad2d = self_.strategy_state["grad2d"]
        count = self_.strategy_state["count"]
        # first 64 gaussians are in the high-error top-left tile -> boosted
        selected = grad2d[:64]
        self.assertTrue(torch.all(selected >= 2.0 * grow * 100.0))
        # others keep their previous (low) value -> still below threshold
        self.assertTrue(torch.all(grad2d[64:] == before[64:]))
        self.assertTrue(torch.all(grad2d[64:] < 2.0 * grow))
        # count untouched in hybrid
        self.assertTrue(torch.all(count == 100.0))

    def test_pure_zeroes_non_selected(self):
        self_, colors, pixels, info, grow = self._harness(mode="pure")
        self.helpers["inject"](self_, info, colors, pixels, 1000)
        grad2d = self_.strategy_state["grad2d"]
        count = self_.strategy_state["count"]
        self.assertTrue(torch.all(grad2d[:64] >= 2.0 * grow * 100.0))
        self.assertTrue(torch.all(grad2d[64:] == 0.0))
        self.assertTrue(torch.all(count[64:] == 1e6))
        self.assertTrue(torch.all(count[:64] == 100.0))

    def test_ratio_controls_selection_size(self):
        self_, colors, pixels, info, grow = self._harness(mode="hybrid", ratio=0.5)
        self.helpers["inject"](self_, info, colors, pixels, 1000)
        grad2d = self_.strategy_state["grad2d"]
        # larger ratio still selects the error tile (top-left is top-k for any
        # k >= 1/16 tiles); low-error tile remains unselected
        self.assertTrue(torch.all(grad2d[:64] >= 2.0 * grow * 100.0))
        self.assertTrue(torch.all(grad2d[64:] < 2.0 * grow))

    def test_guards_are_noops(self):
        # disabled mode
        self_, colors, pixels, info, grow = self._harness(mode="")
        before = self_.strategy_state["grad2d"].clone()
        self.helpers["inject"](self_, info, colors, pixels, 1000)
        self.assertTrue(torch.equal(self_.strategy_state["grad2d"], before))
        # before start step
        self_, colors, pixels, info, grow = self._harness(mode="hybrid")
        before = self_.strategy_state["grad2d"].clone()
        self.helpers["inject"](self_, info, colors, pixels, 100)
        self.assertTrue(torch.equal(self_.strategy_state["grad2d"], before))
        # non-refine step
        self_, colors, pixels, info, grow = self._harness(
            mode="hybrid", refine_every=37
        )
        before = self_.strategy_state["grad2d"].clone()
        self.helpers["inject"](self_, info, colors, pixels, 1000)
        self.assertTrue(torch.equal(self_.strategy_state["grad2d"], before))
        # past refine stop
        self_, colors, pixels, info, grow = self._harness(
            mode="hybrid", refine_stop=1000
        )
        before = self_.strategy_state["grad2d"].clone()
        self.helpers["inject"](self_, info, colors, pixels, 1000)
        self.assertTrue(torch.equal(self_.strategy_state["grad2d"], before))
        # missing state
        self_, colors, pixels, info, grow = self._harness(mode="hybrid")
        self_.strategy_state = None
        self.helpers["inject"](self_, info, colors, pixels, 1000)  # must not raise

    def test_tile_mean_errors_border(self):
        frame = torch.zeros((1, 18, 20, 3))
        ref = torch.ones((1, 18, 20, 3))
        err = self.helpers["tile_err"](frame, ref, 16)
        # 2x2 tiles; bottom/right border tiles use real pixel counts
        self.assertEqual(err.shape, (1, 2, 2))
        self.assertAlmostEqual(err[0, 0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(err[0, 0, 1].item(), 1.0, places=5)
        self.assertAlmostEqual(err[0, 1, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(err[0, 1, 1].item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
