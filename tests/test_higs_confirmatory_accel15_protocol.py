"""Focused tests for the pre-registered confirmatory matrix for the frozen
accel15 candidate (confirmatory_accel15_11s3).

132 jobs = 4 methods x 11 scenes x 3 seeds. Methods: gsplat (official 30k
control), gsplat_25k (early-stop control), higs_visible_only (re-based on the
gsplat-higs-accel15 accel tree), and gsplat_30k_fused_prune10_rclip05 (frozen candidate). The matched controls
prevent attributing early-stop or baseline-code gains to the candidate.
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

CONFIRMATORY_MATRIX = "confirmatory_accel15_11s3"
CONFIRMATORY_METHODS = ["gsplat", "gsplat_25k", "higs_visible_only", "gsplat_30k_fused_prune10_rclip05"]


class HigsConfirmatoryAccel15ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark/higs-confirmatory-accel15-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_confirmatory_protocol_validates(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["scene_count"], 11)
        self.assertEqual(report["confirmatory_jobs"], 132)
        self.assertEqual(report["executable_jobs"], 132)

    def test_confirmatory_matrix_is_frozen(self):
        matrix = next(m for m in self.protocol["matrices"] if m["id"] == CONFIRMATORY_MATRIX)
        self.assertEqual(matrix["phase"], "confirmatory")
        self.assertEqual(matrix["methods"], CONFIRMATORY_METHODS)
        self.assertEqual(matrix["matched_controls"], CONFIRMATORY_METHODS[:-1])
        self.assertEqual(matrix["scenes"], "all")
        self.assertEqual(matrix["seeds"], [0, 1, 2])
        self.assertEqual(self.protocol["frozen_candidates"], ['gsplat_30k_fused_prune10_rclip05'])

    def test_confirmatory_plan_uses_interleaved_scheduler(self):
        # The A100 matrix runner must interleave matched controls and the
        # candidate per scene+seed (not plain method-major ordering), so that
        # wall-clock pairing is not biased by run order across GPUs.
        from scripts.run_higs_paper_a100_matrix import _plan_jobs  # noqa: E402

        jobs = _plan_jobs(
            self.protocol,
            set(CONFIRMATORY_METHODS),
            {CONFIRMATORY_MATRIX},
        )
        self.assertEqual(len(jobs), 132)
        # every scene+seed block contains all four methods consecutively
        idx = 0
        while idx < len(jobs):
            scene = jobs[idx]["scene"]
            seed = jobs[idx]["seed"]
            block = [job["method"] for job in jobs[idx : idx + 4]]
            self.assertEqual(
                sorted(block),
                sorted(CONFIRMATORY_METHODS),
                msg=f"non-interleaved block at {scene} s{seed}",
            )
            idx += 4
        # method order rotates per scene (balanced start order across GPUs)
        orders = set()
        for i in range(0, len(jobs), 12):  # first seed-0 block of each scene
            orders.add(tuple(job["method"] for job in jobs[i : i + 4]))
        self.assertGreater(len(orders), 1)

    def test_confirmatory_plan_132_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        confirm = [job for job in plan if job["matrix"] == CONFIRMATORY_MATRIX]
        self.assertEqual(len(confirm), 132)
        methods = Counter(job["method"] for job in confirm)
        self.assertEqual(methods, {name: 33 for name in CONFIRMATORY_METHODS})
        self.assertEqual(sorted(set(job["seed"] for job in confirm)), [0, 1, 2])
        self.assertTrue(all(job["executable"] for job in confirm))


if __name__ == "__main__":
    unittest.main()
