import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_ablation_protocol import (  # noqa: E402
    HigsAblationProtocolError,
    build_ablation_experiment_plan,
    validate_ablation_protocol,
)


class HigsAblationProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-ablation-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_ablation_protocol_defines_pilot_and_confirmatory_matrices(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["seed_count"], 3)
        self.assertEqual(report["scene_count"], 11)
        self.assertEqual(report["planned_jobs"], 205)
        self.assertEqual(report["executable_jobs"], 205)
        self.assertEqual(report["confirmatory_jobs"], 165)
        self.assertIn("higs_three_stage", report["blocked_methods"])

    def test_plan_has_unique_auditable_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(job["initialization"] == "from_scratch_sfm" for job in plan))
        self.assertTrue(all(job["iterations"] == 30000 for job in plan))
        methods = {job["method"] for job in plan}
        self.assertEqual(len(methods), 9)
        self.assertIn("higs_visible_only", methods)
        self.assertIn("higs_switch_15k", methods)
        self.assertIn("gsplat", methods)
        self.assertNotIn("higs_three_stage", methods)
        matrices = {job["matrix"] for job in plan}
        self.assertEqual(matrices, {"ablation_pilot_30k", "confirmatory_formal_30k"})

    def test_confirmatory_matrix_uses_frozen_candidates_and_matched_controls(self):
        confirmatory = next(
            m for m in self.protocol["matrices"]
            if m["id"] == "confirmatory_formal_30k"
        )
        self.assertEqual(confirmatory["phase"], "confirmatory")
        self.assertEqual(
            confirmatory["methods"],
            ["gsplat", "higs_full", "higs_current", "higs_switch_12k", "higs_switch_21k"],
        )
        self.assertEqual(self.protocol["frozen_candidates"], ["higs_switch_12k", "higs_switch_21k"])
        self.assertEqual(
            confirmatory["matched_controls"],
            ["gsplat", "higs_full", "higs_current"],
        )
        jobs = [
            job for job in build_ablation_experiment_plan(self.protocol)
            if job["matrix"] == "confirmatory_formal_30k"
        ]
        self.assertEqual(len(jobs), 5 * 11 * 3)
        self.assertEqual(len({job["seed"] for job in jobs}), 3)
        self.assertEqual(len({job["scene"] for job in jobs}), 11)

    def test_confirmatory_matrix_rejects_non_frozen_candidates(self):
        protocol = copy.deepcopy(self.protocol)
        confirmatory = next(
            m for m in protocol["matrices"]
            if m["id"] == "confirmatory_formal_30k"
        )
        confirmatory["methods"][-1] = "higs_switch_18k"
        with self.assertRaisesRegex(HigsAblationProtocolError, "confirmatory methods"):
            validate_ablation_protocol(protocol)

    def test_requires_core_ablation_methods(self):
        protocol = copy.deepcopy(self.protocol)
        del protocol["methods"]["higs_visible_only"]
        with self.assertRaisesRegex(HigsAblationProtocolError, "higs_visible_only"):
            validate_ablation_protocol(protocol)

    def test_rejects_unknown_matrix_member(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["matrices"][0]["methods"].append("unknown_method")
        with self.assertRaisesRegex(HigsAblationProtocolError, "unknown matrix member"):
            validate_ablation_protocol(protocol)

    def test_patched_methods_require_source_locks(self):
        protocol = copy.deepcopy(self.protocol)
        del protocol["methods"]["higs_full"]["source_state_sha256"]
        with self.assertRaisesRegex(HigsAblationProtocolError, "SHA-256 source locks"):
            validate_ablation_protocol(protocol)

    def test_blocked_method_requires_blocking_gate(self):
        protocol = copy.deepcopy(self.protocol)
        del protocol["methods"]["higs_three_stage"]["blocking_gate"]
        with self.assertRaisesRegex(HigsAblationProtocolError, "blocking_gate"):
            validate_ablation_protocol(protocol)

    def test_rejects_short_budget_and_missing_a100(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["training"]["iterations"] = 3000
        with self.assertRaisesRegex(HigsAblationProtocolError, "30000"):
            validate_ablation_protocol(protocol)

        protocol = copy.deepcopy(self.protocol)
        del protocol["hardware"]["a100"]
        with self.assertRaisesRegex(HigsAblationProtocolError, "a100"):
            validate_ablation_protocol(protocol)

    def test_shorttail_protocol_plans_per_method_budgets(self):
        path = ROOT / "benchmark" / "higs-shorttail-protocol.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        report = validate_ablation_protocol(protocol)
        self.assertEqual(report["status"], "ablation_protocol_ready")
        plan = build_ablation_experiment_plan(protocol)
        shorttail = [job for job in plan if job["matrix"] == "exploration_shorttail_11s0"]
        self.assertEqual(len(shorttail), 44)
        by_method = {job["method"]: job["iterations"] for job in shorttail}
        self.assertEqual(by_method["gsplat_25k"], 25000)
        self.assertEqual(by_method["higs_visible_24k"], 24000)
        self.assertEqual(by_method["higs_visible_25k"], 25000)
        self.assertEqual(by_method["higs_visible_27k"], 27000)

    def test_rejects_out_of_range_per_method_max_steps(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["methods"]["higs_visible_only"]["algorithm"]["max_steps"] = 40000
        with self.assertRaisesRegex(HigsAblationProtocolError, "max_steps"):
            validate_ablation_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
