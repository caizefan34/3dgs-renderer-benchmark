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

    def test_ablation_protocol_defines_pilot_matrix(self):
        report = validate_ablation_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertEqual(report["iterations"], 30000)
        self.assertEqual(report["seed_count"], 1)
        self.assertEqual(report["pilot_scene_count"], 5)
        self.assertEqual(report["planned_jobs"], 40)
        self.assertEqual(report["executable_jobs"], 40)
        self.assertIn("higs_three_stage", report["blocked_methods"])

    def test_plan_has_unique_auditable_jobs(self):
        plan = build_ablation_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(job["initialization"] == "from_scratch_sfm" for job in plan))
        self.assertTrue(all(job["iterations"] == 30000 for job in plan))
        methods = {job["method"] for job in plan}
        self.assertEqual(len(methods), 8)
        self.assertIn("higs_visible_only", methods)
        self.assertIn("higs_switch_15k", methods)
        self.assertNotIn("higs_three_stage", methods)

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


if __name__ == "__main__":
    unittest.main()
