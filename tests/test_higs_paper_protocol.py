import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_paper_protocol import HigsPaperProtocolError, build_experiment_plan, validate_protocol


class HigsPaperProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "benchmark" / "higs-paper-protocol.json"
        cls.protocol = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_repository_protocol_defines_top_conference_matrix(self):
        report = validate_protocol(self.protocol)
        self.assertEqual(report["initialization"], "from_scratch_sfm")
        self.assertGreaterEqual(report["iterations"], 30000)
        self.assertEqual(report["seed_count"], 3)
        self.assertEqual(report["primary_scene_count"], 11)
        self.assertIn("higs_proposed", report["primary_methods"])
        self.assertIn("original_3dgs", report["primary_methods"])
        self.assertIn("gsplat", report["primary_methods"])
        self.assertGreater(report["planned_jobs"], 200)
        self.assertIn("consumer", report["hardware_classes"])

    def test_rejects_checkpoint_initialization_and_proxy_headline_baseline(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["training"]["initialization"] = "pretrained_ply"
        with self.assertRaisesRegex(HigsPaperProtocolError, "from_scratch_sfm"):
            validate_protocol(protocol)

        protocol = copy.deepcopy(self.protocol)
        protocol["methods"]["speedy_splat"]["implementation"] = "style_proxy"
        with self.assertRaisesRegex(HigsPaperProtocolError, "proxy"):
            validate_protocol(protocol)

    def test_plan_has_unique_auditable_jobs(self):
        plan = build_experiment_plan(self.protocol)
        ids = [job["job_id"] for job in plan]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(job["initialization"] == "from_scratch_sfm" for job in plan))
        self.assertTrue(all(job["iterations"] == 30000 for job in plan))

    def test_patched_methods_require_source_locks(self):
        protocol = copy.deepcopy(self.protocol)
        del protocol["methods"]["higs_full"]["source_diff_sha256"]
        with self.assertRaisesRegex(HigsPaperProtocolError, "source locks"):
            validate_protocol(protocol)

    def test_ready_non_reference_runner_requires_smoke_evidence(self):
        protocol = copy.deepcopy(self.protocol)
        del protocol["methods"]["gsplat"]["runner_evidence"]
        with self.assertRaisesRegex(HigsPaperProtocolError, "runner evidence"):
            validate_protocol(protocol)

        protocol = copy.deepcopy(self.protocol)
        protocol["methods"]["gsplat"]["commit"] = "0" * 40
        with self.assertRaisesRegex(HigsPaperProtocolError, "source mismatch"):
            validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
