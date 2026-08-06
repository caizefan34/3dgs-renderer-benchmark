"""Validate paper/claims.yaml against the frozen claim schema.

Guards the claim manifest so every statement maps to real artifacts and
the canonical protocol hash matches benchmark/suite.json. Run with:

    python -m unittest tests.test_paper_claims_manifest -v
"""

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "paper" / "claims.yaml"
SUITE_PATH = ROOT / "benchmark" / "suite.json"

VALID_CATEGORIES = {
    "throughput",
    "compression",
    "trainability",
    "training-speed",
    "methodology",
    "diagnostics",
    "packaging",
    "negative-result",
}
VALID_STATUSES = {"measured", "exploratory"}

REQUIRED_CLAIM_FIELDS = {
    "id",
    "category",
    "status",
    "statement",
    "scope",
    "evidence",
    "statistics",
    "limitations",
}


class PaperClaimsManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CLAIMS_PATH.open(encoding="utf-8") as handle:
            cls.manifest = yaml.safe_load(handle)
        with SUITE_PATH.open(encoding="utf-8") as handle:
            cls.suite = json.load(handle)

    def test_schema_version(self):
        self.assertEqual(self.manifest["schema_version"], 1)

    def test_protocol_hash_matches_suite(self):
        self.assertEqual(
            self.manifest["protocol"]["sha256"],
            self.suite["protocol_sha256"],
        )

    def test_claim_ids_are_unique_and_well_formed(self):
        ids = [claim["id"] for claim in self.manifest["claims"]]
        self.assertEqual(len(ids), len(set(ids)))
        for claim_id in ids:
            self.assertRegex(claim_id, r"^C-\d{3}$")

    def test_claims_have_required_fields_and_valid_enums(self):
        for claim in self.manifest["claims"]:
            self.assertTrue(
                REQUIRED_CLAIM_FIELDS.issubset(claim),
                msg=f"{claim['id']} is missing fields: "
                f"{REQUIRED_CLAIM_FIELDS - set(claim)}",
            )
            self.assertIn(claim["category"], VALID_CATEGORIES)
            self.assertIn(claim["status"], VALID_STATUSES)
            self.assertIsInstance(claim["evidence"], list)
            self.assertIsInstance(claim["limitations"], list)
            self.assertGreaterEqual(len(claim["evidence"]), 1)

    def test_every_evidence_path_exists(self):
        for claim in self.manifest["claims"]:
            for evidence in claim["evidence"]:
                path = ROOT / evidence
                self.assertTrue(
                    path.exists(),
                    msg=f"{claim['id']}: evidence path missing: {evidence}",
                )

    def test_freeze_status_and_gates_are_present(self):
        self.assertIn(self.manifest["freeze_status"], {"draft", "frozen"})
        self.assertIsInstance(self.manifest["required_gates"], list)
        self.assertGreaterEqual(len(self.manifest["required_gates"]), 1)

    def test_exploratory_claims_declare_limitations(self):
        for claim in self.manifest["claims"]:
            if claim["status"] == "exploratory":
                self.assertGreaterEqual(
                    len(claim["limitations"]), 1,
                    msg=f"{claim['id']} is exploratory but has no limitations",
                )


if __name__ == "__main__":
    unittest.main()