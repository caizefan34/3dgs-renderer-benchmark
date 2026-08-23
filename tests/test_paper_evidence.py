import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_evidence import PaperEvidenceError, validate_paper_evidence


class PaperEvidenceTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        evidence = root / "evidence.json"
        evidence.write_text(json.dumps({"rows": [
            {"id": "a", "score": 2.0, "pass": True},
            {"id": "b", "score": 3.0, "pass": True},
        ]}), encoding="utf-8")
        digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
        manifest = root / "claims.json"
        manifest.write_text(json.dumps({
            "schema_version": "1.0",
            "paper_scope": "A fixed benchmark cohort.",
            "contributions": [
                {"id": "protocol", "statement": "A protocol."},
                {"id": "results", "statement": "Measured results."},
            ],
            "claims": [{
                "id": "measured-result",
                "contribution_id": "results",
                "statement": "Two passing rows score from 2 to 3.",
                "status": "supported",
                "evidence": [{
                    "path": "evidence.json",
                    "sha256": digest,
                    "assertions": [
                        {"type": "value_equals", "pointer": "/rows/0/id", "equals": "a"},
                        {"type": "select_count", "pointer": "/rows", "where": {}, "equals": 2},
                        {"type": "select_all", "pointer": "/rows", "where": {}, "field": "/pass", "equals": True},
                        {"type": "select_min", "pointer": "/rows", "where": {}, "field": "/score", "equals": 2.0},
                        {"type": "select_max", "pointer": "/rows", "where": {}, "field": "/score", "equals": 3.0},
                    ],
                }],
            }, {
                "id": "future-result",
                "contribution_id": "results",
                "statement": "Cross-host result.",
                "status": "blocked",
                "blocking_gate": "Run on a second host.",
            }],
        }), encoding="utf-8")
        return manifest

    def test_validates_supported_and_blocked_claims(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = validate_paper_evidence(self._fixture(Path(temp_dir)))
        self.assertEqual(summary, {"supported": 1, "blocked": 1, "out_of_scope": 0})

    def test_rejects_hash_mismatch_and_more_than_three_contributions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self._fixture(root)
            (root / "evidence.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(PaperEvidenceError, "SHA-256 mismatch"):
                validate_paper_evidence(manifest_path)

            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            document["contributions"] = [
                {"id": str(index), "statement": "x"} for index in range(4)
            ]
            manifest_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(PaperEvidenceError, "at most three"):
                validate_paper_evidence(manifest_path)

    def test_repository_claim_manifest_is_current(self):
        summary = validate_paper_evidence(ROOT / "paper" / "claims.json", repository_root=ROOT)
        self.assertGreaterEqual(summary["supported"], 3)
        self.assertGreaterEqual(summary["blocked"], 2)

        claims = json.loads((ROOT / "paper" / "claims.json").read_text(encoding="utf-8"))
        higs = next(claim for claim in claims["claims"] if claim["id"] == "higs-training-method")
        self.assertNotIn("not currently a Git-tracked", higs["rationale"])


if __name__ == "__main__":
    unittest.main()
