import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from release_bundle import build_release_bundle


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ReleaseBundleTest(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        claims = {
            "schema_version": "1.0",
            "paper_scope": "fixture",
            "contributions": [{"id": "c1", "statement": "s"}],
            "claims": [{
                "id": "claim-1",
                "contribution_id": "c1",
                "statement": "evidence exists",
                "status": "supported",
                "evidence": [{
                    "path": "evidence.json",
                    "sha256": "",
                    "assertions": [],
                }],
            }],
        }
        evidence = {"value": 1}
        evidence_digest = hashlib.sha256(json.dumps(evidence).encode()).hexdigest()
        claims["claims"][0]["evidence"][0]["sha256"] = evidence_digest
        for relative, text in {
            "README.md": "# Fixture\n",
            "LICENSE": "MIT\n",
            "CITATION.cff": "cff-version: 1.2.0\n",
            "benchmark/suite.json": "{}",
            "benchmark/protocol.json": "{}",
            "docs/methodology.md": "# Methodology\n",
            "docs/protocol.md": "# Protocol\n",
            "docs/leaderboard/ranking.json": "{}",
            "docs/leaderboard/ranking.md": "# Ranking\n",
            "reports/generated/compression-expanded-final/compression-results.json": "{}",
            "paper/README.md": "# Paper evidence\n",
            "paper/claims.json": json.dumps(claims),
            "evidence.json": json.dumps(evidence),
        }.items():
            _write(self.root / relative, text)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.root, check=True)
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=self.root, check=True)

    def tearDown(self):
        self._temp.cleanup()

    def test_bundle_is_deterministic_and_self_verifying(self):
        first = self.root / "first.zip"
        second = self.root / "second.zip"
        first_summary = build_release_bundle(self.root, first)
        second_summary = build_release_bundle(self.root, second)

        self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(),
                         hashlib.sha256(second.read_bytes()).hexdigest())
        self.assertEqual(first_summary, second_summary)
        self.assertGreaterEqual(first_summary["file_count"], 8)

        with zipfile.ZipFile(first) as archive:
            names = set(archive.namelist())
            self.assertIn("artifact-manifest.json", names)
            self.assertIn("paper/claims.json", names)
            manifest = json.loads(archive.read("artifact-manifest.json"))
            for item in manifest["files"]:
                self.assertIn(item["path"], names)
                payload = archive.read(item["path"])
                self.assertEqual(len(payload), item["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()