import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_program import ResearchProgramError, validate_research_program


class ResearchProgramTest(unittest.TestCase):
    def _manifest(self, root: Path, name: str, track_id: str) -> Path:
        path = root / name
        path.write_text(json.dumps({
            "schema_version": "1.0",
            "track_id": track_id,
            "paper_scope": f"Scope for {track_id}.",
            "contributions": [{"id": "method", "statement": "A method."}],
            "claims": [{
                "id": "future-result",
                "contribution_id": "method",
                "statement": "A future result.",
                "status": "blocked",
                "blocking_gate": "Run the experiment.",
            }],
        }), encoding="utf-8")
        return path

    def test_validates_distinct_tracks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = [
                self._manifest(root, "survey.json", "survey"),
                self._manifest(root, "higs.json", "higs"),
                self._manifest(root, "compression.json", "compression"),
            ]
            summary = validate_research_program(manifests)
        self.assertEqual(set(summary), {"survey", "higs", "compression"})
        self.assertEqual(summary["higs"]["blocked"], 1)

    def test_rejects_duplicate_track_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = [
                self._manifest(root, "one.json", "higs"),
                self._manifest(root, "two.json", "higs"),
            ]
            with self.assertRaisesRegex(ResearchProgramError, "duplicate track_id"):
                validate_research_program(manifests)

    def test_repository_tracks_are_current(self):
        manifests = [
            ROOT / "paper" / "survey-claims.json",
            ROOT / "paper" / "higs-claims.json",
            ROOT / "paper" / "compression-claims.json",
        ]
        summary = validate_research_program(manifests, repository_root=ROOT)
        self.assertEqual(set(summary), {"survey", "higs", "compression"})
        self.assertGreaterEqual(summary["higs"]["blocked"], 2)
        self.assertGreaterEqual(summary["compression"]["supported"], 1)


if __name__ == "__main__":
    unittest.main()
