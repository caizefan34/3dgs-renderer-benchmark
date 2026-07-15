import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.official import load_official_dataset_manifest, validate_training_manifest
from renderers.candidate_registry import load_renderer_candidates, validate_renderer_candidates


class OfficialTrainingPolicyTest(unittest.TestCase):
    def test_committed_official_dataset_manifest_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        manifest = load_official_dataset_manifest(
            root / "data" / "datasets" / "official_training_datasets.json"
        )

        report = validate_training_manifest(manifest)

        self.assertEqual(report["status"], "ok")
        self.assertIn("Mip-NeRF 360", report["official_dataset_families"])
        self.assertTrue(all(job.endswith("_eval") for job in report["validated_training_jobs"]))

    def test_training_jobs_must_use_eval_split(self):
        manifest = {
            "schema_version": 1,
            "policy": {"training_data_rule": "Official datasets only"},
            "official_sources": [
                {"dataset_family": "Mip-NeRF 360", "used_by_official_3dgs": True}
            ],
            "training_jobs": [
                {
                    "job_id": "bad_job",
                    "dataset_family": "Mip-NeRF 360",
                    "train_command": "python train.py -s data/official/garden",
                }
            ],
        }

        with self.assertRaises(ValueError):
            validate_training_manifest(manifest)

    def test_renderer_candidate_registry_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        registry = load_renderer_candidates(
            root / "data" / "renderers" / "renderer_candidates.json"
        )

        report = validate_renderer_candidates(registry)

        self.assertEqual(report["status"], "ok")
        self.assertGreaterEqual(report["candidate_count"], 8)


if __name__ == "__main__":
    unittest.main()

