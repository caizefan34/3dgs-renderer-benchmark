import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_suite import (
    BENCHMARK_SUITE_VERSION,
    load_benchmark_suite,
    resolve_suite_case,
    validate_file_hash,
)


class BenchmarkSuiteTest(unittest.TestCase):
    def test_committed_suite_defines_scenes_cameras_and_resolutions(self):
        suite = load_benchmark_suite()

        self.assertEqual(suite["version"], BENCHMARK_SUITE_VERSION)
        self.assertEqual(set(suite["resolution_profiles"]), {"720p", "1080p", "4k"})
        self.assertEqual({scene["scene_id"] for scene in suite["scenes"]}, {
            "garden", "bicycle", "room"
        })
        for scene in suite["scenes"]:
            self.assertEqual(len(scene["dataset_sha256"]), 64)
            self.assertEqual(len(scene["scene_sha256"]), 64)
            self.assertEqual(len(scene["camera_sha256"]), 64)

    def test_resolves_a_fixed_suite_case(self):
        case = resolve_suite_case("garden", "1080p", verify_assets=False)

        self.assertEqual(case["suite_case_id"], "garden@1080p")
        self.assertEqual(case["resolution"], [1920, 1080])
        self.assertTrue(case["official"])

    def test_hash_validation_rejects_changed_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"official")
            digest = hashlib.sha256(b"official").hexdigest()
            validate_file_hash(path, digest, "asset")
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "asset SHA-256 mismatch"):
                validate_file_hash(path, digest, "asset")


if __name__ == "__main__":
    unittest.main()
