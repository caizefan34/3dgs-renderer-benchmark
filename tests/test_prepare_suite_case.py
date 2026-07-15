import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from prepare_suite_case import build_camera_candidate, canonical_asset_hashes


class PrepareSuiteCaseTest(unittest.TestCase):
    def test_suite_pins_every_canonical_case(self):
        suite = json.loads((ROOT / "benchmark" / "suite.json").read_text(encoding="utf-8"))
        self.assertEqual(suite["version"], "3.1.0")
        for case in suite["cases"]:
            canonical = case["canonical_assets"]
            self.assertEqual(canonical["status"], "pinned")
            for key in ("checkpoint_sha256", "camera_sha256", "ground_truth_manifest_sha256"):
                self.assertRegex(canonical[key], r"^[0-9a-f]{64}$")

    def test_camera_candidate_is_ordered_cropped_and_deterministic(self):
        source = [
            {
                "id": index,
                "img_name": name,
                "width": 4000,
                "height": 3000,
                "fx": 2000.0,
                "fy": 2000.0,
                "position": [0.0, 0.0, float(index)],
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
            for index, name in enumerate(("c", "a", "b"))
        ]
        image_sizes = {name: (1000, 750) for name in ("a", "b", "c")}

        payload, selected = build_camera_candidate(source, image_sizes, 2, 1920, 1080)
        decoded = json.loads(payload)

        self.assertEqual([row["img_name"] for row in decoded], ["a", "c"])
        self.assertEqual(selected, ["a", "c"])
        self.assertEqual(decoded[0]["reference_crop"], [0, 94, 1000, 656])
        self.assertEqual((decoded[0]["width"], decoded[0]["height"]), (1920, 1080))
        self.assertAlmostEqual(decoded[0]["fx"], 960.0)
        self.assertAlmostEqual(decoded[0]["fy"], 960.0)
        self.assertEqual(payload, build_camera_candidate(source, image_sizes, 2, 1920, 1080)[0])

    def test_unpinned_canonical_assets_fail_closed(self):
        actual = canonical_asset_hashes(b"ply", b"[]\n", [{"image": "a.jpg", "sha256": "1" * 64}])
        with self.assertRaisesRegex(ValueError, "unpublished"):
            canonical_asset_hashes(
                b"ply", b"[]\n", [{"image": "a.jpg", "sha256": "1" * 64}],
                expected={"status": "unpublished"},
            )
        self.assertEqual(len(actual["checkpoint_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
