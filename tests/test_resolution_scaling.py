import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.generate_resolution_scaling import records_from_document


class ResolutionScalingTest(unittest.TestCase):
    def test_extracts_one_record_per_renderer(self):
        document = {
            "scene": {"scene_id": "garden"},
            "protocol": {"resolution_name": "720p", "resolution": [1280, 720]},
            "results": {
                "higs": {
                    "renderer": "gsplat_higs_auto",
                    "mean_fps": 100.0,
                    "mean_latency_ms": 10.0,
                    "p99_latency_ms": 11.0,
                    "peak_vram_mb": 500.0,
                    "stability_score": 0.9,
                }
            },
        }

        records = records_from_document(document, "result.json")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["scene_id"], "garden")
        self.assertEqual(records[0]["resolution_name"], "720p")
        self.assertEqual(records[0]["width"], 1280)
