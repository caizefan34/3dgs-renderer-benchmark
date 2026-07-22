import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.analyze_temporal_sequence import analyze


class TemporalSequenceTest(unittest.TestCase):
    def test_temporal_residual_detects_wrong_frame_delta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            render_root, gt_root = root / "renders", root / "gt"
            render_root.mkdir(); gt_root.mkdir()
            names = ["a.png", "b.png"]
            for index, name in enumerate(names):
                Image.fromarray(np.full((8, 8, 3), 20 + index * 10, dtype=np.uint8)).save(gt_root / name)
                Image.fromarray(np.full((8, 8, 3), 20 + index * 12, dtype=np.uint8)).save(render_root / name)
            raw = {
                "render_output_root": str(render_root),
                "render_outputs": [{"image": name, "path": name} for name in names],
                "quality_frames": [{"image": name} for name in names],
            }
            raw_path = root / "raw.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            metrics = {
                "schema_version": "2.0", "result_id": "test", "evidence_tier": "measured",
                "renderer": {"id": "gsplat", "config_id": "gsplat"},
                "benchmark": {"case_id": "case"},
                "metrics": {"raw_samples": {"uri": str(raw_path), "sha256": raw_sha}},
            }
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

            result = analyze(metrics_path, gt_root)

        self.assertEqual(result["sequence"]["transition_count"], 1)
        self.assertAlmostEqual(result["metrics"]["mean_temporal_residual"], 2 / 255)
        self.assertGreater(result["metrics"]["temporal_delta_psnr_db"], 40)


if __name__ == "__main__":
    unittest.main()
