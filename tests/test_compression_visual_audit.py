import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.audit_compression_visuals import (
    _sha256, compare, finalize_metrics, make_contact_sheet,
)


class CompressionVisualAuditTest(unittest.TestCase):
    def _metrics(self, root: Path, name: str, value: int) -> Path:
        render_root = root / name / "renders"
        render_root.mkdir(parents=True)
        image = np.full((16, 24, 3), value, dtype=np.uint8)
        image_path = render_root / "frame.png"
        Image.fromarray(image).save(image_path)
        raw = {
            "render_output_root": str(render_root),
            "render_outputs": [{"image": "frame.png", "path": "frame.png"}],
        }
        raw_path = root / name / "raw_samples.json"
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        metrics = {"raw_samples": {"uri": str(raw_path), "sha256": _sha256(raw_path)}}
        metrics_path = root / name / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return metrics_path

    def test_comparison_and_contact_sheet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = self._metrics(root, "reference", 100)
            candidate = self._metrics(root, "candidate", 102)
            frames, summary = compare(reference, candidate)
            sheet = root / "sheet.png"
            make_contact_sheet(frames, sheet, count=1)

            self.assertEqual(summary["frame_count"], 1)
            self.assertAlmostEqual(frames[0]["mean_abs_error"], 2 / 255)
            self.assertEqual(frames[0]["pixel_fraction_over_1_255"], 1.0)
            self.assertTrue(sheet.is_file())

    def test_finalize_metrics_records_audit_decision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics_path = root / "metrics.json"
            audit_path = root / "visual-audit.json"
            metrics_path.write_text(json.dumps({
                "metrics": {"near_lossless_gate": {
                    "numeric_pass": True, "visual_audit": "pending",
                    "overall_pass": False,
                }},
                "provenance": {},
            }), encoding="utf-8")
            audit_path.write_text(json.dumps({"decision": "pass"}), encoding="utf-8")

            finalize_metrics(metrics_path, audit_path, "pass")

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            gate = metrics["metrics"]["near_lossless_gate"]
            self.assertEqual(gate["visual_audit"], "pass")
            self.assertTrue(gate["overall_pass"])
            self.assertEqual(
                metrics["provenance"]["visual_audit"]["sha256"],
                _sha256(audit_path),
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                finalize_metrics(metrics_path, audit_path, "fail")


if __name__ == "__main__":
    unittest.main()
